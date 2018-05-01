
# coding: utf-8

# In[1]:


import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# In[2]:


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

KEEP_PROB = .9
LEARNING_RATE = 1e-4
EPOCHS = 20
BATCH_SIZE = 2


# In[3]:


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3, layer4, layer7)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return (image_input, keep_prob, layer3, layer4, layer7)
tests.test_load_vgg(load_vgg, tf)


# In[4]:


def layers(layer3, layer4, layer7, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param layer3: TF Tensor for VGG Layer 3 output
    :param layer4: TF Tensor for VGG Layer 4 output
    :param layer7: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3)
    kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d()
    
    # First freeze the VGG-16 model encoder layers
    # We are only going to train the decoder layers
    # in the interest of time and quick iteration
    layer3 = tf.stop_gradient(layer3)
    layer4 = tf.stop_gradient(layer4)
    layer7 = tf.stop_gradient(layer7)
    
    # The model below is replicated from the original FCN-8 model by the authors of
    # https://arxiv.org/abs/1411.4038
    # The model is described here 
    # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn8s-atonce/net.py
    # Note especially the hardcoded values of scaling of pool3 and pool4 layers
    # before fusing and susequent upsampling
    
    # Now the decoder part begins
    # First perform 1x1 convolutions on frozen layer3, layer4 and layer7 tensors
    layer3_1x1 = tf.layers.conv2d(layer3, num_classes, 1, padding='SAME', 
                                  kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer,
                                  name='layer3_conv_1x1')
    layer4_1x1 = tf.layers.conv2d(layer4, num_classes, 1, padding='SAME', 
                                  kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer,
                                  name='layer4_conv_1x1')
    layer7_1x1 = tf.layers.conv2d(layer7, num_classes, 1, padding='SAME', 
                                  kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer,
                                  name='fc7_conv_1x1')
    
    # 2x upsample
    fuse = layer7_1x1
    output = tf.layers.conv2d_transpose(fuse, num_classes, 4, 2, padding='SAME', 
                                        kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer,
                                        name='fc7_conv_transpose')
    # Fuse with layer4_1x1
    scale_factor = 1e-2
    fuse = output + layer4_1x1*scale_factor
    
    # 2x upsample
    output = tf.layers.conv2d_transpose(fuse, num_classes, 4, 2, padding='SAME', 
                                        kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer, 
                                        name='layer4_conv_transpose')
    # Fuse with layer3_1x1
    scale_factor = 1e-4
    fuse = output + layer3_1x1*scale_factor
    
    # 8x upsample
    output = tf.layers.conv2d_transpose(fuse, num_classes, 16, 8, padding='SAME', 
                                        kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer, 
                                        name='layer3_conv_transpose')

    return output
tests.test_layers(layers)


# In[5]:


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    with tf.name_scope('optimizer') as scope:
        logits = tf.reshape(nn_last_layer, [-1, num_classes])
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
        reg_losses = []
        for scope in ['fc7_conv_1x1', 'fc7_conv_transpose',
                      'layer4_conv_1x1', 'layer4_conv_transpose',
                      'layer3_conv_1x1', 'layer3_conv_transpose']:
            for reg_loss in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope):
                cross_entropy_loss += reg_loss
        
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(cross_entropy_loss)
        
        return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


# In[6]:


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """    
    for epoch in range(epochs):
        losses = []
        for images, labels in get_batches_fn(batch_size):
            loss, _ = sess.run([cross_entropy_loss, train_op], 
                               feed_dict={input_image:images, 
                                          correct_label:labels, 
                                          keep_prob:KEEP_PROB,
                                          learning_rate:LEARNING_RATE})
            losses.append(loss)
        loss = sum(losses)/len(losses)
        print('Epoch: {:3} Loss: {:.4E}'.format(epoch, loss))
tests.test_train_nn(train_nn)


# In[7]:


def run():
    num_classes = 2    
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        output = layers(layer3, layer4, layer7, num_classes)
        
        correct_label = tf.placeholder(tf.int32)
        learning_rate = tf.placeholder(tf.float32)
        logits, train_op, cross_entropy_loss = optimize(output, correct_label, learning_rate, num_classes)
                
        # Initialize the variables of the decoder and optimizer
        variables = []
        for scope in ['fc7_conv_1x1', 'fc7_conv_transpose',
                      'layer4_conv_1x1', 'layer4_conv_transpose',
                      'layer3_conv_1x1', 'layer3_conv_transpose', 
                      'optimizer']:
            variables.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        init = tf.variables_initializer(variables)
        _ = sess.run(init)
        
        # Train NN using the train_nn function
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, image_input,
                 correct_label, keep_prob, learning_rate)
                    
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video


# In[8]:


if __name__ == '__main__':
    run()

