# Writeup
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Rubric

Rubric is specified [here](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/f1820894-8322-4bb3-81aa-b26b3c6dcbaf/lessons/e8235395-22dd-4b87-88e0-d108c5e5bbf4/concepts/6a4d8d42-6a04-4aa6-b284-1697c0fd6562).

Goal of this project is to:

1. Build a PID controller to drive the car successfully around the track in a simulated environment.
2. Cross-trak error and velocity of the vehicle will be input.
3. Describe the effects of P, I and D components of the controller.
4. Record a small video and describe what each component does.
5. Describe how the final values of P, I and D coefficients are chosen.

### Result

| [![PID controller](https://img.youtube.com/vi/ScqaSfJm7qw/0.jpg)](https://www.youtube.com/watch?v=ScqaSfJm7qw "PID controller") | [![Non-zero P value](https://img.youtube.com/vi/YpzMn57sZ1A/0.jpg)](https://www.youtube.com/watch?v=YpzMn57sZ1A "Non-zero P value") | [![Non-zero D value](https://img.youtube.com/vi/A_u7ClWXVQ0/0.jpg)](https://www.youtube.com/watch?v=A_u7ClWXVQ0 "Non-zero D value") | [![Non-zero I value](https://img.youtube.com/vi/2AJ4ZtSlLtk/0.jpg)](https://www.youtube.com/watch?v=2AJ4ZtSlLtk "Non-zero I value") |
|:--:|:--:|:--:|:--:|
| *Full PID controller* | *Non-zero P* | *Non-zero D* | *Non-zero I* |

### Implementation
I chose the hyperparameters by some trial and error experiments. I observed that the crosstrek error was mostly within `[-5, 5]` range and since the steering needs to be in the range `[-1, 1]`, the `Kp` value has to be in the order of `10^-1` magnitude. 

I calculate the differential error using the time elapsed between the two observations `(dcte/dt)`, so the order of magnitude of `Kd` value would be the same as that of `Kp`.

Lastly, I calulate the integral error differently. Instead of simply adding on the current crosstrek error to the integral term, I discount the previously accumulated integral term so as not to let it blow up as the time passes. Since I receive an observation rougly every `0.1` second, the `Kp` value should be `1/0.1=10` time smaller than the `Kp` value (if at all there is any bias in the mechanics of the vehicle, otherwise this value can be `0`).

After a little trial and experimentation, I settled on the values of `-0.3, -0.15, -0.001`.

### Reflection
I have recorded videos with only one of the `Kp`, `Kd` or `Ki` parameter set to show the effect of each variable on driving.

In the [video](https://www.youtube.com/watch?v=YpzMn57sZ1A) where only `Kp` is non-zero, it can be seen that the `Kp` parameter does a good job of steering the vehicle back to the center of the lane, however, in the abses of `Kd` parameter, it overshoots and thus oschillates.

In the [video](https://www.youtube.com/watch?v=A_u7ClWXVQ0) where only `Kd` is non-zero, it can be see that the differential term does do a good job of steering the vehicle back to the center when crosstrek error is large and removes the oscillations when it reaches there, but it is not as effective on the turns.

In the [video](https://www.youtube.com/watch?v=2AJ4ZtSlLtk) where only `Ki` is non-zero, the vehicle is clearly lost without the strong steering guidance that is provided by `Kp` and `Kd` terms. `Ki` would mostly be effective in the case of a bias in the mechanics of the car and it does not seem that the vehicle has any bias.

## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1(mac, linux), 3.81(Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `./install-mac.sh` or `./install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Simulator. You can download these from the [project intro page](https://github.com/udacity/self-driving-car-sim/releases) in the classroom.

There's an experimental patch for windows in this [PR](https://github.com/udacity/CarND-PID-Control-Project/pull/3)

## Basic Build Instructions
1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./pid`. 
