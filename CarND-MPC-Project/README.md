# CarND-Controls-MPC
Self-Driving Car Engineer Nanodegree Program

---

## Project Instructions

More information is only accessible by people who are already enrolled in Term 2
of CarND. If you are enrolled, see [the project page](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/f1820894-8322-4bb3-81aa-b26b3c6dcbaf/lessons/b1ff3be0-c904-438e-aad3-2b5379f0e0c3/concepts/1a2255a0-e23c-44cf-8d41-39b8a3c8264a)
for instructions and the project rubric.


## Rubrics

### 1. Compilation
Following the build instructions below will compile the code successfully.

### 2. Implementation
#### The Model
MPC model is described in [MPC.cpp](src/MPC.cpp). The model, given the state of the vehicle and a 3rd degree fitted polynomial to the waypoints, predicts a trajectory for the vehicle over a short time horizon (0.5 sec) using a non-linear, constrait optimizer.

The cost function is made up of different parts:

1. Sum of squared errors `cte`, `espi` and difference between velocity and reference velocity
2. Sum of squared actuator inputs (`delta` and `a`)
3. Sum of squared actuator differences from one time step to the next (from `t` to `t+dt`)

The above cost function makes sure that not only primary errors `cte` and `epsi` are minimized, but the vehicle drive is robust and smooth.

The different weightings were found by trail and experimentation.

#### State
We keep track of vehicles pose `(x, y, psi)`, longitudinal velocity `v` and errors - Cross Track error `cte` and heading angle error `epsi`. We keep track of errors because they are main componenets in the cost function of the optimizer.

#### Timestep Length and Elapsed Duration (N & dt)
We chose a small time horizon of `0.5 sec` (`N=10` steps, `dt=0.05` sec). Small value of `dt` helps turn the vehicle on sharp turns (as the vehicle can keep track of sharper turn rate).

#### Polynomial Fitting and MPC Preprocessing
We convert the waypoints from the global coordinate system to the vehicle's one (makes it easier to calculate Cross Track error) by a sequence of translation and rotation with the resultant x-axis pointing in the direction of vehicle's heading. Vehicle state is modified accordingly.

A 3rd degree polynomial is fitted to the waypoints to better track the reference trajectory.

The fitted polynomial coefficients and the state is passed to the optimizer.

Following are the state update equations:

```
Lf = wheelbase

State:
state = (x, y, phi, v, cte, epsi)

Actuators:
delta = steering
a = accelaration

Pose variables:
x_next = x + v*cos(psi)*dt
y_next = y + v*sin(psi)*dt
psi_next = psi + v/Lf*delta*dt
v_next = v + a*dt

Error variables:
cte = f(x) - y
cte_next = cte + v*sin(epsi)*dt

psi_desired = arctan(f'(x))
epsi = psi - psi_desired
epsi_next = epsi + v/Lf*delta*dt
```

#### Model Predictive Control with Latency
There is a latency of `100ms` between the actuator input and actual execution. To combat this, we forward the pose of the model by the latency time duration before passing it on to the solver. 

### 3. Simulation
The vehicle drive successfully around the track with a reference speed of 40mph.


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
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.

* **Ipopt and CppAD:** Please refer to [this document](https://github.com/udacity/CarND-MPC-Project/blob/master/install_Ipopt_CppAD.md) for installation instructions.
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page). This is already part of the repo so you shouldn't have to worry about it.
* Simulator. You can download these from the [releases tab](https://github.com/udacity/self-driving-car-sim/releases).
* Not a dependency but read the [DATA.md](./DATA.md) for a description of the data sent back from the simulator.


## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./mpc`.
