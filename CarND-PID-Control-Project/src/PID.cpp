#include "PID.h"
#include <time.h> 

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
	this->Kp = Kp;
	this->Ki = Ki;
	this->Kd = Kd;

	p_error = 0;
	i_error = 0;
	d_error = 0;
}

void PID::UpdateError(double cte) {
	d_error = cte - p_error;
	p_error = cte;

	if (previous_time<0)
		previous_time = 0;
	double current_time = clock();
	dt = (current_time - previous_time)/CLOCKS_PER_SEC;
	previous_time = current_time;
	i_error = i_error + cte*dt;
}

double PID::TotalError() {	
	return Kp*p_error + Kd*d_error/dt + Ki*i_error;
}

