/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	if (is_initialized)
		return;

	num_particles = 10;
	default_random_engine gen;
  	normal_distribution<double> d(0.0, 1.0);

	for (int i=0; i<num_particles; i++) {
		Particle p = {
			i,
			x+d(gen)*std[0],
			y+d(gen)*std[1],
			theta+d(gen)*std[2],
			1.0
		};
		particles.push_back(p);
		weights.push_back(1.0);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;
  	normal_distribution<double> d(0.0, 1.0);

	for (int i=0; i<num_particles; i++) {
		double theta = particles[i].theta;
		if (fabs(yaw_rate)<1e-3) {
			particles[i].x += velocity*delta_t*cos(theta);
			particles[i].y += velocity*delta_t*sin(theta);
		} else {
			particles[i].x += velocity/yaw_rate*( sin(theta+yaw_rate*delta_t)-sin(theta));
			particles[i].y += velocity/yaw_rate*(-cos(theta+yaw_rate*delta_t)+cos(theta));
			particles[i].theta += yaw_rate*delta_t;
			particles[i].theta += d(gen)*std_pos[2]*delta_t;
		}

		particles[i].x += d(gen)*std_pos[0]*delta_t;
		particles[i].y += d(gen)*std_pos[1]*delta_t;
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i=0; i<observations.size(); i++) {
		double min_dist = 1e99;
		double x_car = observations[i].x;
		double y_car = observations[i].y;
		for (int j=0; j<predicted.size(); j++) {
			double x_predicted = predicted[j].x;
			double y_predicted = predicted[j].y;
			double distance = dist(x_car, y_car, x_predicted, y_predicted);
			if (distance<min_dist) {
				min_dist = distance;
				observations[i].id = j;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (int i=0; i<num_particles; i++) {
		double x_particle = particles[i].x;
		double y_particle = particles[i].y;
		double theta = particles[i].theta;

		// Predict sensor measurements if they were taken by this particle
		vector<LandmarkObs> observationsPredicted;
		for (int j=0; j<map_landmarks.landmark_list.size(); j++) {
			double x_landmark = map_landmarks.landmark_list[j].x_f;
			double y_landmark = map_landmarks.landmark_list[j].y_f;
			int landmark_id = map_landmarks.landmark_list[j].id_i;

			double distance = dist(x_particle, y_particle, x_landmark, y_landmark);
			if (distance<sensor_range) {
				observationsPredicted.push_back({
					landmark_id,
					x_landmark,
					y_landmark
				});
			}
		}

		// Transform car observations into map coordinates (as if they were taken by the current particle)
		std::vector<LandmarkObs> observationsTransformed;
		for (int j=0; j<observations.size(); j++) {
			double x_car = observations[j].x;
			double y_car = observations[j].y;

			double x = x_particle + (cos(theta)*x_car-sin(theta)*y_car);
			double y = y_particle + (sin(theta)*x_car+cos(theta)*y_car);
			observationsTransformed.push_back({
				-1,
				x,
				y
			});
		}

		dataAssociation(observationsPredicted, observationsTransformed);

		double weight = 1.0;
		for (int j=0; j<observationsTransformed.size(); j++) {
			double x = observationsTransformed[j].x;
			double y = observationsTransformed[j].y;
			int landmark_id = observationsTransformed[j].id;

			double x_landmark = observationsPredicted[landmark_id].x;
			double y_landmark = observationsPredicted[landmark_id].y;

			double dx = (x-x_landmark);
			double dy = (y-y_landmark);
			double sig_x = std_landmark[0];
			double sig_y = std_landmark[1];

			double exponent = (dx*dx)/(sig_x*sig_x) + (dy*dy)/(sig_y*sig_y);
			double denominator = 2*M_PI*sig_x*sig_y;
			double prob = exp(-exponent)/denominator;
			weight *= prob;
		}

		particles[i].weight = weight;
		weights[i] = weight;
	}

	double sum = 0.0;
	for (int i=0; i<num_particles; i++)
		sum += weights[i];

	for (int i=0; i<num_particles; i++) {
		particles[i].weight /= sum;
		weights[i] /= sum;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;
	discrete_distribution<int> d {weights.begin(), weights.end()};

	vector<Particle> new_particles;
	for (int i=0; i<num_particles; i++) {
		int j = d(gen);
		Particle p = {
			i,
			particles[j].x,
			particles[j].y,
			particles[j].theta,
			particles[j].weight
		};
		new_particles.push_back(p);
		weights[i] = particles[j].weight;
	}

	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
