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

  num_particles = 120;
  default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i) {
    Particle p{};
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;

    particles.push_back(p);
    // initialize all weights to 1.0
    weights.push_back(1.0);
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

  // normal distributions for sensor noise
  default_random_engine gen;
  normal_distribution<double> noise_x(0, std_pos[0]);
  normal_distribution<double> noise_y(0, std_pos[1]);
  normal_distribution<double> noise_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++) {

    // calculate new state
    if (fabs(yaw_rate) < 0.00001) {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    // add noise
    particles[i].x += noise_x(gen);
    particles[i].y += noise_y(gen);
    particles[i].theta += noise_theta(gen);
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

  for (unsigned int i = 0; i < observations.size(); i++) {

    // grab current observation
    auto observation = observations[i];
    double min_distance = numeric_limits<double>::max();
    int landmark_id = -1;

    // find the landmark id for the current observation
    for (size_t j = 0; j < predicted.size(); j++) {
      auto prediction = predicted[j];
      double temp = dist(observation.x, observation.y, prediction.x, prediction.y);

      if (temp < min_distance) {
        min_distance = temp;
        landmark_id = prediction.id;
      }
    }

    observations[i].id = landmark_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

  for (int i = 0; i < num_particles; i++) {

    double px = particles[i].x;
    double py = particles[i].y;
    double ptheta = particles[i].theta;

    vector<LandmarkObs> predictions;

    for (size_t j = 0; j < map_landmarks.landmark_list.size(); j++) {

      int landmark_id = map_landmarks.landmark_list[j].id_i;
      float landmark_x = map_landmarks.landmark_list[j].x_f;
      float landmark_y = map_landmarks.landmark_list[j].y_f;

      if (fabs(landmark_x - px) <= sensor_range && fabs(landmark_y - py) <= sensor_range) {

        predictions.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y});
      }
    }

    //coordinate system tranformation, vehicle -> map
    vector<LandmarkObs> transformed_observations;
    for (unsigned int j = 0; j < observations.size(); j++) {
      double temp_x = cos(ptheta) * observations[j].x - sin(ptheta) * observations[j].y + px;
      double temp_y = sin(ptheta) * observations[j].x + cos(ptheta) * observations[j].y + py;
      transformed_observations.push_back(LandmarkObs{observations[j].id, temp_x, temp_y});
    }

    // data association
    dataAssociation(predictions, transformed_observations);

    //update weight
    particles[i].weight = 1.0;
    for (size_t j = 0; j < transformed_observations.size(); j++) {
      double observed_x = transformed_observations[j].x, observed_y = transformed_observations[j].y;

      // find the coordinates of the predicted landmark associated with the current observation
      double predicted_x, predicted_y;
      for (size_t k = 0; k < predictions.size(); k++) {
        if (predictions[k].id == transformed_observations[j].id) {
          predicted_x = predictions[k].x;
          predicted_y = predictions[k].y;
          break;
        }
      }

      // weight calculation using multi-varaiate Gaussian
      double delta_x = observed_x - predicted_x;
      double delta_y = observed_y - predicted_y;
      double std_x = std_landmark[0], std_y = std_landmark[1];
      double weight = (1.0/(2.0*M_PI*std_x*std_y))
                      * exp(-(pow(delta_x, 2) / 2.0 /pow(std_x, 2) + pow(delta_y, 2)/2.0/pow(std_y, 2)));

      // product of this obersvation weight with total observations weight
      particles[i].weight *= weight;
    }
  }
}

void ParticleFilter::resample() {
  // discrete_distribution does not support
  // random number generation with weight of type 'double'
  // adopt the method introduced by Sebastian in the lecture
  default_random_engine gen;
  vector<Particle> resampled_particles;

  // get all the weights and find the max
  vector<double> weights;
  double max_weight = numeric_limits<double>::min();
  for (int i = 0; i < num_particles; i++) {
    max_weight = max_weight < particles[i].weight ? particles[i].weight : max_weight;
    weights.push_back(particles[i].weight);
  }

  uniform_int_distribution<int> uniintdist(0, num_particles - 1);
  auto index = uniintdist(gen);

  // uniform random distribution [0.0, 2 * max_weight)
  uniform_real_distribution<double> unirealdist(0.0, 2.0 * max_weight);

  double beta = 0.0;
  for (int i = 0; i < num_particles; i++) {
    beta += unirealdist(gen);
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampled_particles.push_back(particles[index]);
  }

  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
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
