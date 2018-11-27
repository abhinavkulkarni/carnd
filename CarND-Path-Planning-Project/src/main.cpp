#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "spline.h"

using namespace std;

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

double distance(double x1, double y1, double x2, double y2)
{
	return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}
int ClosestWaypoint(double x, double y, const vector<double> &maps_x, const vector<double> &maps_y)
{

	double closestLen = 100000; //large number
	int closestWaypoint = 0;

	for(int i = 0; i < maps_x.size(); i++)
	{
		double map_x = maps_x[i];
		double map_y = maps_y[i];
		double dist = distance(x,y,map_x,map_y);
		if(dist < closestLen)
		{
			closestLen = dist;
			closestWaypoint = i;
		}

	}

	return closestWaypoint;

}

int NextWaypoint(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y)
{

	int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);

	double map_x = maps_x[closestWaypoint];
	double map_y = maps_y[closestWaypoint];

	double heading = atan2((map_y-y),(map_x-x));

	double angle = fabs(theta-heading);
  angle = min(2*pi() - angle, angle);

  if(angle > pi()/4)
  {
    closestWaypoint++;
  if (closestWaypoint == maps_x.size())
  {
    closestWaypoint = 0;
  }
  }

  return closestWaypoint;
}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y)
{
	int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y);

	int prev_wp;
	prev_wp = next_wp-1;
	if(next_wp == 0)
	{
		prev_wp  = maps_x.size()-1;
	}

	double n_x = maps_x[next_wp]-maps_x[prev_wp];
	double n_y = maps_y[next_wp]-maps_y[prev_wp];
	double x_x = x - maps_x[prev_wp];
	double x_y = y - maps_y[prev_wp];

	// find the projection of x onto n
	double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
	double proj_x = proj_norm*n_x;
	double proj_y = proj_norm*n_y;

	double frenet_d = distance(x_x,x_y,proj_x,proj_y);

	//see if d value is positive or negative by comparing it to a center point

	double center_x = 1000-maps_x[prev_wp];
	double center_y = 2000-maps_y[prev_wp];
	double centerToPos = distance(center_x,center_y,x_x,x_y);
	double centerToRef = distance(center_x,center_y,proj_x,proj_y);

	if(centerToPos <= centerToRef)
	{
		frenet_d *= -1;
	}

	// calculate s value
	double frenet_s = 0;
	for(int i = 0; i < prev_wp; i++)
	{
		frenet_s += distance(maps_x[i],maps_y[i],maps_x[i+1],maps_y[i+1]);
	}

	frenet_s += distance(0,0,proj_x,proj_y);

	return {frenet_s,frenet_d};

}

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, const vector<double> &maps_s, const vector<double> &maps_x, const vector<double> &maps_y)
{
	int prev_wp = -1;

	while(s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1) ))
	{
		prev_wp++;
	}

	int wp2 = (prev_wp+1)%maps_x.size();

	double heading = atan2((maps_y[wp2]-maps_y[prev_wp]),(maps_x[wp2]-maps_x[prev_wp]));
	// the x,y,s along the segment
	double seg_s = (s-maps_s[prev_wp]);

	double seg_x = maps_x[prev_wp]+seg_s*cos(heading);
	double seg_y = maps_y[prev_wp]+seg_s*sin(heading);

	double perp_heading = heading-pi()/2;

	double x = seg_x + d*cos(perp_heading);
	double y = seg_y + d*sin(perp_heading);

	return {x,y};

}

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;
  while (getline(in_map_, line)) {
  	istringstream iss(line);
  	double x;
  	double y;
  	float s;
  	float d_x;
  	float d_y;
  	iss >> x;
  	iss >> y;
  	iss >> s;
  	iss >> d_x;
  	iss >> d_y;
  	map_waypoints_x.push_back(x);
  	map_waypoints_y.push_back(y);
  	map_waypoints_s.push_back(s);
  	map_waypoints_dx.push_back(d_x);
  	map_waypoints_dy.push_back(d_y);
  }

  double car_v = 0;
  int car_lane = 1;
  h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy, &car_v, &car_lane](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    //auto sdata = string(data).substr(0, length);
    //cout << sdata << endl;

    double ref_acc = .224;
    

    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
        	// Main car's localization Data
          	double car_x = j[1]["x"];
          	double car_y = j[1]["y"];
          	double car_s = j[1]["s"];
          	double car_d = j[1]["d"];
          	double car_yaw = j[1]["yaw"];
          	double car_speed = j[1]["speed"];

          	// Previous path data given to the Planner
          	auto previous_path_x = j[1]["previous_path_x"];
          	auto previous_path_y = j[1]["previous_path_y"];
          	// Previous path's end s and d values 
          	double end_path_s = j[1]["end_path_s"];
          	double end_path_d = j[1]["end_path_d"];

          	// Sensor Fusion Data, a list of all other cars on the same side of the road.
          	auto sensor_fusion = j[1]["sensor_fusion"];

          	json msgJson;

          	vector<double> next_x_vals;
          	vector<double> next_y_vals;
            
            // Previous path may have been partially executed
            // If so, make sure we start from previous path's end
            int prev_size = previous_path_x.size();
            if (prev_size>1) {
              car_s = end_path_s;
            }

            // We will maintain 3 boolean flags to facilitate lane & speed changes
            bool flag_ahead = false;
            bool flag_left = false;
            bool flag_right = false;
            for (int i=0; i<sensor_fusion.size(); i++) {
              float d = sensor_fusion[i][6];
              // Find out neighboring car's lane
              int lane = 0;
              for (; lane<3; lane++)
                if (d>(2+4*lane-2) && d<(2+4*lane+2)) 
                  break;

              double vx = sensor_fusion[i][3];
              double vy = sensor_fusion[i][4];
              double v = sqrt(vx*vx + vy*vy);
              double s = sensor_fusion[i][5];

              // Predict neighboring car's position after executing previous path
              s += prev_size*.02*v;
              
              // If car is within 30m ahead
              if (lane==car_lane && s>car_s && s-car_s<30)
                flag_ahead = true;
              // If car is within 20m to the left 
              else if (car_lane-lane==1 && fabs(s-car_s)<30)
                flag_left = true;
              // If car is within 20m to the right 
              else if (lane-car_lane==1 && fabs(s-car_s)<30)
                flag_right = true;
            }

            // With these flags set, we'll transition into appropriate states
            if (flag_ahead) {
              // If we have a car ahead, it means, it's moving slower compared to our desired speed
              // Otherwise we wouldn't have encountered it. So, change lanes if possible else keep lane
              if (!flag_right && car_lane<2)
                car_lane += 1;
              else if (!flag_left && car_lane>0)
                car_lane -= 1;
              else 
                car_v -= ref_acc;
            } else {
              // If no car was ahead changes have occurred, make sure car runs at max possible speed
              if (car_v<49.5-ref_acc)
                car_v += ref_acc;
              // Moreover, if the car is not in the center lane, bring it back.
              // Being in the center lane opens up more possibilities (change left/right) when we 
              // encounter a slower vehicle ahead
              if (car_lane!=1) {
                if (car_lane==0 && !flag_right)
                  car_lane += 1;
                else if (car_lane==2 && !flag_left)
                  car_lane -= 1;
                else
                  ;
              }
            }
           
            double dist_inc = .5;

            // We fit a spline to waypoints
            // Firstly, we collect 5 anchor points, two of those come from previous path (if exists, if not we simulate previous path)
            // 3 of those are from 30, 60, 90m ahead in the same lane
            vector<double> xs, ys;
            double ref_x, ref_y, ref_yaw;
            if (prev_size<2) {
              // Get car's s & d
              double car_s_prev = car_s - dist_inc;
              double car_d_prev = car_d;
              vector<double> xy = getXY(car_s_prev, car_d_prev, map_waypoints_s, map_waypoints_x, map_waypoints_y);
              xs.push_back(xy[0]);
              xs.push_back(car_x);
              ys.push_back(xy[1]);
              ys.push_back(car_y);
              ref_x = car_x;
              ref_y = car_y;
              ref_yaw = deg2rad(car_yaw);
            } else {
              double car_x_prev = previous_path_x[prev_size-1];
              double car_y_prev = previous_path_y[prev_size-1];
              double car_x_prev_prev = previous_path_x[prev_size-2];
              double car_y_prev_prev = previous_path_y[prev_size-2];
              double dy = car_y_prev - car_y_prev_prev;
              double dx = car_x_prev - car_x_prev_prev;

              xs.push_back(car_x_prev_prev);
              xs.push_back(car_x_prev);
              ys.push_back(car_y_prev_prev);
              ys.push_back(car_y_prev);
              ref_x = car_x_prev;
              ref_y = car_y_prev;
              ref_yaw = atan2(dy, dx);
            }

            // Add points 30, 60, 90m ahead
            for (int i=0; i<3; i++) {
              vector<double> xy = getXY(car_s+30*(i+1), (2+4*car_lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
              xs.push_back(xy[0]);
              ys.push_back(xy[1]);
            }

            // Transform all these 5 anchor points from global to car's local coordinates
            for (int i=0; i<xs.size(); i++) {
              double x = xs[i]-ref_x;
              double y = ys[i]-ref_y;

              xs[i] = x*cos(0-ref_yaw) - y*sin(0-ref_yaw);
              ys[i] = x*sin(0-ref_yaw) + y*cos(0-ref_yaw);
            }

            // Fit spline
            tk::spline s;
            s.set_points(xs, ys);

            // Add any points leftover from previous path
            for (int i=0; i<previous_path_x.size(); i++) {
              next_x_vals.push_back(previous_path_x[i]);
              next_y_vals.push_back(previous_path_y[i]);
            }

            // Add up to 50 points, based on spline fit
            // The points on spline are being added at such intervals so that the 
            // x values are dist_inc apart. Hopefully, the spline won't have crazy 
            // curvature and hence, given the x values are dist_incr apart, spline
            // points (which are what the car will visit every .02 sec) will be approx
            // apart the same amount, thus we needn't worry about exceeding max jerk.
            // If max jerk is exceeded, we can always reduce dist_incr.
            double X = 30;
            double Y = s(X);
            double dist = distance(0, 0, X, Y);
            double dist_inc_x = .02*car_v/2.24;
            for (int i=1; i<=50-previous_path_x.size(); i++) {
              double x = dist_inc_x*i;
              double y = s(x);

              double x_tmp = x;
              double y_tmp = y;

              x = x_tmp*cos(ref_yaw)-y_tmp*sin(ref_yaw);
              y = x_tmp*sin(ref_yaw)+y_tmp*cos(ref_yaw);

              x += ref_x;
              y += ref_y;

              next_x_vals.push_back(x);
              next_y_vals.push_back(y);
            }
            // END
          	msgJson["next_x"] = next_x_vals;
          	msgJson["next_y"] = next_y_vals;

          	auto msg = "42[\"control\","+ msgJson.dump()+"]";

          	//this_thread::sleep_for(chrono::milliseconds(1000));
          	ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
          
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}