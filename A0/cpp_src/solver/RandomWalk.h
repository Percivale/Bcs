
#ifndef RANDOMWALK_H
#define RANDOMWALK_H
#include <iostream>
#include <random>
#include <Eigen/Dense>



class RandomWalk {
private:
	double dt_ =0.0;
	double time_ = 0;
	int n_ = 0;


	template <typename T>
	T random(T range_from, T range_to);

	std::string get_stop_criteria(double epsilon);


public:
	RandomWalk(double dt, double time);

	void walk(double range_from, double range_to);



};

#endif // !RANDOMWALK_H