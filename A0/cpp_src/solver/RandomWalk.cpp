#include "RandomWalk.h"


RandomWalk::RandomWalk(double dt, double time) {
	dt_ = dt;
	time_ = time;

	n_ = static_cast<int>(time_/dt_);
}

template <typename T>
T RandomWalk::random(T range_from, T range_to) {
	/*
	range_from: Minimum value of random (normal) distribution.
	range_to: Maximum value of random (normal) distribution.

	returns a random number within the range. Used to get epsilon in walk().
	*/

	std::random_device rand_dev;
	std::mt19937 generator(rand_dev());
	std::normal_distribution<T> distr((range_from+ range_to)/2, (range_to-range_from)/6);	

	return distr(generator);
}

std::string RandomWalk::get_stop_criteria(double epsilon) {
	std::string stop_criteria = "undefined";
	if (epsilon > 0) {
		stop_criteria = "negative";
	}
	else if (epsilon < 0) {
		stop_criteria = "positive";
	}
	return stop_criteria;
}


void RandomWalk::walk(double range_from, double range_to) {
	/*
	walk() should perform the random walk process. The variable epsilon is the value that the particle
	moves up or down the y-axis. The position is x(t), equivalent to the vector x. The movement is 
	done when the particle returns to its starting point, x(t) = 0. The value of epsilon is uniformly distributed 
	random numbers, here Normal distribution. 


	range_from: Smallest allowed number in the uniformly distributed random numbers. 

	range_to: Largest allowed number in the uniformly distributed random numbers.
	
	*/

	bool stop_criteria_found = false;
	std::string stop_criteria = "undefined";
	double epsilon = 0.0;

	while (!stop_criteria_found){
		epsilon = random(range_from, range_to);
		stop_criteria = get_stop_criteria(epsilon);

		if (stop_criteria != "undefined") {
			stop_criteria_found = true;
		}
	}
	Eigen::RowVectorXd x(n_);
	//std::cout << x.size() << std::endl;
	x(0) = epsilon;
	
	for (int i = 1; i = n_; ++i) { // Something goes wrong in this loop. 
		epsilon = random(range_from, range_to);
		if ( ((stop_criteria == "positive") && (x(i - 1) + epsilon >= 0)) 
			|| ((stop_criteria == "negative") && (x(i - 1) + epsilon <= 0)) ) {
			//std::cout << x(i) << std::endl;
			x(i) = 0.0;
			//find the time it happened and put in array...
			std::cout << "The particle has returned to the starting point.\n";
			break;
		}
		else {
			x(i) = x(i - 1) + epsilon;
		}
	}
	std::cout << "Here is the vector x: \n";
}
