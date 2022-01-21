//#include <iostream>
#include "../solver/RandomWalk.h"
 
using Eigen::MatrixXd;
 
int main()
{
  RandomWalk test(0.1, 10);
  //Eigen::RowVectorXd x(10);
  //x(0) = 0.231;
  //std::cout << x << std::endl;
  test.walk(-1.0, 1.0);
}
