/*
 * utility.cuh
 *
 *  Created on: Dec 4, 2017
 *      Author: liang
 */

#ifndef UTILITY_CUH_
#define UTILITY_CUH_

#include <math.h>
#include "globals.h"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <iostream>

using namespace Eigen;
using namespace std;

__device__ double normalize_angle(double angle);
__host__ __device__ double euclidean_distance(double* a, double* b, int size);
__device__ VectorXd normalize_vector(VectorXd v, int dim, double scale);
VectorXd pointerToVector(double* p, int size);
double* vectorToPointer(VectorXd v, int size);

#endif /* UTILITY_CUH_ */
