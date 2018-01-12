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

__device__ double normalize_angle(double angle);
__host__ __device__ double euclidean_distance(double* a, double* b, int size);
__device__ Eigen::VectorXd normalize_vector(Eigen::VectorXd v, int dim, double scale);
Eigen::VectorXd pointerToVector(double* p, int size);
double* vectorToPointer(Eigen::VectorXd v, int size);
__host__ __device__ void copy_array(double* copy, double* origin, int size);

#endif /* UTILITY_CUH_ */
