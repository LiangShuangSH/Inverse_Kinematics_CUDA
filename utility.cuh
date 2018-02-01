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
__host__ __device__ Eigen::VectorXd pointerToVector(double* p, int size);
__host__ __device__ void pointerToVector2(Eigen::VectorXd& v, double* p, int size);
__host__ __device__ void vectorToArray(double a[], Eigen::VectorXd v, int size);
__host__ __device__ double* vectorToPointer(Eigen::VectorXd v, int size);
__host__ __device__ void pointerToMatrix(Eigen::MatrixXd& m, double* p, int rows, int cols);
__host__ __device__ void copy_array(double* copy, double* origin, int size);

#endif /* UTILITY_CUH_ */
