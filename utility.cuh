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
__host__ __device__ double angleDiff(double a,double b);
__device__ Eigen::VectorXd normalize_vector(Eigen::VectorXd v, int dim, double scale);
__host__ __device__ Eigen::VectorXd pointerToVector(double* p, int size);
double* vectorToPointer(Eigen::VectorXd v, int size);
__host__ __device__ void copy_array(double* copy, double* origin, int size);
__host__ double fRand(double fMin, double fMax);
__device__ double calculate_dist_travelled(double a1, double a2, double a3, double t1, double t2, double t3);
__host__  __device__ bool angle_check(double a1, double a2, double a3, double t1, double t2, double t3);
__host__ void get_best_results(double* errors_min, unsigned long int idx_min[][2], double** errors_gpu, int** indicator, unsigned long int* data_per_gpu, int num);

#endif /* UTILITY_CUH_ */
