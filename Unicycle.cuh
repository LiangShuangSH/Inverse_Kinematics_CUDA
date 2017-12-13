/*
 * Unicycle.cuh
 *
 *  Created on: Nov 21, 2017
 *      Author: liang
 */

#ifndef UNICYCLE_CUH_
#define UNICYCLE_CUH_

#include <Eigen/Dense>
#include <Eigen/QR>
#include "PseudoInverse.h"

using namespace std;
using namespace Eigen;

__host__ __device__ double* forward_kinematics(double* state, double a, double b, double t);

__host__ __device__ double* forward_kinematics_3(double* state, double a1, double b1, double t1,
									 	 	 	 	double a2, double b2, double t2,
									 	 	 	 	double a3, double b3, double t3);

__host__ __device__ double* forward_kinematics_3(double* state, double* u);

__host__ __device__ MatrixXd jacobian(double* state, double a, double b, double t);

__host__ __device__ MatrixXd jacobian2(double* state, double a1, double b1, double t1, double a, double b, double t);

__host__ __device__ MatrixXd jacobian3(double* state, double a1, double b1, double t1, double a2, double b2, double t2, double a3, double b3, double t3);

__host__ __device__ MatrixXd jacobian(double* state, double a1, double b1, double t1, double a2, double b2, double t2, double a3, double b3, double t3);

__host__ double* inverse_kinematics_3(double* sq, double* uk, double* sk, double* s0);

#endif /* UNICYCLE_CUH_ */
