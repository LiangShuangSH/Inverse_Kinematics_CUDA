/*
 * utility.cu
 *
 *  Created on: Dec 4, 2017
 *      Author: liang
 */

#include "utility.cuh"

using namespace Eigen;
using namespace std;

__device__ double normalize_angle(double angle) {
	double new_angle = fmod(angle + PI, 2 * PI);
	if (new_angle < 0) {
		new_angle += 2 * PI;
	}
	return new_angle - PI;
}

__host__ __device__ double euclidean_distance(double* a, double* b, int size) {
	double result = 0.0;

	for (int i = 0; i < size; i++) {
		result += pow(a[i] - b[i], 2);
	}
	result = sqrt(result);

	return result;
}

__device__ VectorXd normalize_vector(VectorXd v, int dim, double scale) {
	double IVI = 0.0;
	for (int i = 0; i < dim; i++) {
		IVI += pow(v(i), 2);
	}
	IVI = sqrt(IVI);
	VectorXd result = scale * v / IVI;

	return result;
}


__host__ __device__ VectorXd pointerToVector(double* p, int size) {
	VectorXd v(size);
	for (int i = 0; i < size; i++) {
		v(i) = p[i];
	}
	return v;
}

double* vectorToPointer(VectorXd v, int size) {
	double* p = new double[size];
	for (int i = 0; i < size; i++) {
		p[i] = v(i);
	}
	return p;
}

__host__ __device__ void copy_array(double* copy, double* origin, int size) {
	for (int i = 0; i < size; i++) {
		copy[i] = origin[i];
	}
}

__host__ double fRand(double fMin, double fMax) {
	double f = (double)rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}

