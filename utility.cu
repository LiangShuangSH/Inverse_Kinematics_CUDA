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

__host__ __device__ double angleDiff(double a,double b){
    double dif = fmod(b - a + PI2,PII);
    if (dif < 0)
        dif += PII;
    return dif - PI2;
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

__host__ __device__ void pointerToVector2(VectorXd& v, double* p, int size) {
	for (int i = 0; i < size; i++) {
		v(i) = p[i];
	}
	return;
}

__host__ __device__ double* vectorToPointer(VectorXd v, int size) {
	double* p = new double[size];
	for (int i = 0; i < size; i++) {
		p[i] = v(i);
	}
	return p;
}

__host__ __device__ void vectorToArray(double a[], VectorXd v, int size) {
	for (int i = 0; i < size; i++) {
		a[i] = v(i);
	}
	return;
}

__host__ __device__ void pointerToMatrix(MatrixXd& m, double* p, int rows, int cols) {
	int idx = 0;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			m(i, j) = p[idx++];
		}
	}
	return;
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

__device__ double calculate_dist_travelled(double a1, double a2, double a3, double t1, double t2, double t3) {
	double dist = 0.0;
	double v1 = 0.0 + a1 * t1;
	double sgn = 0.0;
	// u1
	sgn = (v1 >= 0.0) ? 1.0 : -1.0;
	dist += (0.5*a1*pow(t1, 2)) * sgn; // 0.5at² * sgn(v1)

	// u2
	double v2 = v1 + a2*t2;
	// If not going other direction
	if (v1*v2 >= 0.0) {
		sgn = (v1 >= 0.0) ? 1.0 : -1.0;
		dist += (v1 * t2 + 0.5 * a2 * pow(t2, 2)) * sgn; // (v1*t2 + 0.5*a2*t2²) * sgn(v1)
	}
	// Going other direction in the middle
	else {
		double t_0 = abs(v1 / a2);
		sgn = (v1 >= 0.0) ? 1.0 : -1.0;
		dist += (v1 * t_0 + 0.5 * a2 * pow(t_0, 2)) * sgn; // (v1*t_0 + 0.5*a2*t2²) * sgn(v1)
		sgn = (v2 >= 0.0) ? 1.0 : -1.0;
		dist += (0.0 + 0.5 * a2 * pow(t2 - t_0, 2)) * sgn; // (v1*t_0 + 0.5*a2*(t2-t_0)²) * sgn(v2)
	}

	// u3
	double v3 = v2 + a3*t3;
	// If not going other direction
	if (v2*v3 >= 0.0) {
		sgn = (v2 >= 0.0) ? 1.0 : -1.0;
		dist += (v2 * t3 + 0.5 * a3 * pow(t3, 2)) * sgn; // (v2*t3 + 0.5*a3*t3²)* sgn(v2)
	}
	// Going other direction in the middle
	else {
		double t_0 = abs(v2 / a3);
		sgn = (v2 >= 0.0) ? 1.0 : -1.0;
		dist += (v2 * t_0 + 0.5 * a3 * pow(t_0, 2)) * sgn;
		sgn = (v3 >= 0.0) ? 1.0 : -1.0;
		dist += (0.0 + 0.5 * a3 * pow(t3 - t_0, 2)) * sgn;
	}

	return dist;
}

__host__ __device__ bool angle_check(double a1, double a2, double a3, double t1, double t2, double t3) {
	bool nice = true;
	double angle = 0.0;
	double v1 = 0.0 + a1 * t1;

	// u1
	angle += (0.5*a1*pow(t1, 2));
	if (abs(angle) >= PII) {
		nice = false;
		return nice;
	}

	// u2
	double v2 = v1 + a2*t2;
	// If not going other direction
	if (v1*v2 >= 0.0) {
		angle += (v1 * t2 + 0.5 * a2 * pow(t2, 2)); // (v1*t2 + 0.5*a2*t2²)
		if (abs(angle) >= PII) {
			nice = false;
			return nice;
		}
	}
	// Going other direction in the middle
	else {
		double t_0 = abs(v1 / a2);
		angle += (v1 * t_0 + 0.5 * a2 * pow(t_0, 2)); // (v1*t_0 + 0.5*a2*t2²)
		if (abs(angle) >= PII) {
			nice = false;
			return nice;
		}
		angle += (0.0 + 0.5 * a2 * pow(t2 - t_0, 2)); // (v1*t_0 + 0.5*a2*(t2-t_0)²)
		if (abs(angle) >= PII) {
			nice = false;
			return nice;
		}
	}

	// u3
	double v3 = v2 + a3*t3;
	// If not going other direction
	if (v2*v3 >= 0.0) {
		angle += (v2 * t3 + 0.5 * a3 * pow(t3, 2)); // (v2*t3 + 0.5*a3*t3²)
		if (abs(angle) >= PII) {
			nice = false;
			return nice;
		}
	}
	// Going other direction in the middle
	else {
		double t_0 = abs(v2 / a3);
		angle += (v2 * t_0 + 0.5 * a3 * pow(t_0, 2));
		if (abs(angle) >= PII) {
			nice = false;
			return nice;
		}
		angle += (0.0 + 0.5 * a3 * pow(t3 - t_0, 2));
		if (abs(angle) >= PII) {
			nice = false;
			return nice;
		}
	}

	return nice;
}


// Find top n min errors in record
__host__ void get_best_results(double* errors_min, unsigned long int idx_min[][2], double** errors_gpu, int** indicator, unsigned long int* data_per_gpu, int num) {
	double curr_err;
	// Iterate through all errors responding by all kernels
	for (int gpu = 0; gpu < 4; gpu++) {
		for (unsigned long int i = 0; i < data_per_gpu[gpu]; i++) {
			// If bad trajectory, skip
			if (indicator[gpu][i] == 0) {
				continue;
			}
			curr_err = errors_gpu[gpu][i];
			// Iterate through top n errors record to find the max error
			double max = errors_min[0];
			int max_idx = 0;
			for (int err_idx = 1; err_idx < num; err_idx++) {
				if (errors_min[err_idx] > max) {
					max = errors_min[err_idx];
					max_idx = err_idx;
				}
			}
			// If curr_err is smaller than max min_error in record, replace the max one with curr_error
			errors_min[max_idx] = (curr_err < max) ? curr_err : max;
			idx_min[max_idx][0] = gpu;
			idx_min[max_idx][1] = i;
		}
	}
}

