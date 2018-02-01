#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <fstream>
#include "globals.h"
#include <Eigen/Dense>
#include <Eigen/QR>
#include "Unicycle.cuh"
#include "utility.cuh"
#include "Files.h"
#include <curand.h>
#include <curand_kernel.h>
#include "PseudoInverse.h"
#include <string>
#include <ctime>

using namespace std;
using namespace Eigen;

const double A_MAX = 0.8;
const double A_MIN = -0.8;
const double B_MAX = 0.8;
const double B_MIN = -0.8;
const double T_MAX = 0.8;
const double T_MIN = 0.0;
const double V_MAX = 120000.0/3600.0;
const double W_MAX = PI;

using namespace std;
using namespace Eigen;

void test_code();
__global__ void test_kernel(double* sq, double* inv_jcb_data,
							double itv, int a_num, int b_num, int t_num,
							double* errors, double* uqs, unsigned long int data_num);
__global__ void Experiment_Check_IK(double* sq, double* sk, double* uk, double* inv_jcbs, unsigned long int data_num, double* record);
void inverse_jcb(double* inv_jcb_record, double* uk_record, double* sk_record, unsigned long int data_num);

__global__ void test_host_code(double* J) {
	double state[5] = {0};
	forward_kinematics_3(state, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
	MatrixXd mat_J = jacobian(state, 1.0, 1.0, 1.0);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 5; j++) {
			J[i*3 + j] = mat_J(j, i);
		}
	}
}

__global__ void uniform_sampling(double* uk_record, double* sk_record, double itv, int a_num, int b_num, int t_num, unsigned long int data_num) {
	//unsigned long int tid =  blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long int blockId = blockIdx.x //1D
	        + blockIdx.y * gridDim.x //2D
	        + gridDim.x * gridDim.y * blockIdx.z; //3D
	unsigned long int tid = blockId * blockDim.x + threadIdx.x;
	if (tid >= data_num) {return;}

	double* uk = new double[9];
	double* sk = new double[5];
	for (int i = 0; i < 5; i++) {
		sk[i] = 0.0;
	}

	// Locate the sample point in 9 dim space by thread idx
	int sample_idx[9] = {0};
	int num = tid;
	int location = 0;

	while (num != 0) {
		int remainder;
		if (location % 3 == 0) {
			remainder = num % a_num;
			num = num / a_num;
		}
		else if (location % 3 == 1) {
			remainder = num % b_num;
			num = num / b_num;
		}
		else {
			remainder = num % t_num;
			num = num / t_num;
		}
		sample_idx[location] = remainder;
		location++;
	}

	// Translate to uk
	uk[0] = A_MIN + itv * sample_idx[0];
	uk[1] = B_MIN + itv * sample_idx[1];
	uk[2] = T_MIN + itv * sample_idx[2];
	uk[3] = A_MIN + itv * sample_idx[3];
	uk[4] = B_MIN + itv * sample_idx[4];
	uk[5] = T_MIN + itv * sample_idx[5];
	uk[6] = A_MIN + itv * sample_idx[6];
	uk[7] = B_MIN + itv * sample_idx[7];
	uk[8] = T_MIN + itv * sample_idx[8];

	// Get sk
	forward_kinematics_3(sk, uk[0], uk[1], uk[2],
							 uk[3], uk[4], uk[5],
							 uk[6], uk[7], uk[8]);

	// Record
	for (int i = 0; i < 9; i++) {
		uk_record[tid*9 + i] = uk[i];
	}

	for (int i = 0; i < 5; i++) {
		sk_record[tid*5 + i] = sk[i];
	}

	free(uk);
	free(sk);
}

void cuda_uniform_sampling() {
	double itv = 0.2;
	// Num of data
	int a_num = (A_MAX - A_MIN) / itv + 1;
	int b_num = (B_MAX - B_MIN) / itv + 1;
	int t_num = (T_MAX - T_MIN) / itv + 1;
	unsigned long int data_num = pow(a_num * b_num * t_num, 3);

	int thread_num = 256;
	dim3 block_num(128, 64, 64);

	// Assign memory on host
	double* uk_record = new double[data_num*9];
	double* sk_record = new double[data_num*5];

	// Assign memory on device
	double *dev_uk_record;
	cudaMalloc(&dev_uk_record, data_num * 9 * sizeof(double));
	cudaMemcpy(dev_uk_record, uk_record, (data_num*9)*sizeof(double), cudaMemcpyHostToDevice);
	double *dev_sk_record;
	cudaMalloc(&dev_sk_record, data_num * 5 * sizeof(double));
	cudaMemcpy(dev_sk_record, sk_record, (data_num*5)*sizeof(double), cudaMemcpyHostToDevice);

	cout << "Sampling" << endl;
	// Perform kernel
	uniform_sampling<<<block_num, thread_num>>>(dev_uk_record, dev_sk_record, itv, a_num, b_num, t_num, data_num);

	// Fetch the data from device to host
	cudaMemcpy(uk_record, dev_uk_record, (data_num*9)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(sk_record, dev_sk_record, (data_num*5)*sizeof(double), cudaMemcpyDeviceToHost);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	// Start recording inverse matrix of jcb
	int m = 5;
	int n = 9;
	double* inv_jcb_record = new double[data_num*m*n];
	inverse_jcb(inv_jcb_record, uk_record, sk_record, data_num);

	cout << "Start Writing" << endl;

	Write_File(uk_record, sk_record, inv_jcb_record, data_num);

	cout << "Done" << endl;

	free(uk_record);
	cudaFree(dev_uk_record);

	free(sk_record);
	cudaFree(dev_sk_record);

	free(inv_jcb_record);
}

__global__ void find_inverse_kinematics(double* sq, double* dist_record, double* sk_data, unsigned long int data_num) {
	unsigned long int blockId = blockIdx.x //1D
		        + blockIdx.y * gridDim.x //2D
		        + gridDim.x * gridDim.y * blockIdx.z; //3D
	unsigned long int tid = blockId * blockDim.x + threadIdx.x;
	if (tid >= data_num) {return;}

	double* sk = new double[5];

	for (int i = 0; i < 5; i++) {
		sk[i] = sk_data[tid*5 + i];
	}

	// Compare it with the query state
	double dist = euclidean_distance(sk, sq, 5);

	dist_record[tid] = dist;

	free(sk);
}


void cuda_find_inverse_kinematics() {
	double itv = 0.2;
	// Num of data
	int a_num = (A_MAX - A_MIN) / itv + 1;
	int b_num = (B_MAX - B_MIN) / itv + 1;
	int t_num = (T_MAX - T_MIN) / itv + 1;
	unsigned long int data_num = pow(a_num * b_num * t_num, 3);

	int thread_num = 256;
	dim3 block_num(128, 64, 64);

	double* s0 = new double[5];
	s0[0] = 0.0; s0[1] = 0.0; s0[2] = 0.0; s0[3] = 0.0; s0[4] = 0.0;
	double* sq = new double[5];
	sq[0] = s0[0]; sq[1] = s0[1]; sq[2] = s0[2]; sq[3] = s0[3]; sq[4] = s0[4];
	double* u = new double[9];
	u[0] = 0.4; u[1] = 0.4; u[2] = 0.4;
	u[3] = 0.5; u[4] = 0.5; u[5] = 0.5;
	u[6] = 0.4; u[7] = 0.4; u[8] = 0.4;
	forward_kinematics_3(sq, u);

	double* dev_sq;
	cudaMalloc(&dev_sq, 5 * sizeof(double));
	cudaMemcpy(dev_sq, sq, 5*sizeof(double), cudaMemcpyHostToDevice);

	double* dist_record = new double[data_num];
	double* sk_data = new double[data_num*5];
	double* uk_data = new double[data_num*9];
	double* inv_jcb_data = new double[data_num*9*5];
	Read_File(uk_data, sk_data, inv_jcb_data, false);

	double* dev_dist_record;
	cudaMalloc(&dev_dist_record, data_num*sizeof(double));
	cudaMemcpy(dev_dist_record, dist_record, data_num*sizeof(double), cudaMemcpyHostToDevice);
	double* dev_sk_data;
	cudaMalloc(&dev_sk_data, data_num * 5 * sizeof(double));
	cudaMemcpy(dev_sk_data, sk_data, (data_num*5)*sizeof(double), cudaMemcpyHostToDevice);

	cout << "Calculating Distance" << endl;
	// Perform kernel
	find_inverse_kinematics<<<block_num, thread_num>>>(dev_sq, dev_dist_record, dev_sk_data, data_num);

	// Fetch the data from device to host
	cudaMemcpy(dist_record, dev_dist_record, data_num*sizeof(double), cudaMemcpyDeviceToHost);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	cout << "Inverse Kinematics" << endl;

	ofstream ik_file;
	ik_file.open("./inverse_kinematics.data");
	ik_file << "Sample interval: " << itv << endl;
	ik_file << "Query: ";
	for (int i = 0; i < 5; i++) {
		ik_file << sq[i] << " ";
	}
	ik_file << "" << endl;
	ik_file << "" << endl;

	double error;
	double uk[9];
	double sk[5];

	double error_min = 100.0;
	double* uk_min = new double[9];
	double* sk_min = new double[5];
	double* uq_min = new double[9];

	double threshold = 0.4;

	for (unsigned long int i = 0; i < data_num; i++) {
		if (dist_record[i] > threshold ) {
			continue;
		}
		// Do inverse kinematics
		// Assign uk, sk
		for (int j = 0; j < 9; j++) { uk[j] = uk_data[i*9 + j]; }
		for (int j = 0; j < 5; j++) { sk[j] = sk_data[i*5 + j]; }
		double* uq = inverse_kinematics_3(sq, uk, sk, s0);
		double* sf = new double[5];
		sf[0] = s0[0]; sf[1] = s0[1]; sf[2] = s0[2]; sf[3] = s0[3]; sf[4] = s0[4];
		forward_kinematics_3(sf, uq);
		error = euclidean_distance(sf, sq, 2);

		// Check Min
		if (error < error_min) {
			error_min = error;
			for (int j = 0; j < 9; j++) { uk_min[j] = uk[j]; }
			for (int j = 0; j < 5; j++) { sk_min[j] = sk[j]; }
			for (int j = 0; j < 9; j++) { uq_min[j] = uq[j]; }
		}

		// Write file
		/*
		ik_file << "No." << i << ", " << "error: " << error << endl;
		ik_file << "uk: ";
		for(int j = 0; j < 9; j++) { ik_file << uk[j] << " "; }
		ik_file << "" << endl;
		ik_file << "sk: ";
		for(int j = 0; j < 5; j++) { ik_file << sk[j] << " "; }
		ik_file << "" << endl;
		ik_file << "uq: ";
		for(int j = 0; j < 9; j++) { ik_file << uq[j] << " "; }
		ik_file << "" << endl;
		ik_file << "s diff: ";
		for(int j = 0; j < 5; j++) { ik_file << sf[j] - sq[j] << " "; }
		ik_file << "" << endl;
		ik_file << "" << endl;
		*/
	}

	// Write Min
	ik_file << "The best result: " << endl;
	ik_file << "error: " << error_min << endl;
	ik_file << "uk: ";
	for(int j = 0; j < 9; j++) { ik_file << uk_min[j] << " "; }
	ik_file << "" << endl;
	ik_file << "sk: ";
	for(int j = 0; j < 5; j++) { ik_file << sk_min[j] << " "; }
	ik_file << "" << endl;
	ik_file << "uq: ";
	for(int j = 0; j < 9; j++) { ik_file << uq_min[j] << " "; }
	ik_file << "" << endl;
	ik_file << "" << endl;

	ik_file.close();
}



int main()
{
	//cuda_find_inverse_kinematics();
	//cuda_uniform_sampling();
	test_code();

    return 0;
}



// The kernel deals one query at a time in limited search space
__global__ void Experiment_Check_IK(double* sq, double* sk, double* uk, double* inv_jcbs, unsigned long int data_num, double* record) {
	// TID
	int tid = threadIdx.x;

	// Create a random control
	// Init a cuRand state
	curandState_t rstate;
	curand_init(0, 0, 0, &rstate);

	double* u = new double[9];
	for (int i = 0; i < 3; i++) {
		u[i*3 + 0] = A_MIN + curand_uniform_double(&rstate) * (A_MAX - A_MIN);
		u[i*3 + 1] = B_MIN + curand_uniform_double(&rstate) * (B_MAX - B_MIN);
		u[i*3 + 2] = T_MIN + curand_uniform_double(&rstate) * (T_MAX - T_MIN);
	}
	// Get sq by random control
	for (int i = 0; i < 5; i++) {sq[i] = 0;}
	forward_kinematics_3(sq, u);

	// Get inverse kinematics

}

void inverse_jcb(double* inv_jcb_record, double* uk_record, double* sk_record, unsigned long int data_num) {
	int m = 5;
	int n = 9;

	double* uk = new double[9];
	double* sk = new double[5];

	for (unsigned long int i = 0; i < data_num; i++) {
		// Assign uk, sk
		for (int j = 0; j < 9; j++) { uk[j] = uk_record[i * 9 + j]; }
		for (int j = 0; j < 5; j++) { sk[j] = sk_record[i * 5 + j]; }
		// Inverse the jcb
		MatrixXd jcb = jacobian(sk, uk[0], uk[1], uk[2], uk[3], uk[4], uk[5], uk[6], uk[7], uk[8]);
		MatrixXd inv_jcb = pinv(jcb);
		// Record
		for (int j = 0; j < m; j++) {
			for (int k = 0; k < n; k++) {
				inv_jcb_record[i*m*n + j * m + k] = inv_jcb(k, j);
			}
		}
	}
}

void test_code() {
	double itv = 0.2;
	// Num of data
	int a_num = (A_MAX - A_MIN) / itv + 1;
	int b_num = (B_MAX - B_MIN) / itv + 1;
	int t_num = (T_MAX - T_MIN) / itv + 1;
	unsigned long int data_num = pow(a_num * b_num * t_num, 3);
	//data_num = 100;
	//unsigned long int data_per_gpu = (data_num - 1) / 4 + 1;

	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 134217728);
	size_t size;
	cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
	cout << "Heap size: " << size << endl;

	double* inv_jcb_data;
	cudaMallocManaged(&inv_jcb_data, data_num*9*5*sizeof(double));
	Read_Inv_Jcb(inv_jcb_data);

	double* s0 = new double[5];
	s0[0] = 0.0; s0[1] = 0.0; s0[2] = 0.0; s0[3] = 0.0; s0[4] = 0.0;
	double* sq;
	cudaMallocManaged(&sq, 5*sizeof(double));
	sq[0] = s0[0]; sq[1] = s0[1]; sq[2] = s0[2]; sq[3] = s0[3]; sq[4] = s0[4];
	double* u = new double[9];
	u[0] = 0.4; u[1] = 0.4; u[2] = 0.4;
	u[3] = 0.5; u[4] = 0.5; u[5] = 0.5;
	u[6] = 0.4; u[7] = 0.4; u[8] = 0.4;
	forward_kinematics_3(sq, u);

	// Searching space
	clock_t begin = clock();

	// Host
	double* errors;
	double* uqs;
	cudaMallocManaged(&errors, data_num*sizeof(double));
	cudaMallocManaged(&uqs, data_num*9*sizeof(double));

	// Device
	/*
	double* dev_sq;
	double* dev_inv_jcb_data;
	double* dev_errors;
	double* dev_uqs;

	cudaMalloc(&dev_sq, 5*sizeof(double));
	cudaMemcpy(dev_sq, sq, 5*sizeof(double), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_inv_jcb_data, data_per_gpu*9*5*sizeof(double));
	cudaMemcpy(dev_inv_jcb_data, inv_jcb_data, data_per_gpu*9*5*sizeof(double), cudaMemcpyHostToDevice);
	printf("Device Variable Copying:\t%s\n", cudaGetErrorString(cudaGetLastError()));

	cudaMalloc(&dev_errors, data_per_gpu*sizeof(double));
	cudaMemcpy(dev_errors, errors, data_per_gpu*sizeof(double), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_uqs, data_per_gpu*9*sizeof(double));
	cudaMemcpy(dev_uqs, uqs, data_per_gpu*9*sizeof(double), cudaMemcpyHostToDevice);
	*/


	// Launch Kernel
	int thread_num = 256;
	dim3 block_num(32, 32, 32);

	test_kernel<<<block_num, thread_num>>>(sq, inv_jcb_data,
										   itv, a_num, b_num, t_num,
										   errors, uqs, data_num);

	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	// Fetch the data
	//cudaMemcpy(errors, dev_errors, data_per_gpu*sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(uqs, dev_uqs, data_per_gpu*9*sizeof(double), cudaMemcpyDeviceToHost);

	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << "Inverse kinematics time: " << elapsed_secs << endl;
	cout << "" << endl;

	for(int i = 888000; i < 10; i++) {
		cout << errors[i] << endl;
	}

	// Search best
	begin = clock();
	double error_min = errors[0];
	unsigned long int idx_min = 0;
	for (int i = 1; i < data_num; i++) {
		if(error_min > errors[i]) {
			error_min = errors[i];
			idx_min = i;
		}
	}

	cout << "Best results:" << endl;
	cout << "Error: " << error_min << endl;
	cout << "uq_min: ";
	for (int i = 0; i < 9; i++) {
		cout << uqs[idx_min*9 + i] << " ";
	}
	cout << "" << endl;

	end = clock();
	elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << "Search time: " << elapsed_secs;
}

__global__ void test_kernel(double* sq, double* inv_jcb_data,
							double itv, int a_num, int b_num, int t_num,
							double* errors, double* uqs, unsigned long int data_num) {
	unsigned long int blockId = blockIdx.x //1D
			        + blockIdx.y * gridDim.x //2D
			        + gridDim.x * gridDim.y * blockIdx.z; //3D
	unsigned long int tid = blockId * blockDim.x + threadIdx.x;
	if (tid >= data_num) {return;}

	// Get uk, sk
	double uk[9];
	double sk[5];
	for (int i = 0; i < 5; i++) {
		sk[i] = 0.0;
	}

	// Locate the sample point in 9 dim space by thread idx
	int sample_idx[9] = {0};
	unsigned long int num = tid;
	int location = 0;

	while (num != 0) {
		int remainder;
		if (location % 3 == 0) {
			remainder = num % a_num;
			num = num / a_num;
		}
		else if (location % 3 == 1) {
			remainder = num % b_num;
			num = num / b_num;
		}
		else {
			remainder = num % t_num;
			num = num / t_num;
		}
		sample_idx[location] = remainder;
		location++;
	}

	// Translate to uk
	uk[0] = A_MIN + itv * sample_idx[0];
	uk[1] = B_MIN + itv * sample_idx[1];
	uk[2] = T_MIN + itv * sample_idx[2];
	uk[3] = A_MIN + itv * sample_idx[3];
	uk[4] = B_MIN + itv * sample_idx[4];
	uk[5] = T_MIN + itv * sample_idx[5];
	uk[6] = A_MIN + itv * sample_idx[6];
	uk[7] = B_MIN + itv * sample_idx[7];
	uk[8] = T_MIN + itv * sample_idx[8];

	// Get sk
	forward_kinematics_3(sk, uk[0], uk[1], uk[2],
							 uk[3], uk[4], uk[5],
							 uk[6], uk[7], uk[8]);

	double error;
	double inv_jcb[45];
	MatrixXd inv_jcb_mat(9, 5);
	VectorXd vec_uq(9);

	for (unsigned long int i = 0; i < 9*5; i++) {
		inv_jcb[i] = inv_jcb_data[tid*9*5 + i];
	}

	// Compute inverse kinematics
	VectorXd vec_sq(5);
	pointerToVector2(vec_sq, sq, 5);
	VectorXd vec_sk(5);
	pointerToVector2(vec_sk, sk, 5);
	VectorXd vec_uk(9);
	pointerToVector2(vec_uk, uk, 9);

	pointerToMatrix(inv_jcb_mat, &inv_jcb[0], 9, 5);
	vec_uq = inv_jcb_mat * (vec_sq - vec_sk) + vec_uk;
	// Free inv_jcb_mat, vec_sq, vec_sk, vec_uk
	inv_jcb_mat.resize(0, 0);
	vec_sq.resize(0);
	vec_sk.resize(0);
	vec_uk.resize(0);

	double uq[9];
	vectorToArray(uq, vec_uq, 9);
	// Free vec_uq
	vec_uq.resize(0);
	// Record uq
	for (int j = 0; j < 9; j++) {
		uqs[tid*9 + j] = uq[j];
		if (j%3 == 2 && uq[j] < 0.0) {
			uqs[tid*9 + j] = 0.0;
		}
	}

	// Compute forward result
	double sf[5];
	sf[0] = 0.0; sf[1] = 0.0; sf[2] = 0.0; sf[3] = 0.0; sf[4] = 0.0;
	forward_kinematics_3(&sf[0], uq);

	// Compute error and record
	error = euclidean_distance(sf, sq, 2);
	errors[tid] = error;
}
