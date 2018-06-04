#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>

#include <iostream>
#include <stdio.h>
#include <time.h>
#include <fstream>
#include "globals.h"
#include <Eigen/QR>
#include "Unicycle.cuh"
#include "utility.cuh"
#include "Files.h"


const double A_MAX1 = 1.0;
const double A_MIN1 = -1.0;
const double B_MAX1 = 1.0;
const double B_MIN1 = -1.0;
const double T_MAX1 = 1.0;
const double T_MIN1 = 0.0;

const double A_MAX2 = 1.0;
const double A_MIN2 = -1.0;
const double B_MAX2 = 1.0;
const double B_MIN2 = -1.0;
const double T_MAX2 = 1.0;
const double T_MIN2 = 0.0;

const double A_MAX3 = 1.0;
const double A_MIN3 = -1.0;
const double B_MAX3 = 1.0;
const double B_MIN3 = -1.0;
const double T_MAX3 = 1.0;
const double T_MIN3 = 0.0;

const double V_MAX = 120000.0/3600.0;
const double W_MAX = PI;


using namespace std;
using namespace Eigen;

#define cudaCheckErrors(msg) \
do { \
	cudaError_t __err = cudaGetLastError(); \
	if (__err != cudaSuccess) { \
		fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
			msg, cudaGetErrorString(__err), \
			__FILE__, __LINE__); \
		fprintf(stderr, "*** FAILED - ABORTING\n"); \
		exit(1); \
	} \
} while (0)

__global__ void linear_approximation(double* u, double* sq, double* inv_jcb_data, int* indicator,
								 double* uk_data, double* sk_data, unsigned long int data_num,
								 double* abs_errors, double* rlt_errors, double* uqs);
void generate_u_with_constraints(double** u, int gpu_num);



// Kernels for uniform sampling with angle constraints
__global__ void uniform_sampling(double* uk_record, double* sk_record,
								 double itv, int a_num, int b_num, int t_num, unsigned long int data_num) {
	unsigned long int blockId = blockIdx.x //1D
	        + blockIdx.y * gridDim.x //2D
	        + gridDim.x * gridDim.y * blockIdx.z; //3D
	unsigned long int tid = blockId * blockDim.x + threadIdx.x;
	if (tid >= data_num) {return;}

	double* uk = new double[9];
	double* sk = new double[5];
	// Initialite s0
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
	uk[0] = A_MIN1 + itv * sample_idx[0];
	uk[1] = B_MIN1 + itv * sample_idx[1];
	uk[2] = T_MIN1 + itv * sample_idx[2];
	uk[3] = A_MIN2 + itv * sample_idx[3];
	uk[4] = B_MIN2 + itv * sample_idx[4];
	uk[5] = T_MIN2 + itv * sample_idx[5];
	uk[6] = A_MIN3 + itv * sample_idx[6];
	uk[7] = B_MIN3 + itv * sample_idx[7];
	uk[8] = T_MIN3 + itv * sample_idx[8];

	// Get sk
	for (int i = 0; i < 3; i++) {
		forward_kinematics(sk, uk[i*3 + 0], uk[i*3 + 1], uk[i*3 + 2]);
	}

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

// Sample the space with angle constraints
void cuda_uniform_sampling() {
	double itv = 0.5;
	// Num of data
	int a_num = (A_MAX1 - A_MIN1) / itv + 1;
	int b_num = (B_MAX1 - B_MIN1) / itv + 1;
	int t_num = (T_MAX1 - T_MIN1) / itv + 1;
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
	cudaFree(dev_uk_record);
	cudaFree(dev_sk_record);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	// Inverse Jacobian
	double s0[5] = {0.0};
	MatrixXd jcb(5, 9);
	MatrixXd inv_jcb(9, 5);
	double* inv_jcb_record = new double[data_num*9*5];
	unsigned long int idx = 0;
	for (unsigned long int i = 0; i < data_num; i++) {
		// Get jcb
		jcb = jacobian(&s0[0], uk_record[i*9 + 0], uk_record[i*9 + 1], uk_record[i*9 + 2],
							   uk_record[i*9 + 3], uk_record[i*9 + 4], uk_record[i*9 + 5],
							   uk_record[i*9 + 6], uk_record[i*9 + 7], uk_record[i*9 + 8]);

		inv_jcb = pinv(jcb);

		for (int c = 0; c < 5; c++) {
			for (int r = 0; r < 9; r++) {
				inv_jcb_record[idx++] = inv_jcb(r, c);
			}
		}
	}

	cout << "Start Writing" << endl;

	Write_File(uk_record, sk_record, inv_jcb_record, data_num);

	cout << "Done" << endl;

	free(uk_record);

	free(sk_record);

	free(inv_jcb_record);
}


__global__ void random_sampling_with_constraints(double* uk_record, double* sk_record, unsigned long int data_num) {
	unsigned long int blockId = blockIdx.x //1D
	        + blockIdx.y * gridDim.x //2D
	        + gridDim.x * gridDim.y * blockIdx.z; //3D
	unsigned long int tid = blockId * blockDim.x + threadIdx.x;
	if (tid >= data_num) {return;}

	// Init curand
	curandState state;
	curand_init(0, tid, 0, &state);

	double* uk = new double[9];
	double* sk = new double[5];

	bool nice = false; // Nice trajectory?

	// Loop until find a control for nice trajectory
	while (!nice) {
		// Random uk
		uk[0] = A_MIN1 + (A_MAX1 - A_MIN1) * curand_uniform(&state);
		uk[1] = B_MIN1 + (B_MAX1 - B_MIN1) * curand_uniform(&state);
		uk[2] = T_MIN1 + (T_MAX1 - T_MIN1) * curand_uniform(&state);
		uk[3] = A_MIN2 + (A_MAX2 - A_MIN2) * curand_uniform(&state);
		uk[4] = B_MIN2 + (B_MAX2 - B_MIN2) * curand_uniform(&state);
		uk[5] = T_MIN2 + (T_MAX2 - T_MIN2) * curand_uniform(&state);
		uk[6] = A_MIN3 + (A_MAX3 - A_MIN3) * curand_uniform(&state);
		uk[7] = B_MIN3 + (B_MAX3 - B_MIN3) * curand_uniform(&state);
		uk[8] = T_MIN3 + (T_MAX3 - T_MIN3) * curand_uniform(&state);

		// Get sk
		// Reset s0
		for (int i = 0; i < 5; i++) {
			sk[i] = 0.0;
		}
		double z[4] = {0.0}; // z0, z1, z2, z3
		for (int i = 0; i < 3; i++) {
			forward_kinematics(sk, uk[i*3 + 0], uk[i*3 + 1], uk[i*3 + 2]);
			z[i+1] = sk[2];

			// Angle check
			if (abs(z[i+1] - z[i]) >= PII) {
				nice = false;
				break;
			}
			else if ((i >= 1) && (abs(z[i+1] - z[i-1]) >= PII)) {
				nice = false;
				break;
			}
			else if ((i >= 2) && (abs(z[i+1] - z[i-2]) >= PII)) {
				nice = false;
				break;
			}
			else {
				nice = true;
			}
		}
	}

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


void cuda_random_sampling_with_constraints(unsigned long int data_num) {
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
	random_sampling_with_constraints<<<block_num, thread_num>>>(dev_uk_record, dev_sk_record, data_num);

	// Fetch the data from device to host
	cudaMemcpy(uk_record, dev_uk_record, (data_num*9)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(sk_record, dev_sk_record, (data_num*5)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(dev_uk_record);
	cudaFree(dev_sk_record);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	// Inverse Jacobian
	double s0[5] = {0.0};
	MatrixXd jcb(5, 9);
	MatrixXd inv_jcb(9, 5);
	double* inv_jcb_record = new double[data_num*9*5];
	unsigned long int idx = 0;
	for (unsigned long int i = 0; i < data_num; i++) {
		// Get jcb
		jcb = jacobian(&s0[0], uk_record[i*9 + 0], uk_record[i*9 + 1], uk_record[i*9 + 2],
							   uk_record[i*9 + 3], uk_record[i*9 + 4], uk_record[i*9 + 5],
							   uk_record[i*9 + 6], uk_record[i*9 + 7], uk_record[i*9 + 8]);

		inv_jcb = pinv(jcb);

		for (int c = 0; c < 5; c++) {
			for (int r = 0; r < 9; r++) {
				inv_jcb_record[idx++] = inv_jcb(r, c);
			}
		}
	}

	cout << "Start Writing" << endl;

	Write_File(uk_record, sk_record, inv_jcb_record, data_num);

	cout << "Done" << endl;

	free(uk_record);

	free(sk_record);

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
	int a_num = (A_MAX1 - A_MIN1) / itv + 1;
	int b_num = (B_MAX1 - B_MIN1) / itv + 1;
	int t_num = (T_MAX1 - T_MIN1) / itv + 1;
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
	double* jcb_data = new double[data_num*9*5];
	Read_File(uk_data, sk_data, jcb_data, false);

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
	//int progress;
	double error_min = 100.0;
	double* uk_min = new double[9];
	double* sk_min = new double[5];
	double* uq_min = new double[9];

	for (unsigned long int i = 0; i < data_num; i++) {
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

		//progress = int(double(i) / double(data_num) * 100.0);
		//cout << "\rProgress: " << progress << "%" << flush;
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


void cuda_linear_approx_experiment_with_constraints(int exp_num, unsigned long int kernel_num) {
	// Managing CUDA variables
	double** u = new double*[4];
	double** sq = new double*[4];
	double** inv_jcb_data = new double*[4];
	int** indicator = new int*[4];
	double** uk_data = new double*[4];
	double** sk_data = new double*[4];
	double** abs_errors = new double*[4];
	double** rlt_errors = new double*[4];
	double** uqs = new double*[4];	// The results of inverse kinematics
	unsigned long int data_per_gpu[4];
	double* kernel_time = new double[exp_num];

	// Create streams for multi gpus
	cudaStream_t streams[4];
	cudaSetDevice(0);
	cudaStreamCreate(&streams[0]);
	cudaSetDevice(1);
	cudaStreamCreate(&streams[1]);
	cudaSetDevice(2);
	cudaStreamCreate(&streams[2]);
	cudaSetDevice(3);
	cudaStreamCreate(&streams[3]);
	cudaCheckErrors("cudaStreamCreate fail");

	for (int gpu = 0; gpu < 4; gpu++) {
		// Switch gpu
		cudaSetDevice(gpu);
		// Enable more heap on CUDA: 512mb
		cudaDeviceSetLimit(cudaLimitMallocHeapSize, 512*1024*1024);

		data_per_gpu[gpu] = (kernel_num + 4 -1) / 4;
		if (gpu == 3) {
			data_per_gpu[gpu] = kernel_num - data_per_gpu[gpu]*3;
		}
		cout << data_per_gpu[gpu] << endl;
		cudaMallocManaged(&u[gpu], 9*sizeof(double));
		cudaMallocManaged(&sq[gpu], 5*sizeof(double));
		cudaMallocManaged(&inv_jcb_data[gpu], 9*5*data_per_gpu[gpu]*sizeof(double));
		cudaMallocManaged(&indicator[gpu], 1*data_per_gpu[gpu]*sizeof(int));
		cudaMallocManaged(&uk_data[gpu], 9*data_per_gpu[gpu]*sizeof(double));
		cudaMallocManaged(&sk_data[gpu], 5*data_per_gpu[gpu]*sizeof(double));
		cudaMallocManaged(&abs_errors[gpu], data_per_gpu[gpu]*sizeof(double));
		cudaMallocManaged(&rlt_errors[gpu], data_per_gpu[gpu]*sizeof(double));
		cudaMallocManaged(&uqs[gpu], 9*data_per_gpu[gpu]*sizeof(double));
	}

	// Attach memory to stream
	for (int gpu = 0; gpu < 4; gpu++) {
		cudaSetDevice(gpu);
		cudaStreamAttachMemAsync(streams[gpu], u[gpu]);
		cudaStreamAttachMemAsync(streams[gpu], sq[gpu]);
		cudaStreamAttachMemAsync(streams[gpu], inv_jcb_data[gpu]);
		cudaStreamAttachMemAsync(streams[gpu], indicator[gpu]);
		cudaStreamAttachMemAsync(streams[gpu], uk_data[gpu]);
		cudaStreamAttachMemAsync(streams[gpu], sk_data[gpu]);
		cudaStreamAttachMemAsync(streams[gpu], abs_errors[gpu]);
		cudaStreamAttachMemAsync(streams[gpu], rlt_errors[gpu]);
		cudaStreamAttachMemAsync(streams[gpu], uqs[gpu]);
	}
	for (int gpu = 0; gpu < 4; gpu++) {
		cudaSetDevice(gpu);
		cudaDeviceSynchronize();
	}

	// Read in data
	Read_File_MG(uk_data, sk_data, inv_jcb_data, data_per_gpu);

	// Start experiment
	// Set random seed
	srand(1);

	// Thread num and block num
	int thread_num = 256;
	dim3 block_num(64, 64, 16);

	// Initialize record arrays
	double** exp_us = new double*[exp_num];
	for (int i = 0; i < exp_num; i++) {
		exp_us[i] = new double[9];
	}

	double** abs_record_errors = new double*[exp_num];
	double** rlt_record_errors = new double*[exp_num];

	double** abs_uqs = new double*[exp_num];
	double** abs_uks = new double*[exp_num];
	double** abs_sds = new double*[exp_num];
	double** rlt_uqs = new double*[exp_num];
	double** rlt_uks = new double*[exp_num];
	double** rlt_sds = new double*[exp_num];

	for (int i = 0; i < exp_num; i++) {
		abs_record_errors[i] = new double[10];
		rlt_record_errors[i] = new double[10];
		abs_uqs[i] = new double[9*10];
		abs_uks[i] = new double[9*10];
		abs_sds[i] = new double[5*10];
		rlt_uqs[i] = new double[9*10];
		rlt_uks[i] = new double[9*10];
		rlt_sds[i] = new double[5*10];
	}

	clock_t begin = clock();
	for (int exp = 0; exp < exp_num; exp++) {
		// Make a random query point with constraints
		clock_t rand_begin = clock();
		generate_u_with_constraints(u, 4);
		clock_t rand_end = clock();
		double rand_elapsed_sec = double(rand_end - rand_begin) / CLOCKS_PER_SEC;
		cout << "Random Generation time: " << rand_elapsed_sec << endl;

		for (int i = 0; i < 4; i++) {
			sq[i][0] = 0.0; sq[i][1] = 0.0; sq[i][2] = 0.0;
			sq[i][3] = 0.0; sq[i][4] = 0.0;
			forward_kinematics_3(sq[i], u[i]);
		}

		// Launch kernels
		clock_t kernel_begin = clock();
		for (int gpu = 0; gpu < 4; gpu++) {
			cudaSetDevice(gpu);

			linear_approximation<<<block_num, thread_num, 0, streams[gpu]>>>(u[gpu], sq[gpu], inv_jcb_data[gpu], indicator[gpu],
														uk_data[gpu], sk_data[gpu], data_per_gpu[gpu],
														abs_errors[gpu], rlt_errors[gpu], uqs[gpu]);
		}

		// Wait for GPU to finish before accessing on host
		for (int gpu = 0; gpu < 4; gpu++) {
			cudaSetDevice(gpu);
			cudaDeviceSynchronize();
		}
		clock_t kernel_end = clock();
		double kernel_elapsed_sec = double(kernel_end - kernel_begin) / CLOCKS_PER_SEC;
		cout << "Kernels time: " << kernel_elapsed_sec << endl;
		kernel_time[exp] = kernel_elapsed_sec;

		clock_t record_begin = clock();
		// Record u
		for (int i = 0; i < 9; i++) {
			exp_us[exp][i] = u[0][i];
		}

		// Get best 10 result with abs error
		double error_min[10];
		unsigned long int idx_min[10][2];
		for (int i = 0; i < 10; i++) {
			error_min[i] = 10000.0;
			for (int j = 0; j < 2; j++) {
				idx_min[i][j] = 0;
			}
		}
		get_best_results(&error_min[0], idx_min, abs_errors, indicator, &data_per_gpu[0], 10);
		// Bubble sorting error_min and idx_min
		for (int i = 0; i < 10-1; i++) {
			for (int j = 0; j < 10-i-1; j++) {
				if (error_min[j] > error_min[j+1]) {
					// Swap error
					double temp_error = error_min[j];
					error_min[j] = error_min[j+1];
					error_min[j+1] = temp_error;
					// Swap idx
					double temp_idx[2];
					temp_idx[0] = idx_min[j][0]; temp_idx[1] = idx_min[j][1];
					idx_min[j][0] = idx_min[j+1][0]; idx_min[j][1] = idx_min[j+1][1];
					idx_min[j+1][0] = temp_idx[0]; idx_min[j+1][1] = temp_idx[1];
				}
			}
		}

		// Recording data related to abs error
		for (int i = 0; i < 10; i++) {
			int gpu_idx = idx_min[i][0];
			unsigned long int no_in_gpu_idx = idx_min[i][1];

			abs_record_errors[exp][i] = error_min[i];

			for (int j = 0; j < 9; j++) {
				abs_uqs[exp][i*9 + j] = uqs[gpu_idx][no_in_gpu_idx * 9 + j];
				abs_uks[exp][i*9 + j] = uk_data[gpu_idx][no_in_gpu_idx * 9 + j];
			}

			// s is where we really reach
			double s[5] = { 0.0 };
			forward_kinematics_3(&s[0], abs_uqs[exp][i*9 + 0], abs_uqs[exp][i*9 + 1], abs_uqs[exp][i*9 + 2],
										abs_uqs[exp][i*9 + 3], abs_uqs[exp][i*9 + 4], abs_uqs[exp][i*9 + 5],
										abs_uqs[exp][i*9 + 6], abs_uqs[exp][i*9 + 7], abs_uqs[exp][i*9 + 8]);
			for (int j = 0; j < 5; j++) {
				abs_sds[exp][i*5 + j] = s[j] - sq[0][j];
			}
		}

		// Get best 10 result with rlt error
		for (int i = 0; i < 10; i++) {
			error_min[i] = 10000.0;
			for (int j = 0; j < 2; j++) {
				idx_min[i][j] = 0;
			}
		}
		get_best_results(&error_min[0], idx_min, rlt_errors, indicator, &data_per_gpu[0], 10);
		// Bubble sorting error_min and idx_min
		for (int i = 0; i < 10-1; i++) {
			for (int j = 0; j < 10-i-1; j++) {
				if (error_min[j] > error_min[j+1]) {
					// Swap error
					double temp_error = error_min[j];
					error_min[j] = error_min[j+1];
					error_min[j+1] = temp_error;
					// Swap idx
					double temp_idx[2];
					temp_idx[0] = idx_min[j][0]; temp_idx[1] = idx_min[j][1];
					idx_min[j][0] = idx_min[j+1][0]; idx_min[j][1] = idx_min[j+1][1];
					idx_min[j+1][0] = temp_idx[0]; idx_min[j+1][1] = temp_idx[1];
				}
			}
		}

		// Recording data related to rlt error
		for (int i = 0; i < 10; i++) {
			int gpu_idx = idx_min[i][0];
			unsigned long int no_in_gpu_idx = idx_min[i][1];

			rlt_record_errors[exp][i] = error_min[i];

			for (int j = 0; j < 9; j++) {
				rlt_uqs[exp][i*9 + j] = uqs[gpu_idx][no_in_gpu_idx * 9 + j];
				rlt_uks[exp][i*9 + j] = uk_data[gpu_idx][no_in_gpu_idx * 9 + j];
			}

			// s is where we really reach
			double s[5] = { 0.0 };
			forward_kinematics_3(&s[0], rlt_uqs[exp][i*9 + 0], rlt_uqs[exp][i*9 + 1], rlt_uqs[exp][i*9 + 2],
										rlt_uqs[exp][i*9 + 3], rlt_uqs[exp][i*9 + 4], rlt_uqs[exp][i*9 + 5],
										rlt_uqs[exp][i*9 + 6], rlt_uqs[exp][i*9 + 7], rlt_uqs[exp][i*9 + 8]);
			for (int j = 0; j < 5; j++) {
				rlt_sds[exp][i*5 + j] = s[j] - sq[0][j];
			}
		}
		clock_t record_end = clock();
		double record_elapsed_sec = double(record_end - record_begin) / CLOCKS_PER_SEC;
		cout << "Recording time: " << record_elapsed_sec << endl;
	}
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << "Query Time: " << elapsed_secs << endl;

	Write_File_Exp(exp_us, abs_record_errors, rlt_record_errors,
						abs_uqs, rlt_uqs, abs_uks, rlt_uks,
						abs_sds, rlt_sds, kernel_time, exp_num);
}


int main()
{
	//cuda_uniform_sampling();
	//cuda_random_sampling_with_constraints(60000000);
	cuda_linear_approx_experiment_with_constraints(10000, 421875);
	//Find_Max();

    return 0;
}


__global__ void linear_approximation(double* u, double* sq, double* inv_jcb_data, int* indicator,
								 double* uk_data, double* sk_data, unsigned long int data_num,
								 double* abs_errors, double* rlt_errors, double* uqs) {
	// Get tid
	unsigned long int blockId = blockIdx.x //1D
		        + blockIdx.y * gridDim.x //2D
		        + gridDim.x * gridDim.y * blockIdx.z; //3D
	unsigned long int tid = blockId * blockDim.x + threadIdx.x;
	if (tid >= data_num) {return;}

	// Get the kernel
	double uk[9];
	double sk[5];

	for (int i = 0; i < 9; i++) {
		uk[i] = uk_data[tid*9 + i];
	}
	for (int i = 0; i < 5; i++) {
		sk[i] = sk_data[tid*5 + i];
	}

	VectorXd sk_vec = pointerToVector(&sk[0], 5);
	VectorXd uk_vec = pointerToVector(&uk[0], 9);
	VectorXd sq_vec = pointerToVector(sq, 5);

	// Get inv_jcb
	MatrixXd inv_jcb(9, 5);
	int idx = 0;
	for (int c = 0; c < 5; c++) {
		for(int r = 0; r < 9; r++) {
			inv_jcb(r, c) = inv_jcb_data[tid*9*5 + idx];
			idx++;
		}
	}

	// Get inverse kinematics: uq
	VectorXd uq_vec(9);
	uq_vec = inv_jcb * (sq_vec - sk_vec) + uk_vec;
	for (int i = 0; i < 9; i++) {
		// if t < 0, then set t=0
		if ((i%3 == 2) && (uq_vec(i) < 0.0)) {
			uq_vec(i) = 0.0;
		}
		// Recording
		uqs[tid*9 + i] = uq_vec(i);
	}

	// Get s
	double s[5] = {0.0};
	for (int i = 0; i < 3; i++) {
		forward_kinematics(&s[0], uq_vec(i*3 + 0), uq_vec(i*3 + 1), uq_vec(i*3 + 2));
	}

	bool nice = true; //angle_check(uq_vec(1), uq_vec(4), uq_vec(7), uq_vec(2), uq_vec(5), uq_vec(8));

	if (nice) {
		// Record
		indicator[tid] = 1;
	}
	else {
		indicator[tid] = 0;
	}

	// Get abs error
	double s_diff[5];
	s_diff[0] = s[0] - sq[0]; s_diff[1] = s[1] - sq[1];
	s_diff[2] = angleDiff(s[2], sq[2]);
	s_diff[3] = s[3] - sq[3]; s_diff[4] = s[4] - sq[4];
	double s0[5] = {0.0};
	abs_errors[tid] = euclidean_distance(&s_diff[0], &s0[0], 5);

	// Get rlt error
	// Calculate euclidean dist
	double euc_dist = calculate_dist_travelled(u[0], u[3], u[6], u[2], u[5], u[8]);
	// Calculate theta dist
	double theta_dist = calculate_dist_travelled(u[1], u[4], u[7], u[2], u[5], u[8]);
	double v_changed = abs(u[0])*u[2] + abs(u[3])*u[5] + abs(u[6])*u[8];
	double w_changed = abs(u[1])*u[2] + abs(u[4])*u[5] + abs(u[7])*u[8];
	rlt_errors[tid] = abs(s_diff[0])/euc_dist + abs(s_diff[1])/euc_dist + abs(s_diff[2])/theta_dist
			+ abs(s_diff[3])/v_changed + abs(s_diff[4])/w_changed;

	sk_vec.resize(0);
	uk_vec.resize(0);
	sq_vec.resize(0);
	inv_jcb.resize(0, 0);
	uq_vec.resize(0);
}

void generate_u_with_constraints(double** u, int gpu_num) {
	bool nice = false; // Nice trajectory?
	double ur[9];
	double s[5];

	while (!nice) {
		// Make a random query point
		ur[0] = fRand(A_MIN1, A_MAX1); ur[1] = fRand(B_MIN1, B_MAX1); ur[2] = fRand(T_MIN1, T_MAX1);
		ur[3] = fRand(A_MIN2, A_MAX2); ur[4] = fRand(B_MIN2, B_MAX2); ur[5] = fRand(T_MIN2, T_MAX2);
		ur[6] = fRand(A_MIN3, A_MAX3); ur[7] = fRand(B_MIN3, B_MAX3); ur[8] = fRand(T_MIN3, T_MAX3);

		nice = angle_check(ur[1], ur[4], ur[7], ur[2], ur[5], ur[8]);
	}

	// Get sk
	// Reset s0
	for (int i = 0; i < 5; i++) {
		s[i] = 0.0;
	}
	for (int i = 0; i < 3; i++) {
		forward_kinematics(s, ur[i*3 + 0], ur[i*3 + 1], ur[i*3 + 2]);
	}

	// Assigning to u
	for (int i = 0; i < gpu_num; i++) {
		for (int dim = 0; dim < 9; dim++) {
			u[i][dim] = ur[dim];
		}
	}

	return;
}
