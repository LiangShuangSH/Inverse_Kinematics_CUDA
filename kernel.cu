#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <time.h>
#include <fstream>
#include "globals.h"
#include <Eigen/QR>
#include "Unicycle.cuh"
#include "utility.cuh"
#include "Files.h"


const double A_MAX = 2.8;
const double A_MIN = 1.2;
const double B_MAX = 2.8;
const double B_MIN = 1.2;
const double T_MAX = 1.6;
const double T_MIN = 0.8;
const double V_MAX = 120000.0/3600.0;
const double W_MAX = PI;

const double error_unit_dist = 0.5*A_MAX*pow(3*(T_MAX-T_MIN), 2) / 2 * 0.1;
const double error_unit_orien = 0.5*B_MAX*pow(3*(T_MAX-T_MIN), 2) / 2 * 0.1;
const double error_unit_v = A_MAX*pow(3*(T_MAX-T_MIN), 2) / 2 * 0.1;
const double error_unit_w = B_MAX*pow(3*(T_MAX-T_MIN), 2) / 2 * 0.1;

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

void test_host_code_um();
__global__ void test_kernel_code(double* sq, double* inv_jcb_data,
								 double* uk_data, double* sk_data, unsigned long int data_num,
								 double* errors, double* uqs,
								 double error_unit_dist, double error_unit_orien, double error_unit_v, double error_unit_w);


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

	// Inverse Jacobian
	double* jcb_record = new double[data_num*9*5];
	double* sk = new double[5];
	double* uk = new double[9];
	int idx = 0;
	for (unsigned long int i = 0; i < data_num; i++) {
		for (int j = 0; j < 5; j++) {
			sk[j] = sk_record[i*5 + j];
		}
		for (int j = 0; j < 9; j++) {
			uk[j] = sk_record[i*9 + j];
		}
		MatrixXd jcb = jacobian(sk, uk[0], uk[1], uk[2], uk[3], uk[4], uk[5], uk[6], uk[7], uk[8]);
		MatrixXd inv_jcb = pinv(jcb);

		for (int c = 0; c < 5; c++) {
			for (int r = 0; r < 9; r++) {
				jcb_record[idx++] = inv_jcb(r, c);
			}
		}
	}

	free(sk);
	free(uk);

	cout << "Start Writing" << endl;

	Write_File(uk_record, sk_record, jcb_record, data_num);

	cout << "Done" << endl;

	free(uk_record);
	cudaFree(dev_uk_record);

	free(sk_record);
	cudaFree(dev_sk_record);
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

void cuda_linear_approx_experiment(int exp_num) {
	double itv = 0.2;
	// Num of data
	int a_num = (A_MAX - A_MIN) / itv + 1;
	int b_num = (B_MAX - B_MIN) / itv + 1;
	int t_num = (T_MAX - T_MIN) / itv + 1;
	unsigned long int data_num = pow(a_num * b_num * t_num, 3);

	// Managing CUDA variables
	double** sq = new double*[4];
	double** inv_jcb_data = new double*[4];
	double** uk_data = new double*[4];
	double** sk_data = new double*[4];
	double** errors = new double*[4];
	double** uqs = new double*[4];
	unsigned long int data_per_gpu[4];

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

		data_per_gpu[gpu] = (data_num + 4 -1) / 4;
		if (gpu == 3) {
			data_per_gpu[gpu] = data_num - data_per_gpu[gpu]*3;
		}
		cudaMallocManaged(&sq[gpu], 5*sizeof(double));
		cudaMallocManaged(&inv_jcb_data[gpu], 9*5*data_per_gpu[gpu]*sizeof(double));
		cudaMallocManaged(&uk_data[gpu], 9*data_per_gpu[gpu]*sizeof(double));
		cudaMallocManaged(&sk_data[gpu], 5*data_per_gpu[gpu]*sizeof(double));
		cudaMallocManaged(&errors[gpu], data_per_gpu[gpu]*sizeof(double));
		cudaMallocManaged(&uqs[gpu], 9*data_per_gpu[gpu]*sizeof(double));
	}

	// Attach memory to stream
	for (int gpu = 0; gpu < 4; gpu++) {
		cudaSetDevice(gpu);
		cudaStreamAttachMemAsync(streams[gpu], sq[gpu]);
		cudaStreamAttachMemAsync(streams[gpu], inv_jcb_data[gpu]);
		cudaStreamAttachMemAsync(streams[gpu], uk_data[gpu]);
		cudaStreamAttachMemAsync(streams[gpu], sk_data[gpu]);
		cudaStreamAttachMemAsync(streams[gpu], errors[gpu]);
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
	double* exp_errors = new double[exp_num];
	double** exp_us = new double*[exp_num];
	for (int i = 0; i < exp_num; i++) {
		exp_us[i] = new double[9];
	}
	double** exp_uqs = new double*[exp_num];
	for (int i = 0; i < exp_num; i++) {
		exp_uqs[i] = new double[9];
	}

	double* u = new double[9];
	clock_t begin = clock();
	for (int exp = 0; exp < exp_num; exp++) {
		// Make a random query point
		u[0] = fRand(A_MIN, A_MAX); u[1] = fRand(B_MIN, B_MAX); u[2] = fRand(T_MIN, T_MAX);
		u[3] = fRand(A_MIN, A_MAX); u[4] = fRand(B_MIN, B_MAX); u[5] = fRand(T_MIN, T_MAX);
		u[6] = fRand(A_MIN, A_MAX); u[7] = fRand(B_MIN, B_MAX); u[8] = fRand(T_MIN, T_MAX);

		for (int i = 0; i < 4; i++) {
			sq[i][0] = 0.0; sq[i][1] = 0.0; sq[i][2] = 0.0;
			sq[i][3] = 0.0; sq[i][4] = 0.0; sq[i][5] = 0.0;
			sq[i][6] = 0.0; sq[i][7] = 0.0; sq[i][8] = 0.0;
			forward_kinematics_3(sq[i], u);
		}

		// Launch kernels
		for (int gpu = 0; gpu < 4; gpu++) {
			cudaSetDevice(gpu);

			test_kernel_code<<<block_num, thread_num, 0, streams[gpu]>>>(sq[gpu], inv_jcb_data[gpu],
														uk_data[gpu], sk_data[gpu], data_per_gpu[gpu],
														errors[gpu], uqs[gpu],
														error_unit_dist, error_unit_orien, error_unit_v, error_unit_w);
		}

		// Wait for GPU to finish before accessing on host
		for (int gpu = 0; gpu < 4; gpu++) {
			cudaSetDevice(gpu);
			cudaDeviceSynchronize();
		}

		// Get best result
		double error_min = errors[0][0];
		unsigned long int idx_min[2];
		for (int gpu = 0; gpu < 4; gpu++) {
			for (unsigned long int i = 0; i < data_per_gpu[gpu]; i++) {
				if (error_min > errors[gpu][i]) {
					error_min = errors[gpu][i];
					idx_min[0] = gpu;
					idx_min[1] = i;
				}
			}
		}

		// Recording
		exp_errors[exp] = error_min;
		for (int i = 0; i < 9; i++) {
			exp_us[exp][i] = u[i];
			exp_uqs[exp][i] = uqs[idx_min[0]][idx_min[1] * 9 + i];
		}
	}
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << "Query Time: " << elapsed_secs << endl;

	Write_File_Exp(exp_errors, exp_us, exp_uqs, exp_num);
}


int main()
{
	//cuda_find_inverse_kinematics();
	//cuda_uniform_sampling();
	cuda_linear_approx_experiment(5);

    return 0;
}

void test_host_code_um() {
	double itv = 0.2;
	// Num of data
	int a_num = (A_MAX - A_MIN) / itv + 1;
	int b_num = (B_MAX - B_MIN) / itv + 1;
	int t_num = (T_MAX - T_MIN) / itv + 1;
	unsigned long int data_num = pow(a_num * b_num * t_num, 3);

	// Managing CUDA variables
	double** sq = new double*[4];
	double** inv_jcb_data = new double*[4];
	double** uk_data = new double*[4];
	double** sk_data = new double*[4];
	double** errors = new double*[4];
	double** uqs = new double*[4];
	unsigned long int data_per_gpu[4];

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

		data_per_gpu[gpu] = (data_num + 4 -1) / 4;
		if (gpu == 3) {
			data_per_gpu[gpu] = data_num - data_per_gpu[gpu]*3;
		}
		cudaMallocManaged(&sq[gpu], 5*sizeof(double));
		cudaMallocManaged(&inv_jcb_data[gpu], 9*5*data_per_gpu[gpu]*sizeof(double));
		cudaMallocManaged(&uk_data[gpu], 9*data_per_gpu[gpu]*sizeof(double));
		cudaMallocManaged(&sk_data[gpu], 5*data_per_gpu[gpu]*sizeof(double));
		cudaMallocManaged(&errors[gpu], data_per_gpu[gpu]*sizeof(double));
		cudaMallocManaged(&uqs[gpu], 9*data_per_gpu[gpu]*sizeof(double));
	}

	// Read in data
	clock_t begin = clock();
	Read_File_MG(uk_data, sk_data, inv_jcb_data, data_per_gpu);
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << "Read Time: " << elapsed_secs << endl;

	for (int i = 0; i < 5; i++) {
		cout << "sk: " << sk_data[3][data_per_gpu[3]*5 -1 - i] << " ";
	}
	cout << "" << endl;

	for (int i = 0; i < 9*5; i++) {
		cout << inv_jcb_data[3][data_per_gpu[3]*9*5-1-i] << " ";
	}
	cout << "" << endl;

	cout << "Finish Reading" << endl;

	// Make a query point
	double* u = new double[9];
	u[0] = 0.4; u[1] = 0.4; u[2] = 0.4;
	u[3] = 0.5; u[4] = 0.5; u[5] = 0.4;
	u[6] = 0.4; u[7] = 0.4; u[8] = 0.4;

	for (int i = 0; i < 4; i++) {
		sq[i][0] = 0.0; sq[i][1] = 0.0; sq[i][2] = 0.0;
		sq[i][3] = 0.0; sq[i][4] = 0.0; sq[i][5] = 0.0;
		sq[i][6] = 0.0; sq[i][7] = 0.0; sq[i][8] = 0.0;
		forward_kinematics_3(sq[i], u);
	}

	for (int gpu = 0; gpu < 4; gpu++) {
		cudaSetDevice(gpu);
		cudaStreamAttachMemAsync(streams[gpu], sq[gpu]);
		cudaStreamAttachMemAsync(streams[gpu], inv_jcb_data[gpu]);
		cudaStreamAttachMemAsync(streams[gpu], uk_data[gpu]);
		cudaStreamAttachMemAsync(streams[gpu], sk_data[gpu]);
		cudaStreamAttachMemAsync(streams[gpu], errors[gpu]);
		cudaStreamAttachMemAsync(streams[gpu], uqs[gpu]);
	}
	for (int gpu = 0; gpu < 4; gpu++) {
		cudaSetDevice(gpu);
		cudaDeviceSynchronize();
	}

	// Perform query
	begin = clock();
	int thread_num = 256;
	dim3 block_num(64, 64, 16);
	for (int gpu = 0; gpu < 4; gpu++) {
		cudaSetDevice(gpu);

		test_kernel_code<<<block_num, thread_num, 0, streams[gpu]>>>(sq[gpu], inv_jcb_data[gpu],
													uk_data[gpu], sk_data[gpu], data_per_gpu[gpu],
													errors[gpu], uqs[gpu],
													error_unit_dist, error_unit_orien, error_unit_v, error_unit_w);
	}

	// Wait for GPU to finish before accessing on host
	for (int gpu = 0; gpu < 4; gpu++) {
		cudaSetDevice(gpu);
		cudaDeviceSynchronize();
	}

	end = clock();
	elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << "Query Time: " << elapsed_secs << endl;

	// Get best result
	double error_min = errors[0][0];
	unsigned long int idx_min[2];
	for (int gpu = 0; gpu < 4; gpu++) {
		for (unsigned long int i = 0; i < data_per_gpu[gpu]; i++) {
			if (error_min > errors[gpu][i]) {
				error_min = errors[gpu][i];
				idx_min[0] = gpu;
				idx_min[1] = i;
			}
		}
	}

	cout << "Best Result: " << endl;
	cout << "Error: " << error_min << endl;
	cout << "Uq: ";
	for(int i = 0; i < 9; i++) {
		cout << uqs[idx_min[0]][idx_min[1] * 9 + i] << " ";
	}
	cout << "" << endl;
}

__global__ void test_kernel_code(double* sq, double* inv_jcb_data,
								 double* uk_data, double* sk_data, unsigned long int data_num,
								 double* errors, double* uqs,
								 double error_unit_dist, double error_unit_orien, double error_unit_v, double error_unit_w) {
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

	// Get error
	double s[5] = {0.0};
	forward_kinematics_3(&s[0], uq_vec[0], uq_vec[1], uq_vec[2],
								uq_vec[3], uq_vec[4], uq_vec[5],
								uq_vec[6], uq_vec[7], uq_vec[8]);
	double euc_error = euclidean_distance(&s[0], sq, 2);
	errors[tid] = sqrt(pow(euc_error/error_unit_dist, 2) + pow((s[2]-sq[2])/error_unit_orien, 2)
					+ pow((s[3]-sq[3])/error_unit_v, 2) + pow((s[4]-sq[4])/error_unit_w, 2));
}

