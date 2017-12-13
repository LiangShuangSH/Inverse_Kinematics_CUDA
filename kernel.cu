#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <fstream>
#include "globals.h"
#include <Eigen/QR>
#include "Unicycle.cuh"
#include "utility.cuh"


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

void Write_File(double* uk_data, double* sk_data, int data_num);
void Read_File(double* uk_data, double* sk_data);

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

	cout << "Start Writing" << endl;

	Write_File(uk_record, sk_record, data_num);

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
	double* u = new double[9];
	u[0] = 0.4; u[1] = 0.4; u[2] = 0.4;
	u[3] = 0.5; u[4] = 0.5; u[5] = 0.4;
	u[6] = 0.4; u[7] = 0.4; u[8] = 0.4;
	double* sq = forward_kinematics_3(s0, u);

	double* dev_sq;
	cudaMalloc(&dev_sq, 5 * sizeof(double));
	cudaMemcpy(dev_sq, sq, 5*sizeof(double), cudaMemcpyHostToDevice);

	double* dist_record = new double[data_num];
	double* sk_data = new double[data_num*5];
	double* uk_data = new double[data_num*9];
	Read_File(uk_data, sk_data);

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
	ik_file << "Query: ";
	for (int i = 0; i < 5; i++) {
		ik_file << sq[i] << " ";
	}
	ik_file << "" << endl;

	double threshold = 0.5;
	double error;
	double uk[9];
	double sk[5];
	double progress;
	for (unsigned long int i = 0; i < data_num; i++) {
		// In state space
		// If the query point is close enough to the sample point
		// Do inverse kinematics
		if (dist_record[i] <= threshold) {
			// Assign uk, sk
			for (int j = 0; j < 9; j++) { uk[j] = uk_data[i*9 + j]; }
			for (int j = 0; j < 9; j++) { sk[j] = sk_data[i*5 + j]; }
			double* uq = inverse_kinematics_3(sq, uk, sk, s0);
			double* sf = forward_kinematics_3(s0, uq);
			error = euclidean_distance(sf, sq, 5);
			// Write file
			ik_file << "No." << i << ", " << "error: " <<error << endl;
			ik_file << "uk: ";
			for(int j = 0; j < 9; j++) { ik_file << uk[j] << " "; }
			ik_file << "" << endl;
			ik_file << "sk: ";
			for(int j = 0; j < 5; j++) { ik_file << sk[j] << " "; }
			ik_file << "" << endl;
		}
		progress = i / data_num;
		cout << int(progress*100.0) << "%\r";
	}
	ik_file.close();
}



int main()
{
	//cuda_find_inverse_kinematics();
	cuda_uniform_sampling();
    return 0;
}

void Write_File(double* uk_data, double* sk_data, int data_num) {
	ofstream uk_file;
	uk_file.open("./uk.data");
	ofstream sk_file;
	sk_file.open("./sk.data");

	for (int i = 0; i < data_num; i++) {
		for (int j = 0; j < 9; j++) {
			uk_file << uk_data[i*9 + j] << " ";
		}
		uk_file << "" << endl;

		for (int j = 0; j < 5; j++) {
			sk_file << sk_data[i*5 + j] << " ";
		}
		sk_file << "" << endl;
	}

	uk_file.close();
	sk_file.close();
}

void Read_File(double* uk_data, double* sk_data) {
	ifstream uk_file;
	uk_file.open("./uk.data");
	ifstream sk_file;
	sk_file.open("./sk.data");

	// Read uk
	int idx = 0;
	double a1, b1, t1, a2, b2, t2, a3, b3, t3;
	while (uk_file >> a1 >> b1 >> t1 >> a2 >> b2 >> t2 >> a3 >> b3 >> t3) {
		uk_data[idx++] = a1;
		uk_data[idx++] = b1;
		uk_data[idx++] = t1;
		uk_data[idx++] = a2;
		uk_data[idx++] = b2;
		uk_data[idx++] = t2;
		uk_data[idx++] = a3;
		uk_data[idx++] = b3;
		uk_data[idx++] = t3;
	}

	// Read sk
	idx = 0;
	double x, y, z, v, w;
	while (sk_file >> x >> y >> z >> v >> w) {
		sk_data[idx++] = x;
		sk_data[idx++] = y;
		sk_data[idx++] = z;
		sk_data[idx++] = v;
		sk_data[idx++] = w;
	}

	uk_file.close();
	sk_file.close();
}
