/*
 * Files.cpp
 *
 *  Created on: Jan 25, 2018
 *      Author: liang
 */

#include "Files.h"

using namespace std;
using namespace Eigen;

string directory = "./Experiments/GRID5_-1-10_-1-10_-1-10/";

void Write_File(double* uk_data, double* sk_data, double* inv_jcb_data, unsigned long int data_num) {
	ofstream uk_file;
	uk_file.open((directory + "uk.data").c_str());
	ofstream sk_file;
	sk_file.open((directory + "sk.data").c_str());
	ofstream jcb_file;
	jcb_file.open((directory + "inv_jcb.data").c_str());

	for (unsigned long int i = 0; i < data_num; i++) {
		for (int j = 0; j < 9; j++) {
			uk_file << uk_data[i*9 + j] << " ";
		}
		uk_file << "" << endl;

		for (int j = 0; j < 5; j++) {
			sk_file << sk_data[i*5 + j] << " ";
		}
		sk_file << "" << endl;

		for (int j = 0; j < 9*5; j++) {
			jcb_file << inv_jcb_data[i*9*5 + j] << " ";
		}
		jcb_file << "" << endl;
	}

	uk_file.close();
	sk_file.close();
	jcb_file.close();
}

void Read_File(double* uk_data, double* sk_data, double* inv_jcb_data, bool test) {
	ifstream uk_file;
	uk_file.open((directory + "uk.data").c_str());

	// Read uk
	unsigned long int idx = 0;
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

		if ((test == true) && (idx == 9*100)) { break; }
	}

	uk_file.close();

	ifstream sk_file;
	sk_file.open((directory + "sk.data").c_str());
	// Read sk
	idx = 0;
	double x, y, z, v, w;
	while (sk_file >> x >> y >> z >> v >> w) {
		sk_data[idx++] = x;
		sk_data[idx++] = y;
		sk_data[idx++] = z;
		sk_data[idx++] = v;
		sk_data[idx++] = w;

		if ((test == true) && (idx == 5*100)) { break; }
	}
	sk_file.close();

	ifstream inv_jcb_file;
	inv_jcb_file.open((directory + "inv_jcb.data").c_str());
	// Read inv_jcb
	string jcb;
	char delimiter = ' ';
	idx = 0;
	while (getline(inv_jcb_file, jcb, delimiter)) {
		inv_jcb_data[idx++] = atof(jcb.c_str());
		if ((test == true) && (idx == 9*5*100)) { break; }
	}
	inv_jcb_file.close();
}

void Read_Inv_Jcb(double* inv_jcb_data) {
	ifstream inv_jcb_file;
	inv_jcb_file.open("./inv_jcb.data");
	// Read inv_jcb
	string jcb;
	char delimiter = ' ';
	unsigned long int idx = 0;
	while (getline(inv_jcb_file, jcb, delimiter)) {
			inv_jcb_data[idx++] = atof(jcb.c_str());
		}
	inv_jcb_file.close();
}

void Read_Inv_Jcb_MG(double** inv_jcb_data, unsigned long int* data_per_gpu) {
	ifstream inv_jcb_file;
	inv_jcb_file.open("./inv_jcb.data");
	// Read inv_jcb
	string jcb;
	char delimiter = ' ';
	unsigned long int idx = 0;
	int gpu = 0;
	while (getline(inv_jcb_file, jcb, delimiter)) {
		inv_jcb_data[gpu][idx++] = atof(jcb.c_str());
		if (idx > 9*5*data_per_gpu[gpu]) {
			gpu++;
			idx = 0;
		}
	}
	inv_jcb_file.close();
}

void Read_File_MG(double** uk_data, double** sk_data, double** inv_jcb_data, unsigned long int* data_per_gpu) {
	ifstream uk_file;
	uk_file.open((directory + "uk.data").c_str());

	// Read uk
	unsigned long int idx = 0;
	int gpu = 0;
	double a1, b1, t1, a2, b2, t2, a3, b3, t3;
	while (uk_file >> a1 >> b1 >> t1 >> a2 >> b2 >> t2 >> a3 >> b3 >> t3) {
		uk_data[gpu][idx++] = a1;
		uk_data[gpu][idx++] = b1;
		uk_data[gpu][idx++] = t1;
		uk_data[gpu][idx++] = a2;
		uk_data[gpu][idx++] = b2;
		uk_data[gpu][idx++] = t2;
		uk_data[gpu][idx++] = a3;
		uk_data[gpu][idx++] = b3;
		uk_data[gpu][idx++] = t3;

		if (idx >= 9*data_per_gpu[gpu]) {
			gpu++;
			idx = 0;
		}
	}

	uk_file.close();

	ifstream sk_file;
	sk_file.open((directory + "sk.data").c_str());
	// Read sk
	idx = 0;
	gpu = 0;
	double x, y, z, v, w;
	while (sk_file >> x >> y >> z >> v >> w) {
		sk_data[gpu][idx++] = x;
		sk_data[gpu][idx++] = y;
		sk_data[gpu][idx++] = z;
		sk_data[gpu][idx++] = v;
		sk_data[gpu][idx++] = w;

		if (idx >= 5*data_per_gpu[gpu]) {
			gpu++;
			idx = 0;
		}
	}
	sk_file.close();

	ifstream inv_jcb_file;
	inv_jcb_file.open((directory + "inv_jcb.data").c_str());
	// Read inv_jcb
	double IJ[9][5];
	idx = 0;
	gpu = 0;

	while (inv_jcb_file >> IJ[0][0] >> IJ[1][0] >> IJ[2][0] >> IJ[3][0] >> IJ[4][0] >> IJ[5][0] >> IJ[6][0] >> IJ[7][0] >> IJ[8][0]
						>> IJ[0][1] >> IJ[1][1] >> IJ[2][1] >> IJ[3][1] >> IJ[4][1] >> IJ[5][1] >> IJ[6][1] >> IJ[7][1] >> IJ[8][1]
						>> IJ[0][2] >> IJ[1][2] >> IJ[2][2] >> IJ[3][2] >> IJ[4][2] >> IJ[5][2] >> IJ[6][2] >> IJ[7][2] >> IJ[8][2]
						>> IJ[0][3] >> IJ[1][3] >> IJ[2][3] >> IJ[3][3] >> IJ[4][3] >> IJ[5][3] >> IJ[6][3] >> IJ[7][3] >> IJ[8][3]
						>> IJ[0][4] >> IJ[1][4] >> IJ[2][4] >> IJ[3][4] >> IJ[4][4] >> IJ[5][4] >> IJ[6][4] >> IJ[7][4] >> IJ[8][4]) {
		for (int c = 0; c < 5; c++) {
			for (int r = 0; r < 9; r++) {
				inv_jcb_data[gpu][idx++] = (IJ[r][c] >= 0.0000001) ? IJ[r][c] : 0.0;
			}
		}
		if (idx >= 9*5*data_per_gpu[gpu]) {
			gpu++;
			idx = 0;
		}
	}

	inv_jcb_file.close();
}

void Write_File_Exp(double** us, double** abs_errors, double** rlt_errors,
					double** abs_uqs, double** rlt_uqs, double** abs_uks, double** rlt_uks,
					double** abs_sds, double** rlt_sds, double* kernel_time, int num) {
	ofstream us_file;
	us_file.open((directory + "exp_us.data").c_str());
	ofstream abs_errors_file;
	abs_errors_file.open((directory + "abs_errors.data").c_str());
	ofstream rlt_errors_file;
	rlt_errors_file.open((directory + "rlt_errors.data").c_str());
	ofstream abs_uqs_file;
	abs_uqs_file.open((directory + "abs_uqs.data").c_str());
	ofstream rlt_uqs_file;
	rlt_uqs_file.open((directory + "rlt_uqs.data").c_str());
	ofstream abs_uks_file;
	abs_uks_file.open((directory + "abs_uks.data").c_str());
	ofstream rlt_uks_file;
	rlt_uks_file.open((directory + "rlt_uks.data").c_str());
	ofstream abs_sds_file;
	abs_sds_file.open((directory + "abs_sds.data").c_str());
	ofstream rlt_sds_file;
	rlt_sds_file.open((directory + "rlt_sds.data").c_str());
	ofstream time_file;
	time_file.open((directory + "time.data").c_str());

	for (int i = 0; i < num; i++) {
		time_file << kernel_time[i] << endl;

		for (int j = 0; j < 9; j++) {
			us_file << us[i][j] << " ";
		}
		us_file << "" << endl;

		for (int k = 0; k < 10; k++) {
			abs_errors_file << abs_errors[i][k] << " ";
			rlt_errors_file << rlt_errors[i][k] << " ";

			for (int j = 0; j < 9; j++) {
				abs_uqs_file << abs_uqs[i][k*9 + j] << " ";
				abs_uks_file << abs_uks[i][k*9 + j] << " ";
				rlt_uqs_file << rlt_uqs[i][k*9 + j] << " ";
				rlt_uks_file << rlt_uks[i][k*9 + j] << " ";
			}
			for (int j = 0; j < 5; j++) {
				abs_sds_file << abs_sds[i][k*5 + j] << " ";
				rlt_sds_file << rlt_sds[i][k*5 + j] << " ";
			}
		}

		abs_errors_file << "" << endl;
		rlt_errors_file << "" << endl;
		abs_uqs_file << "" << endl;
		rlt_uqs_file << "" << endl;
		abs_uks_file << "" << endl;
		rlt_uks_file << "" << endl;
		abs_sds_file << "" << endl;
		rlt_sds_file << "" << endl;

		abs_errors_file << "" << endl;
		rlt_errors_file << "" << endl;
		abs_uqs_file << "" << endl;
		rlt_uqs_file << "" << endl;
		abs_uks_file << "" << endl;
		rlt_uks_file << "" << endl;
		abs_sds_file << "" << endl;
		rlt_sds_file << "" << endl;
	}

	us_file.close();
	abs_errors_file.close();
	abs_uqs_file.close();
	abs_uks_file.close();
	abs_sds_file.close();
	rlt_errors_file.close();
	rlt_uqs_file.close();
	rlt_uks_file.close();
	rlt_sds_file.close();
	time_file.close();
}


void Find_Max() {
	ifstream us_file;
	us_file.open((directory + "exp_us.data").c_str());
	ifstream abs_errors_file;
	abs_errors_file.open((directory + "abs_errors.data").c_str());
	ifstream rlt_errors_file;
	rlt_errors_file.open((directory + "rlt_errors.data").c_str());
	ifstream abs_uqs_file;
	abs_uqs_file.open((directory + "abs_uqs.data").c_str());
	ifstream rlt_uqs_file;
	rlt_uqs_file.open((directory + "rlt_uqs.data").c_str());
	ifstream abs_uks_file;
	abs_uks_file.open((directory + "abs_uks.data").c_str());
	ifstream rlt_uks_file;
	rlt_uks_file.open((directory + "rlt_uks.data").c_str());
	ifstream abs_sds_file;
	abs_sds_file.open((directory + "abs_sds.data").c_str());
	ifstream rlt_sds_file;
	rlt_sds_file.open((directory + "rlt_sds.data").c_str());

	ofstream max_error_file;
	max_error_file.open((directory + "max_error.data").c_str());

	// Dealing with abs one
	int line_num = 0;
	int curr_line = 0;
	double max_error = -1.0;
	double curr_error = 0.0;
	string line;
	while(getline(abs_errors_file, line)) {
		istringstream iss(line);
		iss >> curr_error;
		if (curr_error > max_error) {
			max_error = curr_error;
			line_num = curr_line;
		}
		curr_line++;
	}

	double u[9], uq[9], uk[9], sd[5];
	read_specific_line(line_num, &u[0], us_file, 9);
	read_specific_line(line_num, &uq[0], abs_uqs_file, 9);
	read_specific_line(line_num, &uk[0], abs_uks_file, 9);
	read_specific_line(line_num, &sd[0], abs_sds_file, 5);

	max_error_file << "Max abs error: " << max_error << endl;

	max_error_file << "Corresponding u: ";
	for (int i = 0; i < 9; i++) {
		max_error_file << u[i] << " ";
	}
	max_error_file << "\n";

	max_error_file << "Corresponding uq: ";
	for (int i = 0; i < 9; i++) {
		max_error_file << uq[i] << " ";
	}
	max_error_file << "\n";

	max_error_file << "Corresponding uk: ";
	for (int i = 0; i < 9; i++) {
		max_error_file << uk[i] << " ";
	}
	max_error_file << "\n";

	max_error_file << "Corresponding sd: ";
	for (int i = 0; i < 5; i++) {
		max_error_file << sd[i] << " ";
	}
	max_error_file << "\n";
	max_error_file << "\n";
	max_error_file << "\n";

	// Dealing with rlt one
	line_num = 0;
	curr_line = 0;
	max_error = -1.0;
	curr_error = 0.0;

	while(getline(rlt_errors_file, line)) {
		istringstream iss(line);
		iss >> curr_error;
		if (curr_error > max_error) {
			max_error = curr_error;
			line_num = curr_line;
		}
		curr_line++;
	}

	read_specific_line(line_num, &u[0], us_file, 9);
	read_specific_line(line_num, &uq[0], rlt_uqs_file, 9);
	read_specific_line(line_num, &uk[0], rlt_uks_file, 9);
	read_specific_line(line_num, &sd[0], rlt_sds_file, 5);

	max_error_file << "Max rlt error: " << max_error << endl;

	max_error_file << "Corresponding u: ";
	for (int i = 0; i < 9; i++) {
		max_error_file << u[i] << " ";
	}
	max_error_file << "\n";

	max_error_file << "Corresponding uq: ";
	for (int i = 0; i < 9; i++) {
		max_error_file << uq[i] << " ";
	}
	max_error_file << "\n";

	max_error_file << "Corresponding uk: ";
	for (int i = 0; i < 9; i++) {
		max_error_file << uk[i] << " ";
	}
	max_error_file << "\n";

	max_error_file << "Corresponding sd: ";
	for (int i = 0; i < 5; i++) {
		max_error_file << sd[i] << " ";
	}
	max_error_file << "\n";

	us_file.close();
	abs_errors_file.close();
	abs_uqs_file.close();
	abs_uks_file.close();
	abs_sds_file.close();
	rlt_errors_file.close();
	rlt_uqs_file.close();
	rlt_uks_file.close();
	rlt_sds_file.close();
	max_error_file.close();
}

void read_specific_line(int line_num, double* result, ifstream& file, int dim) {
	string line;
	int curr_line = 0;
	while (getline(file, line)) {
		if (curr_line != line_num) {
			curr_line++;
			continue;
		}
		istringstream iss(line);
		for (int i = 0; i < dim; i++) {
			iss >> result[i];
		}
		break;
	}
}
