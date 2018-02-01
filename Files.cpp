/*
 * Files.cpp
 *
 *  Created on: Jan 25, 2018
 *      Author: liang
 */

#include "Files.h"

using namespace std;
using namespace Eigen;


void Write_File(double* uk_data, double* sk_data, double* inv_jcb_data, unsigned long int data_num) {
	ofstream uk_file;
	uk_file.open("./uk.data");
	ofstream sk_file;
	sk_file.open("./sk.data");
	ofstream jcb_file;
	jcb_file.open("./inv_jcb.data");

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
	uk_file.open("./uk.data");

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
	sk_file.open("./sk.data");
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
	inv_jcb_file.open("./inv_jcb.data");
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
