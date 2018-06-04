/*
 * Files.h
 *
 *  Created on: Jan 25, 2018
 *      Author: liang
 */

#ifndef FILES_H_
#define FILES_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>

void Write_File(double* uk_data, double* sk_data, double* inv_jcb_data, unsigned long int data_num);
void Read_File(double* uk_data, double* sk_data, double* inv_jcb_data, bool test);
void Read_Inv_Jcb(double* inv_jcb_data);
void Read_Inv_Jcb_MG(double** inv_jcb_data, unsigned long int* data_per_gpu);
void Read_File_MG(double** uk_data, double** sk_data, double** inv_jcb_data, unsigned long int* data_per_gpu);
void Write_File_Exp(double** us, double** abs_errors, double** rlt_errors,
					double** abs_uqs, double** rlt_uqs, double** abs_uks, double** rlt_uks,
					double** abs_sds, double** rlt_sds, double* kernel_time, int num);
void Find_Max();
void read_specific_line(int line_num, double* result, std::ifstream& file, int dim);
#endif /* FILES_H_ */
