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
#include <Eigen/Dense>

void Write_File(double* uk_data, double* sk_data, double* inv_jcb_data, unsigned long int data_num);
void Read_File(double* uk_data, double* sk_data, double* inv_jcb_data, bool test);
void Read_Inv_Jcb(double* inv_jcb_data);

#endif /* FILES_H_ */
