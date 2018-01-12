/*
 * PseudoInverse.h
 *
 *  Created on: Dec 12, 2017
 *      Author: liang
 */

#ifndef PSEUDOINVERSE_H_
#define PSEUDOINVERSE_H_

#include <Eigen/Dense>
#include <Eigen/QR>

Eigen::MatrixXd pinv(Eigen::MatrixXd &mat);


//MatrixXd pinv(MatrixXd &mat, double tolerance);

#endif /* PSEUDOINVERSE_H_ */
