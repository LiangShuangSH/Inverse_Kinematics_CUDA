/*
 * PseudoInverse.cpp
 *
 *  Created on: Dec 12, 2017
 *      Author: liang
 */


#include "PseudoInverse.h"

using namespace Eigen;

MatrixXd pinv(MatrixXd &mat) {
	return mat.completeOrthogonalDecomposition().pseudoInverse();
}

/*
MatrixXd pinv(MatrixXd &mat, double tolerance) {
	JacobiSVD<MatrixXd> svd(mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
	const VectorXd &singularValues = svd.singularValues();
	MatrixXd singularValuesInv(mat.cols(), mat.rows());
	singularValuesInv.setZero();

	for(unsigned int i = 0; i < singularValues.size(); i++) {
		if(singularValues(i) > tolerance) {
			singularValuesInv(i, i) = 1.0 / singularValues(i);
		}
		else{
			singularValuesInv(i, i) = 0.0;
		}
	}

	return svd.matrixV() * singularValuesInv * svd.matrixU().adjoint();
}
*/
