/*
 * Unicycle.cu
 *
 *  Created on: Nov 21, 2017
 *      Author: liang
 */

#include "Unicycle.cuh"
#include "utility.cuh"
#include "fresnel.cuh"

using namespace std;
using namespace Eigen;

__host__ __device__ void forward_kinematics(double* state, double a, double b, double t)
{

	double dx, dy, dz, dv, dw;

	double s = state[3];
	double c = state[4];
	double d = state[2];

	// The easy ones.
	dz = 0.5*b*t*t + c*t;
	dv = a*t;
	dw = b*t;

	// dx and dy

	// b = 0 and c = 0 case -> just linear acceleration
	if (fabs(b) < 0.001 && fabs(c) < 0.001)
	{
		double l = 0.5*a*t*t + s*t;
		dx = cos(d) * l;
		dy = sin(d) * l;
	}

	// b = 0 case -> formula singularity covered
	else if (fabs(b) < 0.001)
	{
		dx = a*(cos(c*t + d) - cos(d)) / (c*c) + ((a*t + s)*sin(c*t + d) - s*sin(d)) / c;
		dy = a*(sin(c*t + d) - sin(d)) / (c*c) - ((a*t + s)*cos(c*t + d) - s*cos(d)) / c;
	}

	else
	{
		// Use a mirroring technique if b is negative to avoid the negative sqrt().
		bool flipped = false;
		if (b < 0)
		{
			b = -b;
			c = -c;
			flipped = true;
		}

		double sb = sqrt(b);
		double pb15 = pow(b, 1.5);
		double gamma = cos(0.5*c*c / b - d);
		double sigma = sin(0.5*c*c / b - d);
		double c1, s1, c0, s0;
		frenel((c + b*t) / (sb*SPI), &s1, &c1);
		frenel(c / (sb*SPI), &s0, &c0);
		double C = c1 - c0;
		double S = s1 - s0;

		dx = SPI * (b*s - a*c) * (sigma*S + gamma*C) / pb15 + (a / b)*(sin(0.5*b*t*t + c*t + d) - sin(d));
		dy = SPI * (b*s - a*c) * (gamma*S - sigma*C) / pb15 - (a / b)*(cos(0.5*b*t*t + c*t + d) - cos(d));

		if (flipped)
		{
			double c2d = cos(2 * d);
			double s2d = sin(2 * d);
			double dxt = c2d*dx + s2d*dy;
			dy = s2d*dx - c2d*dy;
			dx = dxt;
		}
	}

	// Apply the state change and transform into world coordinates.
	state[0] += dx;
	state[1] += dy;
	state[2] += dz;
	state[3] += dv;
	state[4] += dw;

	//w = normalize_angle(w);

	return;
}

__host__ __device__ void forward_kinematics_3(double* state, double a1, double b1, double t1,
									 	 	 	 	double a2, double b2, double t2,
									 	 	 	 	double a3, double b3, double t3) {

	forward_kinematics(state, a1, b1, t1);
	forward_kinematics(state, a2, b2, t2);
	forward_kinematics(state, a3, b3, t3);
	return;
}

__host__ __device__ void forward_kinematics_3(double* state, double* u) {
	forward_kinematics_3(state, u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8]);
	return;
}

__host__ __device__ MatrixXd jacobian(double* state, double a, double b, double t)
{
	double dxda, dxdb, dxdt;
	double dyda, dydb, dydt;
	double dzda, dzdb, dzdt;
	double dvda, dvdb, dvdt;
	double dwda, dwdb, dwdt;

	double s = state[3];
	double c = state[4];

	// The easy ones.
	dxdt = (a*t + s) * cos(0.5*b*t*t + c*t);
	dydt = (a*t + s) * sin(0.5*b*t*t + c*t);
	dzda = 0.0;
	dzdb = 0.5*t*t;
	dzdt = b*t + c;
	dvda = t;
	dvdb = 0.0;
	dvdt = a;
	dwda = 0;
	dwdb = t;
	dwdt = b;

	// Difficult ones with zero conditions and b sign flip.

	// b = 0 and c = 0 case -> just linear acceleration
	if (fabs(b) < 0.00001 && fabs(c) < 0.00001)
	{
		dxda = 0.5*t*t;
		dyda = 0.0;

		// There is a singularity at b = 0.
		// Approximate the b gradient with an orthogonal unit vector.
		// This could be replaced by a more accurate "bridge".
		dxdb = 0.0;
		dydb = 1.0;
	}

	// b = 0 case
	else if (fabs(b) < 0.00001)
	{
		dxda = (c*t * sin(c*t) + cos(c*t) - 1) / (c*c);
		dyda = (sin(c*t) - c*t * cos(c*t)) / (c*c);

		// There is a singularity at b = 0.
		// Approximate the b gradient with an orthogonal unit vector.
		// This could be replaced by a more accurate "bridge".
		dxdb = -sin(c*t);
		dydb = cos(c*t);
	}

	else
	{
		double sgn = 1.0;
		if (b < 0)
		{
			b = -b;
			c = -c;
			sgn = -1.0;
		}

		double sb = sqrt(b);
		double pb15 = pow(b, 1.5);
		double pb25 = pow(b, 2.5);
		double pb35 = pow(b, 3.5);
		double gamma = cos(0.5*c*c / b);
		double sigma = sin(0.5*c*c / b);
		double kappa = cos(0.5*t*t*b + c*t);
		double zeta = sin(0.5*t*t*b + c*t);
		double c1, s1, c0, s0;
		frenel((c + b*t) / (sb*SPI), &s1, &c1);
		frenel(c / (sb*SPI), &s0, &c0);
		double C = c1 - c0;
		double S = s1 - s0;
		double dSdb = sin((b*t + c)*(b*t + c) / (2.0*b)) * (t / (SPI*sb) - (c + t*b) / (2.0*SPI*pb15)) + c*sigma / (2.0*SPI*pb15);
		double dCdb = cos((b*t + c)*(b*t + c) / (2.0*b)) * (t / (SPI*sb) - (c + t*b) / (2.0*SPI*pb15)) + c*gamma / (2.0*SPI*pb15);

		dxda = SPI*c / pb15 * (gamma*C - sigma*S) + zeta / b;
		dxdb = sgn * (SPI * (s / sb - a*c / pb15) * (sigma*dSdb + gamma*dCdb)
			+ SPI * (3.0*a*c / (2.0*pb25) - s / (2.0*pb15)) * (sigma*S + gamma*C)
			+ SPI * (a*c*c*c / (2.0*pb35) - s*c*c / (2.0*pb25)) * (gamma*S - sigma*C)
			+ a*t*t / (2 * b) * kappa - a / (b*b)*zeta);
		dyda = sgn * (SPI*c / pb15 * (sigma*C - gamma*S) - (kappa - 1.0) / b);
		dydb = SPI * (s / sb - a*c / pb15) * (gamma*dSdb - sigma*dCdb)
			+ SPI * (3.0*a*c / (2.0*pb25) - s / (2.0*pb15)) * (gamma*S - sigma*C)
			+ SPI * (-a*c*c*c / (2.0*pb35) + s*c*c / (2.0*pb25)) * (sigma*S + gamma*C)
			+ a*t*t / (2.0*b) * zeta + a / (b*b)*(kappa - 1.0);
	}


	// Assign values to Jacobian and rotate the x and y components by z to denormalize.
	MatrixXd J(5, 3);
	double cz = cos(state[2]);
	double sz = sin(state[2]);
	J(0, 0) = cz*dxda - sz*dyda;	J(0, 1) = cz*dxdb - sz*dydb;	J(0, 2) = cz*dxdt - sz*dydt;
	J(1, 0) = sz*dxda + cz*dyda;	J(1, 1) = sz*dxdb + cz*dydb;	J(1, 2) = sz*dxdt + cz*dydt;
	J(2, 0) = dzda;				J(2, 1) = dzdb;				J(2, 2) = dzdt;
	J(3, 0) = dvda;				J(3, 1) = dvdb;				J(3, 2) = dvdt;
	J(4, 0) = dwda;				J(4, 1) = dwdb;				J(4, 2) = dwdt;

	return J;
}

// The second order Jacobian describes the effects of the first parameters on the second bang.
__host__ __device__ MatrixXd jacobian2(double* state, double a1, double b1, double t1, double a, double b, double t)
{
	double dxda1, dxdb1, dxdt1;
	double dyda1, dydb1, dydt1;
	double dzda1, dzdb1, dzdt1;
	double dvda1, dvdb1, dvdt1;
	double dwda1, dwdb1, dwdt1;

	double s = state[3];
	double c = state[4];
	double d = state[2];

	// b = 0 case. Singularity!
	if (fabs(b) < 0.00001)
	{
		double gamma = cos(c*t + d);
		double sigma = sin(c*t + d);
		double gamma0 = cos(d);
		double sigma0 = sin(d);

		if (fabs(c) < 0.00001)
		{
			dxda1 = cos(d)*t*t1;
			dyda1 = sin(d)*t*t1;

			// Approximate the b1 gradient with an orthogonal unit vector.
			// Replace this with a bridge.
			dxdb1 = -sin(c*t);
			dydb1 = cos(c*t);

			dxdt1 = cos(d)*a1*t;
			dydt1 = sin(d)*a1*t;
		}
		else
		{
			dxda1 = t1 / c * (sigma - sigma0);
			dyda1 = t1 / c * (gamma0 - gamma);

			// These are wrong.
			dxdb1 = t1*(t*(a*t + s)*gamma / c + sigma*(s - 2 * (a*t + s)) / (c*c) + 2 * a*(1 - gamma) / (c*c*c));
			dydb1 = t1*(t*(a*t + s)*sigma / c + (gamma - 1)*(2 * (a*t + s) - s) / (c*c) - 2 * a*sigma / (c*c*c));

			// These are wrong.
			dxdt1 = b1*(t*(a*t + s)*gamma / c + sigma*(s - 2 * (a*t + s)) / (c*c) + 2 * a*(1 - gamma) / (c*c*c)) + a1*sigma / c;
			dydt1 = b1*(t*(a*t + s)*sigma / c + (gamma - 1)*(2 * (a*t + s) - s) / (c*c) - 2 * a*sigma / (c*c*c)) + a1*(1 - gamma) / c;
		}

		dzda1 = 0.0;
		dvda1 = 0.0;
		dwda1 = 0.0;

		dzdb1 = t*t1;
		dvdb1 = 0.0;
		dwdb1 = 0.0;

		dzdt1 = t*b1;
		dvdt1 = 0.0;
		dwdt1 = 0.0;
	}

	else
	{
		// Use a mirroring technique if b is negative to avoid the negative sqrt().
		bool flipped = false;
		if (b < 0)
		{
			b = -b;
			b1 = -b1;
			c = -c;
			flipped = true;
		}

		double sb = sqrt(b);
		double pb15 = pow(b, 1.5);
		double sigma = sin(0.5*c*c / b - d);
		double gamma = cos(0.5*c*c / b - d);
		double zeta = sin(0.5*t*t*b + c*t + d);
		double kappa = cos(0.5*t*t*b + c*t + d);
		double s_ = sin((b*t + c)*(b*t + c) / (2.0*b));
		double c_ = cos((b*t + c)*(b*t + c) / (2.0*b));
		double s0_ = sin(c*c / (2.0*b));
		double c0_ = cos(c*c / (2.0*b));
		double sind = sin(d);
		double cosd = cos(d);
		double c1, s1, c0, s0;
		frenel((c + b*t) / (sb*SPI), &s1, &c1);
		frenel(c / (sb*SPI), &s0, &c0);
		double C = c1 - c0;
		double S = s1 - s0;
		double dSdb1 = (s_ - s0_) * t1 / (SPI*sb);
		double dCdb1 = (c_ - c0_) * t1 / (SPI*sb);
		double dSdt1 = (s_ - s0_) * b1 / (SPI*sb);
		double dCdt1 = (c_ - c0_) * b1 / (SPI*sb);

		dxda1 = t1 * SPI / sb * (sigma*S + gamma*C);
		dyda1 = t1 * SPI / sb * (gamma*S - sigma*C);
		dzda1 = 0.0;
		dvda1 = 0.0;
		dwda1 = 0.0;

		dxdb1 = SPI / pb15 * ((b*s - a*c) * (sigma*dSdb1 + gamma*dCdb1)
			+ (b*s - a*c) * (t1*c / b - 0.5*t1*t1) * (gamma*S - sigma*C)
			- t1 * a * (sigma*S + gamma*C))
			+ a / b * ((0.5*t1*t1 + t*t1)*kappa - 0.5*t1*t1*cosd);
		dydb1 = SPI / pb15 * ((b*s - a*c) * (gamma*dSdb1 - sigma*dCdb1)
			- (b*s - a*c) * (t1*c / b - 0.5*t1*t1) * (sigma*S + gamma*C)
			+ t1 * a * (sigma*C - gamma*S))
			+ a / b * ((0.5*t1*t1 + t*t1)*zeta - 0.5*t1*t1*sind);
		dzdb1 = t*t1;
		dvdb1 = 0.0;
		dwdb1 = 0.0;

		dxdt1 = SPI / pb15 * ((b*s - a*c) * (sigma*dSdt1 + gamma*dCdt1)
			+ (a1*b - a*b1) * (sigma*S + gamma*C)
			+ (b*s - a*c) * (b1*c / b - c) * (gamma*S - sigma*C))
			+ a / b * ((b1*t + c)*kappa - c*cosd);
		dydt1 = SPI / pb15 * ((b*s - a*c) * (gamma*dSdt1 - sigma*dCdt1)
			+ (a1*b - a*b1) * (gamma*S - sigma*C)
			- (b*s - a*c) * (b1*c / b - c) * (sigma*S + gamma*C))
			+ a / b * ((b1*t + c)*zeta - c*sind);
		dzdt1 = t*b1;
		dvdt1 = 0.0;
		dwdt1 = 0.0;

		if (flipped)
		{
			double dxt;
			double cos2d = cos(2 * d);
			double sin2d = sin(2 * d);

			dxt = cos2d*dxda1 + sin2d*dyda1;
			dyda1 = sin2d*dxda1 - cos2d*dyda1;
			dxda1 = dxt;

			dxt = cos2d*dxdb1 + sin2d*dydb1;
			dydb1 = -(sin2d*dxdb1 - cos2d*dydb1);
			dxdb1 = -dxt;

			dxt = cos2d*dxdt1 + sin2d*dydt1;
			dydt1 = (sin2d*dxdt1 - cos2d*dydt1);
			dxdt1 = dxt;
		}
	}


	// Assign values to the second order Jacobian.
	MatrixXd J(5, 3);

	J(0, 0) = dxda1; J(0, 1) = dxdb1; J(0, 2) = dxdt1;
	J(1, 0) = dyda1; J(1, 1) = dydb1; J(1, 2) = dydt1;
	J(2, 0) = dzda1; J(2, 1) = dzdb1; J(2, 2) = dzdt1;
	J(3, 0) = dvda1; J(3, 1) = dvdb1; J(3, 2) = dvdt1;
	J(4, 0) = dwda1; J(4, 1) = dwdb1; J(4, 2) = dwdt1;

	return J;
}

// The third order Jacobian describes the effects of the first parameters on the third link.
__host__ __device__ MatrixXd jacobian3(double* state, double a1, double b1, double t1, double a2, double b2, double t2, double a3, double b3, double t3)
{
	double dxda1, dxdb1, dxdt1;
	double dyda1, dydb1, dydt1;
	double dzda1, dzdb1, dzdt1;
	double dvda1, dvdb1, dvdt1;
	double dwda1, dwdb1, dwdt1;

	double s = state[3];
	double c = state[4];
	double d = state[2];
	double a = a3;
	double b = b3;
	double t = t3;

	// b = 0 case. Singularity!
	if (fabs(b) < 0.00001)
	{
		double gamma = cos(c*t + d);
		double sigma = sin(c*t + d);
		double gamma0 = cos(d);
		double sigma0 = sin(d);

		if (fabs(c) < 0.00001)
		{
			dxda1 = cos(d)*t*t1;
			dyda1 = sin(d)*t*t1;

			// Approximate the b1 gradient with an orthogonal unit vector.
			// Replace this with a bridge.
			dxdb1 = -sin(c*t);
			dydb1 = cos(c*t);

			dxdt1 = cos(d)*a1*t;
			dydt1 = sin(d)*a1*t;
		}
		else
		{
			dxda1 = t1 / c * (sigma - sigma0);
			dyda1 = t1 / c * (gamma0 - gamma);

			// These are wrong.
			dxdb1 = t1*(t*(a*t + s)*gamma / c + sigma*(s - 2 * (a*t + s)) / (c*c) + 2 * a*(1 - gamma) / (c*c*c));
			dydb1 = t1*(t*(a*t + s)*sigma / c + (gamma - 1)*(2 * (a*t + s) - s) / (c*c) - 2 * a*sigma / (c*c*c));

			// These are wrong.
			dxdt1 = b1*(t*(a*t + s)*gamma / c + sigma*(s - 2 * (a*t + s)) / (c*c) + 2 * a*(1 - gamma) / (c*c*c)) + a1*sigma / c;
			dydt1 = b1*(t*(a*t + s)*sigma / c + (gamma - 1)*(2 * (a*t + s) - s) / (c*c) - 2 * a*sigma / (c*c*c)) + a1*(1 - gamma) / c;
		}

		dzda1 = 0.0;
		dvda1 = 0.0;
		dwda1 = 0.0;

		dzdb1 = t*t1;
		dvdb1 = 0.0;
		dwdb1 = 0.0;

		dzdt1 = t*b1;
		dvdt1 = 0.0;
		dwdt1 = 0.0;
	}

	else
	{
		// Use a mirroring technique if b is negative to avoid the negative sqrt().
		bool flipped = false;
		if (b < 0)
		{
			b = -b;
			b1 = -b1;
			b2 = -b2;
			c = -c;
			flipped = true;
		}

		double sb = sqrt(b);
		double pb15 = pow(b, 1.5);
		double sigma = sin(0.5*c*c / b - d);
		double gamma = cos(0.5*c*c / b - d);
		double zeta = sin(0.5*t*t*b + c*t + d);
		double kappa = cos(0.5*t*t*b + c*t + d);
		double sind = sin(d);
		double cosd = cos(d);
		double c1, s1, c0, s0;
		frenel((c + b*t) / (sb*SPI), &s1, &c1);
		frenel(c / (sb*SPI), &s0, &c0);
		double C = c1 - c0;
		double S = s1 - s0;
		double s_ = sin((b*t + c)*(b*t + c) / (2.0*b));
		double c_ = cos((b*t + c)*(b*t + c) / (2.0*b));
		double s0_ = sin(c*c / (2.0*b));
		double c0_ = cos(c*c / (2.0*b));
		double dSdb1 = (s_ - s0_) * t1 / (SPI*sb);
		double dCdb1 = (c_ - c0_) * t1 / (SPI*sb);
		double dSdt1 = (s_ - s0_) * b1 / (SPI*sb);
		double dCdt1 = (c_ - c0_) * b1 / (SPI*sb);

		dxda1 = t1 * SPI / sb * (sigma*S + gamma*C);
		dyda1 = t1 * SPI / sb * (gamma*S - sigma*C);
		dzda1 = 0.0;
		dvda1 = 0.0;
		dwda1 = 0.0;

		dxdb1 = SPI / pb15 * ((b*s - a*c) * (sigma*dSdb1 + gamma*dCdb1)
			+ (b*s - a*c) * (t1*c / b - t1*t2 - 0.5*t1*t1) * (gamma*S - sigma*C)
			- t1 * a * (sigma*S + gamma*C))
			+ a / b * ((0.5*t1*t1 + t1*t2 + t*t1)*kappa - (0.5*t1*t1 + t1*t2)*cosd);
		dydb1 = SPI / pb15 * ((b*s - a*c) * (gamma*dSdb1 - sigma*dCdb1)
			- (b*s - a*c) * (t1*c / b - t1*t2 - 0.5*t1*t1) * (sigma*S + gamma*C)
			+ t1 * a * (sigma*C - gamma*S))
			+ a / b * ((0.5*t1*t1 + t1*t2 + t*t1)*zeta - (0.5*t1*t1 + t1*t2)*sind);
		dzdb1 = t*t1;
		dvdb1 = 0.0;
		dwdb1 = 0.0;

		dxdt1 = SPI / pb15 * ((b*s - a*c) * (sigma*dSdt1 + gamma*dCdt1)
			+ (a1*b - a*b1) * (sigma*S + gamma*C)
			+ (b*s - a*c) * (b1*c / b - b1*t2 - (c - b2*t2)) * (gamma*S - sigma*C))
			+ a / b * ((b1*t2 + b1*t + (c - b2*t2))*kappa - (b1*t2 + (c - b2*t2))*cosd);
		dydt1 = SPI / pb15 * ((b*s - a*c) * (gamma*dSdt1 - sigma*dCdt1)
			+ (a1*b - a*b1) * (gamma*S - sigma*C)
			- (b*s - a*c) * (b1*c / b - b1*t2 - (c - b2*t2)) * (sigma*S + gamma*C))
			+ a / b * ((b1*t2 + b1*t + (c - b2*t2))*zeta - (b1*t2 + (c - b2*t2))*sind);
		dzdt1 = t*b1;
		dvdt1 = 0.0;
		dwdt1 = 0.0;

		if (flipped)
		{
			double dxt;
			double cos2d = cos(2 * d);
			double sin2d = sin(2 * d);

			dxt = cos2d*dxda1 + sin2d*dyda1;
			dyda1 = sin2d*dxda1 - cos2d*dyda1;
			dxda1 = dxt;

			dxt = cos2d*dxdb1 + sin2d*dydb1;
			dydb1 = -(sin2d*dxdb1 - cos2d*dydb1);
			dxdb1 = -dxt;

			dxt = cos2d*dxdt1 + sin2d*dydt1;
			dydt1 = (sin2d*dxdt1 - cos2d*dydt1);
			dxdt1 = dxt;
		}
	}

	// Assign values to the third order Jacobian.
	MatrixXd J(5, 3);

	J(0, 0) = dxda1; J(0, 1) = dxdb1; J(0, 2) = dxdt1;
	J(1, 0) = dyda1; J(1, 1) = dydb1; J(1, 2) = dydt1;
	J(2, 0) = dzda1; J(2, 1) = dzdb1; J(2, 2) = dzdt1;
	J(3, 0) = dvda1; J(3, 1) = dvdb1; J(3, 2) = dvdt1;
	J(4, 0) = dwda1; J(4, 1) = dwdb1; J(4, 2) = dwdt1;

	return J;
}

// The complete Jacobian matrix of a three link trajectory.
__host__ __device__ MatrixXd jacobian(double* state, double a1, double b1, double t1, double a2, double b2, double t2, double a3, double b3, double t3)
{
	MatrixXd J(5, 3);
	double dxda1, dxdb1, dxdt1, dxda2, dxdb2, dxdt2, dxda3, dxdb3, dxdt3;
	double dyda1, dydb1, dydt1, dyda2, dydb2, dydt2, dyda3, dydb3, dydt3;
	double dzda1, dzdb1, dzdt1, dzda2, dzdb2, dzdt2, dzda3, dzdb3, dzdt3;
	double dvda1, dvdb1, dvdt1, dvda2, dvdb2, dvdt2, dvda3, dvdb3, dvdt3;
	double dwda1, dwdb1, dwdt1, dwda2, dwdb2, dwdt2, dwda3, dwdb3, dwdt3;

	double* car = new double[5];
	for (int i = 0; i < 5; i++) {
		car[i] = state[i];
	}

	// The first half of the Jacobian is the one link Jacobian of the first state
	// plus the "second order" Jacobian of the second state
	// plus the "third order" Jacobian of the third state.
	J = jacobian(car, a1, b1, t1);

	car[2] += car[4]*t1 + 0.5*b1*t1*t1;
	car[3] += a1*t1;
	car[4] += b1*t1;
	J += jacobian2(car, a1, b1, t1, a2, b2, t2);

	car[2] += car[4]*t2 + 0.5*b2*t2*t2;
	car[3] += a2*t2;
	car[4] += b2*t2;
	J += jacobian3(car, a1, b1, t1, a2, b2, t2, a3, b3, t3);

	dxda1 = J(0, 0);	dxdb1 = J(0, 1);	dxdt1 = J(0, 2);
	dyda1 = J(1, 0);	dydb1 = J(1, 1);	dydt1 = J(1, 2);
	dzda1 = J(2, 0);	dzdb1 = J(2, 1);	dzdt1 = J(2, 2);
	dvda1 = J(3, 0);	dvdb1 = J(3, 1);	dvdt1 = J(3, 2);
	dwda1 = J(4, 0);	dwdb1 = J(4, 1);	dwdt1 = J(4, 2);


	// The second half of the Jacobian is the one link Jacobian of the second state
	// plus the second order Jacobian from the third state.
	for (int i = 0; i < 5; i++) {
		car[i] = state[i];
	}
	car[2] += car[4]*t1 + 0.5*b1*t1*t1;
	car[3] += a1*t1;
	car[4] += b1*t1;
	J = jacobian(car, a2, b2, t2);

	car[2] += car[4]*t2 + 0.5*b2*t2*t2;
	car[3] += a2*t2;
	car[4] += b2*t2;
	J += jacobian2(car, a2, b2, t2, a3, b3, t3);

	dxda2 = J(0, 0);	dxdb2 = J(0, 1);	dxdt2 = J(0, 2);
	dyda2 = J(1, 0);	dydb2 = J(1, 1);	dydt2 = J(1, 2);
	dzda2 = J(2, 0);	dzdb2 = J(2, 1);	dzdt2 = J(2, 2);
	dvda2 = J(3, 0);	dvdb2 = J(3, 1);	dvdt2 = J(3, 2);
	dwda2 = J(4, 0);	dwdb2 = J(4, 1);	dwdt2 = J(4, 2);


	// The third half of the three link Jacobian is the one link Jacobian of the third state.
	J = jacobian(car, a3, b3, t3);
	dxda3 = J(0, 0);	dxdb3 = J(0, 1);	dxdt3 = J(0, 2);
	dyda3 = J(1, 0);	dydb3 = J(1, 1);	dydt3 = J(1, 2);
	dzda3 = J(2, 0);	dzdb3 = J(2, 1);	dzdt3 = J(2, 2);
	dvda3 = J(3, 0);	dvdb3 = J(3, 1);	dvdt3 = J(3, 2);
	dwda3 = J(4, 0);	dwdb3 = J(4, 1);	dwdt3 = J(4, 2);


	// Assign values to the three link Jacobian.
	MatrixXd J3(5, 9);
	J3(0, 0) = dxda1; J3(0, 1) = dxdb1; J3(0, 2) = dxdt1; J3(0, 3) = dxda2; J3(0, 4) = dxdb2; J3(0, 5) = dxdt2; J3(0, 6) = dxda3; J3(0, 7) = dxdb3; J3(0, 8) = dxdt3;
	J3(1, 0) = dyda1; J3(1, 1) = dydb1; J3(1, 2) = dydt1; J3(1, 3) = dyda2; J3(1, 4) = dydb2; J3(1, 5) = dydt2; J3(1, 6) = dyda3; J3(1, 7) = dydb3; J3(1, 8) = dydt3;
	J3(2, 0) = dzda1; J3(2, 1) = dzdb1; J3(2, 2) = dzdt1; J3(2, 3) = dzda2; J3(2, 4) = dzdb2; J3(2, 5) = dzdt2; J3(2, 6) = dzda3; J3(2, 7) = dzdb3; J3(2, 8) = dzdt3;
	J3(3, 0) = dvda1; J3(3, 1) = dvdb1; J3(3, 2) = dvdt1; J3(3, 3) = dvda2; J3(3, 4) = dvdb2; J3(3, 5) = dvdt2; J3(3, 6) = dvda3; J3(3, 7) = dvdb3; J3(3, 8) = dvdt3;
	J3(4, 0) = dwda1; J3(4, 1) = dwdb1; J3(4, 2) = dwdt1; J3(4, 3) = dwda2; J3(4, 4) = dwdb2; J3(4, 5) = dwdt2; J3(4, 6) = dwda3; J3(4, 7) = dwdb3; J3(4, 8) = dwdt3;

	return J3;
}

__host__ double* inverse_kinematics_3(double* sq, double* uk, double* sk, double* s0) {
	double* uq;

	double a1 = uk[0];
	double b1 = uk[1];
	double t1 = uk[2];
	double a2 = uk[3];
	double b2 = uk[4];
	double t2 = uk[5];
	double a3 = uk[6];
	double b3 = uk[7];
	double t3 = uk[8];

	MatrixXd jcb = jacobian(sk, a1, b1, t1, a2, b2, t2, a3, b3, t3);

	MatrixXd jcb_inv = pinv(jcb);

	VectorXd vec_uq = jcb_inv * (pointerToVector(sq, 5) - pointerToVector(sk, 5)) + pointerToVector(uk, 9);

	uq = vectorToPointer(vec_uq, 9);

	// Deal with negative time
	if(uq[2] < 0.0) {uq[2] = 0;}
	if(uq[5] < 0.0) {uq[5] = 0;}
	if(uq[8] < 0.0) {uq[8] = 0;}

	return uq;
}


