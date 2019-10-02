#pragma once
#include <cmath>
#include "configure.h"
#include "utils.h"

inline double x2P(double x, double y, double rho, double vol)
{
	return exp(x + (2 * rho * exp(0.5 * y)) / vol);
}

inline double P2x(double P, double y, double rho, double vol)
{
	return log(P) - 2 * rho * exp(0.5 * y) / vol;
}


void explicit_step1(double* this_mat, size_t y_i, double* last_mat, size_t x_i, size_t n_i, size_t M, double r, constants::OptionType op_type,
	double dt, double dy, double Y_MIN, double rho, double vol, double K);


void explicit_step2(double* this_mat, size_t y_i, double* last_mat, size_t x_i, size_t n_i, size_t M, double r, constants::OptionType op_type,
	double dt, double dy, double Y_MIN, double kappa, double theta, double rho, double vol, double K);


void implicit_step(double* this_mat, size_t y_i, double* last_mat, double* c_vec, size_t n_i, size_t M, double r, constants::OptionType op_type,
	double dt, double dx, double dy, double Y_MIN, double rho, double kappa, double theta, double vol, double K);


inline void early_exercise(double* this_mat, size_t y_i, size_t x_i, size_t M, constants::OptionType op_type, double dx, double X_MIN, double dy, double Y_MIN, double rho, double vol, double K)
{
	if (y_i >= M)
	{
		return;
	}

	double P = x2P(X_MIN + dx * x_i, Y_MIN + dy * y_i, rho, vol), itsc = op_type == constants::Call ? P - K : K - P;
	MATRIX_GET(this_mat, x_i, y_i, M) = MAX(MATRIX_GET(this_mat, x_i, y_i, M), itsc);
}


double get_price_stochastic_vol(double P0, double y0, double K, double r, double vol, double tao, double theta, double rho, double kappa, constants::OptionType op_type,
	size_t M, size_t N, double P_MIN, double P_MAX, double Y_MIN, double Y_MAX);
