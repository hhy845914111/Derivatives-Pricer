#pragma once
#include "armadillo"

namespace rand_tools
{
	constexpr size_t seeds_len = 6;
	constexpr double norm_(2.328306549295728e-10);

	constexpr double a12(1403580), a13n(810728);
	constexpr double a21(527612), a23n(1370589);
	constexpr double m1(4294967087), m2(4294944443);

	static double default_seeds[]{ 298193, 104959, 84736, 727366, 94727, 5928384 };

	struct RollMatrixCache
	{
		int num;
		arma::Mat<int> A1_mat;
		arma::Mat<int> A2_mat;
		RollMatrixCache() : num(0) { }

	};

	static RollMatrixCache _roll_matrix_cache;

	
	inline double rand01(double *dseed)
	{
		int k;
		double p1, p2;

		p1 = a12 * dseed[1] - a13n * dseed[2];
		k = p1 / m1;
		p1 -= k * m1;
		if (p1 < 0.0) p1 += m1;
		dseed[2] = dseed[1]; dseed[1] = dseed[0]; dseed[0] = p1;

		p2 = a21 * dseed[3] - a23n * dseed[5];
		k = p2 / m2;
		p2 -= k * m2;
		if (p2 < 0.0) p2 += m2;
		dseed[5] = dseed[4]; dseed[4] = dseed[3]; dseed[3] = p2;

		if (p1 <= p2) return ((p1 - p2 + m1)*norm_);
		else return ((p1 - p2)*norm_);
	}


	inline void rand01_skip_ahead(unsigned int times, double *seed_ar)
	{
#ifdef NAIVE_SKIP
		unsigned int cnt = 0;
		while (cnt < times)
		{
			rand01(seed_ar);
			++cnt;
		}
#else
		if (times != _roll_matrix_cache.num)
		{
			arma::Mat<int> A1(3, 3, arma::fill::zeros);
			A1.at(0, 1) = a12;
			A1.at(0, 2) = -a13n;
			A1.at(1, 0) = A1.at(2, 1) = 1;

			arma::Mat<int> A2(3, 3, arma::fill::zeros);
			A2.at(0, 0) = a21;
			A2.at(0, 2) = -a23n;
			A2.at(1, 0) = A2.at(2, 1) = 1;

			_roll_matrix_cache.A1_mat = arma::pow(A1, times);
			_roll_matrix_cache.A1_mat = arma::pow(A2, times);
		}
		
		arma::Col<int> Y1{ (int)seed_ar[0], (int)seed_ar[1], (int)seed_ar[2] }, Y2{ (int)seed_ar[3], (int)seed_ar[4], (int)seed_ar[5] };

		Y1 = _roll_matrix_cache.A1_mat * Y1;
		Y2 = _roll_matrix_cache.A1_mat * Y2;

		long int _m1 = m1, _m2 = m2;
		seed_ar[0] = Y1[0] % _m1;
		seed_ar[1] = Y1[1] % _m1;
		seed_ar[2] = Y1[2] % _m1;

		seed_ar[3] = Y2[0] % _m2;
		seed_ar[4] = Y2[1] % _m2;
		seed_ar[5] = Y2[2] % _m2;
#endif
	}

	inline double randn(double *seed)
	{
		double p(rand01(seed)), q, r, ans;

		if (p <= 0.0) {
			ans = -100.0;
		}
		else {
			if (p >= 1.0) ans = 100.00;

			else {
				if (p < 0.02425) {
					q = sqrt(-2.0*log(p));
					ans = (((((-0.007784894002430293*q - 0.3223964580411365)*q - 2.400758277161838)*q - 2.549732539343734)*q + 4.374664141464968)*q + 2.938163982698783) /
						((((0.007784695709041462*q + 0.3224671290700398)*q + 2.445134137142996)*q + 3.754408661907416)*q + 1.0);
				}
				else {
					if (p < 0.97575) {
						q = p - 0.5;
						r = q * q;
						ans = (((((-39.69683028665376*r + 220.9460984245205)*r - 275.9285104469687)*r + 138.3577518672690)*r - 30.66479806614716)*r + 2.506628277459239)*q /
							(((((-54.47609879822406*r + 161.5858368580409)*r - 155.6989798598866)*r + 66.80131188771972)*r - 13.28068155288572)*r + 1.0);
					}
					else {
						if (p < 0.99) {
							q = sqrt(-2.0*log(1.0 - p));
							ans = -(((((-0.007784894002430293*q - 0.3223964580411365)*q - 2.400758277161838)*q - 2.549732539343734)*q + 4.374664141464968)*q + 2.938163982698783) /
								((((0.007784695709041462*q + 0.3224671290700398)*q + 2.445134137142996)*q + 3.754408661907416)*q + 1.0);
						}
						else {
							q = sqrt(-2.0*log(1 - p));
							ans = -(((((-0.007784894002430293*q - 0.3223964580411365)*q - 2.400758277161838)*q - 2.549732539343734)*q + 4.374664141464968)*q + 2.938163982698783) /
								((((0.007784695709041462*q + 0.3224671290700398)*q + 2.445134137142996)*q + 3.754408661907416)*q + 1.0);
						}
					}
				}
			}
		}

		return ans;
	}
}