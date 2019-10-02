// Test_MRG32k3a.cpp : Defines the entry point for the console application.
//


#include "stdafx.h"
#include "SimulationFunctions.h"

#define norm 2.328306549295728e-10
#define m1   4294967087.0
#define m2   4294944443.0
#define a12     1403580.0
#define a13n     810728.0
#define a21      527612.0
#define a23n    1370589.0

#define NR_END 1
#define FREE_ARG char*

using namespace std;

double MRG32k3a(double *dseed);
void RollSeed_MRG32k3a(double *dseed);
double MRG32k3a_v2(unsigned int *seed);
void SkipAhead_MRG32k3a(int n, unsigned int **An1, unsigned int **An2);
void SkipAhead2_MRG32k3a(int n, unsigned int **An1, unsigned int **An2, unsigned int **Bn1, unsigned int **Bn2);
double **dmatrix(int nrl, int nrh, int ncl, int nch);
void free_dmatrix(double **m, int nrl, int nrh, int ncl, int nch);
unsigned int **uint_matrix(int nrl, int nrh, int ncl, int nch);
void free_uint_matrix(unsigned int **m, int nrl, int nrh, int ncl, int nch);

const unsigned int im1 = 4294967087;
const unsigned int im2 = 4294944443;
const unsigned int ia12 = 1403580;
const unsigned int ia13n = 810728;
const unsigned int ia21 = 527612;
const unsigned int ia23n = 1370589;

int main()
{
	int i, j, n, nSim, nSimPerPath;
	long long k;
	unsigned int ib1, ib2, **An1, **An2, **Bn1, **Bn2, *seed;
	unsigned long long lp1, lp2, seed1[3], seed2[3], sseed1[3], sseed2[3];
	double x, x2, x3, *dseed, *MRG_Res1, *MRG_Res2;
	//	double vseed1[3], vseed2[3], vvseed1[3], vvseed2[3];
	//	double ds11, ds12, ds13, ds21, ds22, ds23;
	//	double p1, p2;
	unsigned int s11, s12, s13, s21, s22, s23;
	double time0, time1, time2, time3, time4, time5;
	struct _timeb timebuffer;
	errno_t errcheck;

	An1 = uint_matrix(0, 2, 0, 2);
	An2 = uint_matrix(0, 2, 0, 2);
	seed = (unsigned int *)malloc(6 * sizeof(unsigned int));
	dseed = (double *)malloc(6 * sizeof(double));
	Bn1 = uint_matrix(0, 2, 0, 2);
	Bn2 = uint_matrix(0, 2, 0, 2);

//	Read inputs from text file
	FILE *fin;
	errcheck = fopen_s(&fin, "Test_MRG32k3a_CPU_Parameters.txt", "r");
	if (errcheck) printf(" File Test_MRG32k3a_CPU_Parameters.txt not opened \n");

	fscanf_s(fin, " %i %i ", &nSimPerPath, &nSim);

	printf("  Inputs: nSimPerPath, simulations per path %i  nSim %i \n", nSimPerPath, nSim);

	s11 = 298193;  s12 = 104959;  s13 = 84736;
	s21 = 727366;  s22 = 94727;   s23 = 5928384;

	ib1 = s11;
	ib2 = s12;

	dseed[0] = s11;
	dseed[1] = s12;
	dseed[2] = s13;
	dseed[3] = s21;
	dseed[4] = s22;
	dseed[5] = s23;

	seed[0] = s11;
	seed[1] = s12;
	seed[2] = s13;
	seed[3] = s21;
	seed[4] = s22;
	seed[5] = s23;

	printf("  initial seeds =  %u %u %u %u %u %u \n", s11, s12, s13, s21, s22, s23);

//	n = 10;
//	for (i = 1; i <= n; i++) {
//		x = MRG32k3a(dseed);
//		x2 = MRG32k3a_v2(seed);
//		printf(" i= %3i %13u %13u %13u %13u %13u %13u x= %12.8lf x2= %12.8lf \n",
//			i, seed[0], seed[1], seed[2], seed[3], seed[4], seed[5], x, x2);
//	}

//	n = 100000000;  // 10e07
	//	n = 1000000;   // 10e06
/*
	_ftime64_s(&timebuffer);
	time0 = timebuffer.time + timebuffer.millitm / 1000.0;

	for (i = 1; i <= n; i++) {
		x = MRG32k3a(dseed);
	}

	_ftime64_s(&timebuffer);
	time1 = timebuffer.time + timebuffer.millitm / 1000.0;

	for (i = 1; i <= n; i++) {
		x2 = MRG32k3a_v2(seed);
	}

	_ftime64_s(&timebuffer);
	time2 = timebuffer.time + timebuffer.millitm / 1000.0;

	for (i = 1; i <= n; i++) {
		x3 = rand_u01(ib1, ib2);
	}

	_ftime64_s(&timebuffer);
	time3 = timebuffer.time + timebuffer.millitm / 1000.0;

	time3 = time3 - time2;
	time2 = time2 - time1;
	time1 = time1 - time0;

	printf(" final x = %12.8lf  final x2= %12.8lf  %i simulations: time for DP method %6.3f  time for uint_32 method %6.3f \n",
		x, x2, n, time1, time2);
	printf(" time for Linear Congruential Generator method %6.3f \n \n", time3);
*/

//	n = 100;
//	printf(" Enter values of nSimPerPath and nSim for skipping ahead \n");
//	cin >> nSimPerPath;
//	cin >> nSim;

//	reset the intial seeds
//	s11 = 298193;  s12 = 104959;  s13 = 84736;
//	s21 = 727366;  s22 = 94727;   s23 = 5928384;

//	_ftime64_s(&timebuffer);
//	time0 = timebuffer.time + timebuffer.millitm / 1000.0;
/*
	dseed[0] = s11;
	dseed[1] = s12;
	dseed[2] = s13;
	dseed[3] = s21;
	dseed[4] = s22;
	dseed[5] = s23;

	n = nSim*nSimPerPath;

	for (i = 1; i <= n; i++) RollSeed_MRG32k3a(dseed);

	printf("   seeds after rolling %i times: %12.1lf %12.1lf %12.1lf %12.1lf %12.1lf %12.1lf \n",
		n, dseed[0], dseed[1], dseed[2], dseed[3], dseed[4], dseed[5]);

	_ftime64_s(&timebuffer);
	time1 = timebuffer.time + timebuffer.millitm / 1000.0;

	dseed[0] = s11;
	dseed[1] = s12;
	dseed[2] = s13;
	dseed[3] = s21;
	dseed[4] = s22;
	dseed[5] = s23;
*/

//	n = nSim*nSimPerPath;
	MRG_Res1 = (double *)malloc(nSim * sizeof(double));
	MRG_Res2 = (double *)malloc(nSim * sizeof(double));

	printf("   Now running simulations using MRG32k3a \n");

	_ftime64_s(&timebuffer);
	time0 = timebuffer.time + timebuffer.millitm / 1000.0;
	
	for (i = 0; i < nSim; i++) {
		for (j = 0; j < nSimPerPath; j++) x = MRG32k3a(dseed);
		MRG_Res1[i] = x;
	}

	printf("   Finished MRG32k3a with double precision seeds \n");

//	printf(" new seeds after %i simulations: %12.1lf %12.1lf %12.1lf %12.1lf %12.1lf %12.1lf \n  x = %12.10lf \n",
//		n, dseed[0], dseed[1], dseed[2], dseed[3], dseed[4], dseed[5], x);

	_ftime64_s(&timebuffer);
	time1 = timebuffer.time + timebuffer.millitm / 1000.0;

	for (i = 0; i < nSim; i++) {
		for (j = 0; j < nSimPerPath; j++) x = MRG32k3a_v2(seed);
		MRG_Res2[i] = x;
	}

	printf("   Finished MRG32k3a with integer seeds \n");

	_ftime64_s(&timebuffer);
	time2 = timebuffer.time + timebuffer.millitm / 1000.0;

	time2 = time2 - time1;
	time1 = time1 - time0;

	printf("  Run times: MRG(double precision) %7.3lf  MRG(unsigned int) %7.3lf  \n",
		time1, time2);

	printf("  Print first 20 rows and last 2 rows of the MRG32k3a simulations on CPU \n");
	printf("  MRG(double precision)  MRG(unsigned int) \n");
	for (i = 0; i < 20; i++) printf("   %11.8f            %11.8f  \n", MRG_Res1[i], MRG_Res2[i]);
		printf("\n   %11.8f            %11.8f \n", MRG_Res1[nSim - 2], MRG_Res2[nSim - 2]);
	printf("\n   %11.8f            %11.8f  \n \n", MRG_Res1[nSim - 1], MRG_Res2[nSim - 1]);

/*
	SkipAhead_MRG32k3a(nSimPerPath, An1, An2);
	SkipAhead2_MRG32k3a(nSim, An1, An2, Bn1, Bn2);
//	SkipAhead_MRG32k3a(nSimPerPath, Bn1, Bn2);

	seed1[0] = s11;
	seed1[1] = s12;
	seed1[2] = s13;
	seed2[0] = s21;
	seed2[1] = s22;
	seed2[2] = s23;

	for (i = 0; i < 3; i++) {
		sseed1[i] = 0.0;
		sseed2[i] = 0.0;
		for (j = 0; j < 3; j++) {
			sseed1[i] += (Bn1[i][j] * seed1[j]) % im1;
			sseed2[i] += (Bn2[i][j] * seed2[j]) % im2;
		}

		lp1 = sseed1[i];
		lp1 = lp1 % im1;
		if (lp1 < 0) lp1 += im1;
		sseed1[i] = lp1;

		lp2 = sseed2[i];
		lp2 = lp2 % im2;
		if (lp2 < 0) lp2 += im2;
		sseed2[i] = lp2;

	}

	lp1 = sseed1[0];
	lp2 = sseed2[0];
	if (lp1 <= lp2) x2 = ((lp1 - lp2 + im1)*norm);
	else x2 = ((lp1 - lp2)*norm);

	printf(" new seeds after skip ahead %i: %12llu %12llu %12llu %12llu %12llu %12llu \n  x2 = %12.10lf \n",
		n, sseed1[0], sseed1[1], sseed1[2], sseed2[0], sseed2[1], sseed2[2], x2);

	_ftime64_s(&timebuffer);
	time3 = timebuffer.time + timebuffer.millitm / 1000.0;

	time3 = time3 - time2;
	time2 = time2 - time1;
	time1 = time1 - time0;

	printf("  time1 = %7.3lf   time2 = %7.3lf   time3 = %7.3lf  \n", time1, time2, time3);

	printf(" Bn1: \n");
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) printf(" %12u ", Bn1[i][j]);
		printf("  \n");
	}
	printf(" Bn2: \n");
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) printf(" %12u ", Bn2[i][j]);
		printf("  \n");
	}
	*/

	free(seed);
	free(dseed);
	free(MRG_Res1);
	free(MRG_Res2);
	free_uint_matrix(An1, 0, 2, 0, 2);
	free_uint_matrix(An2, 0, 2, 0, 2);
	free_uint_matrix(Bn1, 0, 2, 0, 2);
	free_uint_matrix(Bn2, 0, 2, 0, 2);

	return 0;

}
double MRG32k3a(double *dseed)
{
	//	This code is an exact copy of the C code in L'Ecuyer (Operations Research 1999)
	int k;
	double p1, p2;

	p1 = a12*dseed[1] - a13n*dseed[2];
	k = p1 / m1;
	p1 -= k*m1;
	if (p1 < 0.0) p1 += m1;
	dseed[2] = dseed[1]; dseed[1] = dseed[0]; dseed[0] = p1;

	p2 = a21*dseed[3] - a23n*dseed[5];
	k = p2 / m2;
	p2 -= k*m2;
	if (p2 < 0.0) p2 += m2;
	dseed[5] = dseed[4]; dseed[4] = dseed[3]; dseed[3] = p2;

	if (p1 <= p2) return ((p1 - p2 + m1)*norm);
	else return ((p1 - p2)*norm);

}

void RollSeed_MRG32k3a(double *dseed)
{
	int k;
	double p1, p2;

	p1 = a12*dseed[1] - a13n*dseed[2];
	k = p1 / m1;
	p1 -= k*m1;
	if (p1 < 0.0) p1 += m1;
	dseed[2] = dseed[1]; dseed[1] = dseed[0]; dseed[0] = p1;

	p2 = a21*dseed[3] - a23n*dseed[5];
	k = p2 / m2;
	p2 -= k*m2;
	if (p2 < 0.0) p2 += m2;
	dseed[5] = dseed[4]; dseed[4] = dseed[3]; dseed[3] = p2;

	return;

}

/*
//	This method is slightly faster and more accurate (always spot on, matches brute force rolling of seeds)
//	This method is very fast
void SkipAhead_MRG32k3a(int n, unsigned int **An1, unsigned int **An2)
{
	int i, j, ii;
	long long kmod, lp1, lp2;
	long long A1[3][3], A2[3][3], B1[3][3], B2[3][3], C1[3][3], C2[3][3];

	A1[0][0] = 0; A1[0][1] = ia12;
	A1[0][2] = 0;
	A1[0][2] -= ia13n;
	//	A1[0][2] = -ia13n;
	A1[1][0] = 1; A1[1][1] = 0; A1[1][2] = 0;
	A1[2][0] = 0; A1[2][1] = 1; A1[2][2] = 0;

	A2[0][0] = ia21; A2[0][1] = 0;
	A2[0][2] = 0;
	A2[0][2] -= ia23n;
	//	A2[0][2] = -ia23n;
	A2[1][0] = 1; A2[1][1] = 0; A2[1][2] = 0;
	A2[2][0] = 0; A2[2][1] = 1; A2[2][2] = 0;

	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			B1[i][j] = A1[i][j];
			B2[i][j] = A2[i][j];
		}
	}

	for (ii = 1; ii <= (n - 1); ii++) {
		//	pre-multiply by Ai, calculating with 64 bit signed integers
		C1[0][0] = A1[0][0] * B1[0][0] + A1[0][1] * B1[1][0] + A1[0][2] * B1[2][0];
		C1[0][1] = A1[0][0] * B1[0][1] + A1[0][1] * B1[1][1] + A1[0][2] * B1[2][1];
		C1[0][2] = A1[0][0] * B1[0][2] + A1[0][1] * B1[1][2] + A1[0][2] * B1[2][2];
		C1[1][0] = A1[1][0] * B1[0][0] + A1[1][1] * B1[1][0] + A1[1][2] * B1[2][0];
		C1[1][1] = A1[1][0] * B1[0][1] + A1[1][1] * B1[1][1] + A1[1][2] * B1[2][1];
		C1[1][2] = A1[1][0] * B1[0][2] + A1[1][1] * B1[1][2] + A1[1][2] * B1[2][2];
		C1[2][0] = A1[2][0] * B1[0][0] + A1[2][1] * B1[1][0] + A1[2][2] * B1[2][0];
		C1[2][1] = A1[2][0] * B1[0][1] + A1[2][1] * B1[1][1] + A1[2][2] * B1[2][1];
		C1[2][2] = A1[2][0] * B1[0][2] + A1[2][1] * B1[1][2] + A1[2][2] * B1[2][2];

		C2[0][0] = A2[0][0] * B2[0][0] + A2[0][1] * B2[1][0] + A2[0][2] * B2[2][0];
		C2[0][1] = A2[0][0] * B2[0][1] + A2[0][1] * B2[1][1] + A2[0][2] * B2[2][1];
		C2[0][2] = A2[0][0] * B2[0][2] + A2[0][1] * B2[1][2] + A2[0][2] * B2[2][2];
		C2[1][0] = A2[1][0] * B2[0][0] + A2[1][1] * B2[1][0] + A2[1][2] * B2[2][0];
		C2[1][1] = A2[1][0] * B2[0][1] + A2[1][1] * B2[1][1] + A2[1][2] * B2[2][1];
		C2[1][2] = A2[1][0] * B2[0][2] + A2[1][1] * B2[1][2] + A2[1][2] * B2[2][2];
		C2[2][0] = A2[2][0] * B2[0][0] + A2[2][1] * B2[1][0] + A2[2][2] * B2[2][0];
		C2[2][1] = A2[2][0] * B2[0][1] + A2[2][1] * B2[1][1] + A2[2][2] * B2[2][1];
		C2[2][2] = A2[2][0] * B2[0][2] + A2[2][1] * B2[1][2] + A2[2][2] * B2[2][2];

		for (i = 0; i < 3; i++) {
			for (j = 0; j < 3; j++) {
				lp1 = C1[i][j];
				lp1 = lp1 % im1;
				if (lp1 < 0) lp1 += im1;
				B1[i][j] = lp1;
				lp2 = C2[i][j];
				lp2 = lp2 % im2;
				if (lp2 < 0) lp2 += im2;
				B2[i][j] = lp2;
			}
		}
	}

	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			An1[i][j] = B1[i][j];
			An2[i][j] = B2[i][j];
		}
	}

	return;

}
*/


//	This method is slightly faster and accurate (always spot on, matches brute force rolling of seeds)
//	This method is very fast because it uses divdie and conquer
void SkipAhead_MRG32k3a(int n, unsigned int **An1, unsigned int **An2)
{
	const unsigned int im1 = 4294967087;
	const unsigned int im2 = 4294944443;
	const unsigned int ia12 = 1403580;
	const unsigned int ia13n = 810728;
	const unsigned int ia21 = 527612;
	const unsigned int ia23n = 1370589;
	int i, j, k, ii;
	long long kmod, lp1, lp2;
	long long A1[3][3], A2[3][3], B1[3][3], B2[3][3], C1[3][3], C2[3][3];
	unsigned long long BB1[3][3], BB2[3][3], CC1[3][3], CC2[3][3];

	A1[0][0] = 0; A1[0][1] = ia12;
	A1[0][2] = 0;
	A1[0][2] -= ia13n;
	//	A1[0][2] = -ia13n;
	A1[1][0] = 1; A1[1][1] = 0; A1[1][2] = 0;
	A1[2][0] = 0; A1[2][1] = 1; A1[2][2] = 0;

	A2[0][0] = ia21; A2[0][1] = 0;
	A2[0][2] = 0;
	A2[0][2] -= ia23n;
	//	A2[0][2] = -ia23n;
	A2[1][0] = 1; A2[1][1] = 0; A2[1][2] = 0;
	A2[2][0] = 0; A2[2][1] = 1; A2[2][2] = 0;

	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			B1[i][j] = A1[i][j];
			B2[i][j] = A2[i][j];
		}
	}

	//	printf(" initial A1: \n");
	//	for (i = 0; i < 3; i++) {
	//		for (j = 0; j < 3; j++) printf(" %12lli ", A1[i][j]);
	//		printf("  \n");
	//	}
	//	printf(" initial A2: \n");
	//	for (i = 0; i < 3; i++) {
	//		for (j = 0; j < 3; j++) printf(" %12lli ", A2[i][j]);
	//		printf("  \n");
	//	}

	for (ii = 2; ii <= 4; ii++) {
		//	pre-multiply by Ai, calculating with 64 bit signed integers
		C1[0][0] = A1[0][0] * B1[0][0] + A1[0][1] * B1[1][0] + A1[0][2] * B1[2][0];
		C1[0][1] = A1[0][0] * B1[0][1] + A1[0][1] * B1[1][1] + A1[0][2] * B1[2][1];
		C1[0][2] = A1[0][0] * B1[0][2] + A1[0][1] * B1[1][2] + A1[0][2] * B1[2][2];
		C1[1][0] = A1[1][0] * B1[0][0] + A1[1][1] * B1[1][0] + A1[1][2] * B1[2][0];
		C1[1][1] = A1[1][0] * B1[0][1] + A1[1][1] * B1[1][1] + A1[1][2] * B1[2][1];
		C1[1][2] = A1[1][0] * B1[0][2] + A1[1][1] * B1[1][2] + A1[1][2] * B1[2][2];
		C1[2][0] = A1[2][0] * B1[0][0] + A1[2][1] * B1[1][0] + A1[2][2] * B1[2][0];
		C1[2][1] = A1[2][0] * B1[0][1] + A1[2][1] * B1[1][1] + A1[2][2] * B1[2][1];
		C1[2][2] = A1[2][0] * B1[0][2] + A1[2][1] * B1[1][2] + A1[2][2] * B1[2][2];

		C2[0][0] = A2[0][0] * B2[0][0] + A2[0][1] * B2[1][0] + A2[0][2] * B2[2][0];
		C2[0][1] = A2[0][0] * B2[0][1] + A2[0][1] * B2[1][1] + A2[0][2] * B2[2][1];
		C2[0][2] = A2[0][0] * B2[0][2] + A2[0][1] * B2[1][2] + A2[0][2] * B2[2][2];
		C2[1][0] = A2[1][0] * B2[0][0] + A2[1][1] * B2[1][0] + A2[1][2] * B2[2][0];
		C2[1][1] = A2[1][0] * B2[0][1] + A2[1][1] * B2[1][1] + A2[1][2] * B2[2][1];
		C2[1][2] = A2[1][0] * B2[0][2] + A2[1][1] * B2[1][2] + A2[1][2] * B2[2][2];
		C2[2][0] = A2[2][0] * B2[0][0] + A2[2][1] * B2[1][0] + A2[2][2] * B2[2][0];
		C2[2][1] = A2[2][0] * B2[0][1] + A2[2][1] * B2[1][1] + A2[2][2] * B2[2][1];
		C2[2][2] = A2[2][0] * B2[0][2] + A2[2][1] * B2[1][2] + A2[2][2] * B2[2][2];

		for (i = 0; i < 3; i++) {
			for (j = 0; j < 3; j++) {
				lp1 = C1[i][j];
				lp1 = lp1 % im1;
				if (lp1 < 0) lp1 += im1;
				B1[i][j] = lp1;
				lp2 = C2[i][j];
				lp2 = lp2 % im2;
				if (lp2 < 0) lp2 += im2;
				B2[i][j] = lp2;
			}
		}

	}

	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			BB1[i][j] = B1[i][j];
			BB2[i][j] = B2[i][j];
		}
	}

	ii = 8;
	while (ii <= n) {
		//	here we are squaring the matrix at each round
		CC1[0][0] = ((BB1[0][0] * BB1[0][0]) % im1 + (BB1[0][1] * BB1[1][0]) % im1 + (BB1[0][2] * BB1[2][0]) % im1) % im1;
		CC1[0][1] = ((BB1[0][0] * BB1[0][1]) % im1 + (BB1[0][1] * BB1[1][1]) % im1 + (BB1[0][2] * BB1[2][1]) % im1) % im1;
		CC1[0][2] = ((BB1[0][0] * BB1[0][2]) % im1 + (BB1[0][1] * BB1[1][2]) % im1 + (BB1[0][2] * BB1[2][2]) % im1) % im1;
		CC1[1][0] = ((BB1[1][0] * BB1[0][0]) % im1 + (BB1[1][1] * BB1[1][0]) % im1 + (BB1[1][2] * BB1[2][0]) % im1) % im1;
		CC1[1][1] = ((BB1[1][0] * BB1[0][1]) % im1 + (BB1[1][1] * BB1[1][1]) % im1 + (BB1[1][2] * BB1[2][1]) % im1) % im1;
		CC1[1][2] = ((BB1[1][0] * BB1[0][2]) % im1 + (BB1[1][1] * BB1[1][2]) % im1 + (BB1[1][2] * BB1[2][2]) % im1) % im1;
		CC1[2][0] = ((BB1[2][0] * BB1[0][0]) % im1 + (BB1[2][1] * BB1[1][0]) % im1 + (BB1[2][2] * BB1[2][0]) % im1) % im1;
		CC1[2][1] = ((BB1[2][0] * BB1[0][1]) % im1 + (BB1[2][1] * BB1[1][1]) % im1 + (BB1[2][2] * BB1[2][1]) % im1) % im1;
		CC1[2][2] = ((BB1[2][0] * BB1[0][2]) % im1 + (BB1[2][1] * BB1[1][2]) % im1 + (BB1[2][2] * BB1[2][2]) % im1) % im1;

		CC2[0][0] = ((BB2[0][0] * BB2[0][0]) % im2 + (BB2[0][1] * BB2[1][0]) % im2 + (BB2[0][2] * BB2[2][0]) % im2) % im2;
		CC2[0][1] = ((BB2[0][0] * BB2[0][1]) % im2 + (BB2[0][1] * BB2[1][1]) % im2 + (BB2[0][2] * BB2[2][1]) % im2) % im2;
		CC2[0][2] = ((BB2[0][0] * BB2[0][2]) % im2 + (BB2[0][1] * BB2[1][2]) % im2 + (BB2[0][2] * BB2[2][2]) % im2) % im2;
		CC2[1][0] = ((BB2[1][0] * BB2[0][0]) % im2 + (BB2[1][1] * BB2[1][0]) % im2 + (BB2[1][2] * BB2[2][0]) % im2) % im2;
		CC2[1][1] = ((BB2[1][0] * BB2[0][1]) % im2 + (BB2[1][1] * BB2[1][1]) % im2 + (BB2[1][2] * BB2[2][1]) % im2) % im2;
		CC2[1][2] = ((BB2[1][0] * BB2[0][2]) % im2 + (BB2[1][1] * BB2[1][2]) % im2 + (BB2[1][2] * BB2[2][2]) % im2) % im2;
		CC2[2][0] = ((BB2[2][0] * BB2[0][0]) % im2 + (BB2[2][1] * BB2[1][0]) % im2 + (BB2[2][2] * BB2[2][0]) % im2) % im2;
		CC2[2][1] = ((BB2[2][0] * BB2[0][1]) % im2 + (BB2[2][1] * BB2[1][1]) % im2 + (BB2[2][2] * BB2[2][1]) % im2) % im2;
		CC2[2][2] = ((BB2[2][0] * BB2[0][2]) % im2 + (BB2[2][1] * BB2[1][2]) % im2 + (BB2[2][2] * BB2[2][2]) % im2) % im2;

		for (i = 0; i < 3; i++) {
			for (j = 0; j < 3; j++) {
				BB1[i][j] = CC1[i][j];
				BB2[i][j] = CC2[i][j];
			}

		}

		ii = 2 * ii;
	}

	k = ii / 2;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			B1[i][j] = BB1[i][j];
			B2[i][j] = BB2[i][j];
		}
	}

	for (ii = (k + 1); ii <= n; ii++) {
		//	pre-multiply by Ai, calculating with 64 bit signed integers
		C1[0][0] = A1[0][0] * B1[0][0] + A1[0][1] * B1[1][0] + A1[0][2] * B1[2][0];
		C1[0][1] = A1[0][0] * B1[0][1] + A1[0][1] * B1[1][1] + A1[0][2] * B1[2][1];
		C1[0][2] = A1[0][0] * B1[0][2] + A1[0][1] * B1[1][2] + A1[0][2] * B1[2][2];
		C1[1][0] = A1[1][0] * B1[0][0] + A1[1][1] * B1[1][0] + A1[1][2] * B1[2][0];
		C1[1][1] = A1[1][0] * B1[0][1] + A1[1][1] * B1[1][1] + A1[1][2] * B1[2][1];
		C1[1][2] = A1[1][0] * B1[0][2] + A1[1][1] * B1[1][2] + A1[1][2] * B1[2][2];
		C1[2][0] = A1[2][0] * B1[0][0] + A1[2][1] * B1[1][0] + A1[2][2] * B1[2][0];
		C1[2][1] = A1[2][0] * B1[0][1] + A1[2][1] * B1[1][1] + A1[2][2] * B1[2][1];
		C1[2][2] = A1[2][0] * B1[0][2] + A1[2][1] * B1[1][2] + A1[2][2] * B1[2][2];

		C2[0][0] = A2[0][0] * B2[0][0] + A2[0][1] * B2[1][0] + A2[0][2] * B2[2][0];
		C2[0][1] = A2[0][0] * B2[0][1] + A2[0][1] * B2[1][1] + A2[0][2] * B2[2][1];
		C2[0][2] = A2[0][0] * B2[0][2] + A2[0][1] * B2[1][2] + A2[0][2] * B2[2][2];
		C2[1][0] = A2[1][0] * B2[0][0] + A2[1][1] * B2[1][0] + A2[1][2] * B2[2][0];
		C2[1][1] = A2[1][0] * B2[0][1] + A2[1][1] * B2[1][1] + A2[1][2] * B2[2][1];
		C2[1][2] = A2[1][0] * B2[0][2] + A2[1][1] * B2[1][2] + A2[1][2] * B2[2][2];
		C2[2][0] = A2[2][0] * B2[0][0] + A2[2][1] * B2[1][0] + A2[2][2] * B2[2][0];
		C2[2][1] = A2[2][0] * B2[0][1] + A2[2][1] * B2[1][1] + A2[2][2] * B2[2][1];
		C2[2][2] = A2[2][0] * B2[0][2] + A2[2][1] * B2[1][2] + A2[2][2] * B2[2][2];

		for (i = 0; i < 3; i++) {
			for (j = 0; j < 3; j++) {
				lp1 = C1[i][j];
				lp1 = lp1 % im1;
				if (lp1 < 0) lp1 += im1;
				B1[i][j] = lp1;
				lp2 = C2[i][j];
				lp2 = lp2 % im2;
				if (lp2 < 0) lp2 += im2;
				B2[i][j] = lp2;
			}
		}
	}

	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			An1[i][j] = B1[i][j];
			An2[i][j] = B2[i][j];
		}
	}

	return;

}




void SkipAhead2_MRG32k3a(int n, unsigned int **An1, unsigned int **An2, unsigned int **Bn1, unsigned int **Bn2)
{
	const unsigned int im1 = 4294967087;
	const unsigned int im2 = 4294944443;
	//	const unsigned int ia12 = 1403580;
	//	const unsigned int ia13n = 810728;
	//	const unsigned int ia21 = 527612;
	//	const unsigned int ia23n = 1370589;
	int i, j, k, ii;
	long long kmod, lp1, lp2;
//	long long B1[3][3], B2[3][3], C1[3][3], C2[3][3];
	unsigned long long A1[3][3], A2[3][3], BB1[3][3], BB2[3][3], CC1[3][3], CC2[3][3];

	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			A1[i][j] = An1[i][j];
			A2[i][j] = An2[i][j];
			BB1[i][j] = An1[i][j];
			BB2[i][j] = An2[i][j];
		}
	}

	ii = 2;
	while (ii <= n) {
		//	here we are squaring the matrix at each round
		CC1[0][0] = ((BB1[0][0] * BB1[0][0]) % im1 + (BB1[0][1] * BB1[1][0]) % im1 + (BB1[0][2] * BB1[2][0]) % im1) % im1;
		CC1[0][1] = ((BB1[0][0] * BB1[0][1]) % im1 + (BB1[0][1] * BB1[1][1]) % im1 + (BB1[0][2] * BB1[2][1]) % im1) % im1;
		CC1[0][2] = ((BB1[0][0] * BB1[0][2]) % im1 + (BB1[0][1] * BB1[1][2]) % im1 + (BB1[0][2] * BB1[2][2]) % im1) % im1;
		CC1[1][0] = ((BB1[1][0] * BB1[0][0]) % im1 + (BB1[1][1] * BB1[1][0]) % im1 + (BB1[1][2] * BB1[2][0]) % im1) % im1;
		CC1[1][1] = ((BB1[1][0] * BB1[0][1]) % im1 + (BB1[1][1] * BB1[1][1]) % im1 + (BB1[1][2] * BB1[2][1]) % im1) % im1;
		CC1[1][2] = ((BB1[1][0] * BB1[0][2]) % im1 + (BB1[1][1] * BB1[1][2]) % im1 + (BB1[1][2] * BB1[2][2]) % im1) % im1;
		CC1[2][0] = ((BB1[2][0] * BB1[0][0]) % im1 + (BB1[2][1] * BB1[1][0]) % im1 + (BB1[2][2] * BB1[2][0]) % im1) % im1;
		CC1[2][1] = ((BB1[2][0] * BB1[0][1]) % im1 + (BB1[2][1] * BB1[1][1]) % im1 + (BB1[2][2] * BB1[2][1]) % im1) % im1;
		CC1[2][2] = ((BB1[2][0] * BB1[0][2]) % im1 + (BB1[2][1] * BB1[1][2]) % im1 + (BB1[2][2] * BB1[2][2]) % im1) % im1;

		CC2[0][0] = ((BB2[0][0] * BB2[0][0]) % im2 + (BB2[0][1] * BB2[1][0]) % im2 + (BB2[0][2] * BB2[2][0]) % im2) % im2;
		CC2[0][1] = ((BB2[0][0] * BB2[0][1]) % im2 + (BB2[0][1] * BB2[1][1]) % im2 + (BB2[0][2] * BB2[2][1]) % im2) % im2;
		CC2[0][2] = ((BB2[0][0] * BB2[0][2]) % im2 + (BB2[0][1] * BB2[1][2]) % im2 + (BB2[0][2] * BB2[2][2]) % im2) % im2;
		CC2[1][0] = ((BB2[1][0] * BB2[0][0]) % im2 + (BB2[1][1] * BB2[1][0]) % im2 + (BB2[1][2] * BB2[2][0]) % im2) % im2;
		CC2[1][1] = ((BB2[1][0] * BB2[0][1]) % im2 + (BB2[1][1] * BB2[1][1]) % im2 + (BB2[1][2] * BB2[2][1]) % im2) % im2;
		CC2[1][2] = ((BB2[1][0] * BB2[0][2]) % im2 + (BB2[1][1] * BB2[1][2]) % im2 + (BB2[1][2] * BB2[2][2]) % im2) % im2;
		CC2[2][0] = ((BB2[2][0] * BB2[0][0]) % im2 + (BB2[2][1] * BB2[1][0]) % im2 + (BB2[2][2] * BB2[2][0]) % im2) % im2;
		CC2[2][1] = ((BB2[2][0] * BB2[0][1]) % im2 + (BB2[2][1] * BB2[1][1]) % im2 + (BB2[2][2] * BB2[2][1]) % im2) % im2;
		CC2[2][2] = ((BB2[2][0] * BB2[0][2]) % im2 + (BB2[2][1] * BB2[1][2]) % im2 + (BB2[2][2] * BB2[2][2]) % im2) % im2;

		for (i = 0; i < 3; i++) {
			for (j = 0; j < 3; j++) {
				BB1[i][j] = CC1[i][j];
				BB2[i][j] = CC2[i][j];
			}
		}

		ii = 2 * ii;
	}

	k = ii / 2;
	//	for (i = 0; i < 3; i++) {
	//		for (j = 0; j < 3; j++) {
	//			B1[i][j] = BB1[i][j];
	//			B2[i][j] = BB2[i][j];
	//		}
	//	}

	for (ii = (k + 1); ii <= n; ii++) {
		//	pre-multiply by Ai, calculating with 64 bit signed integers

		CC1[0][0] = ((A1[0][0] * BB1[0][0]) % im1 + (A1[0][1] * BB1[1][0]) % im1 + (A1[0][2] * BB1[2][0]) % im1) % im1;
		CC1[0][1] = ((A1[0][0] * BB1[0][1]) % im1 + (A1[0][1] * BB1[1][1]) % im1 + (A1[0][2] * BB1[2][1]) % im1) % im1;
		CC1[0][2] = ((A1[0][0] * BB1[0][2]) % im1 + (A1[0][1] * BB1[1][2]) % im1 + (A1[0][2] * BB1[2][2]) % im1) % im1;
		CC1[1][0] = ((A1[1][0] * BB1[0][0]) % im1 + (A1[1][1] * BB1[1][0]) % im1 + (A1[1][2] * BB1[2][0]) % im1) % im1;
		CC1[1][1] = ((A1[1][0] * BB1[0][1]) % im1 + (A1[1][1] * BB1[1][1]) % im1 + (A1[1][2] * BB1[2][1]) % im1) % im1;
		CC1[1][2] = ((A1[1][0] * BB1[0][2]) % im1 + (A1[1][1] * BB1[1][2]) % im1 + (A1[1][2] * BB1[2][2]) % im1) % im1;
		CC1[2][0] = ((A1[2][0] * BB1[0][0]) % im1 + (A1[2][1] * BB1[1][0]) % im1 + (A1[2][2] * BB1[2][0]) % im1) % im1;
		CC1[2][1] = ((A1[2][0] * BB1[0][1]) % im1 + (A1[2][1] * BB1[1][1]) % im1 + (A1[2][2] * BB1[2][1]) % im1) % im1;
		CC1[2][2] = ((A1[2][0] * BB1[0][2]) % im1 + (A1[2][1] * BB1[1][2]) % im1 + (A1[2][2] * BB1[2][2]) % im1) % im1;

		CC2[0][0] = ((A2[0][0] * BB2[0][0]) % im2 + (A2[0][1] * BB2[1][0]) % im2 + (A2[0][2] * BB2[2][0]) % im2) % im2;
		CC2[0][1] = ((A2[0][0] * BB2[0][1]) % im2 + (A2[0][1] * BB2[1][1]) % im2 + (A2[0][2] * BB2[2][1]) % im2) % im2;
		CC2[0][2] = ((A2[0][0] * BB2[0][2]) % im2 + (A2[0][1] * BB2[1][2]) % im2 + (A2[0][2] * BB2[2][2]) % im2) % im2;
		CC2[1][0] = ((A2[1][0] * BB2[0][0]) % im2 + (A2[1][1] * BB2[1][0]) % im2 + (A2[1][2] * BB2[2][0]) % im2) % im2;
		CC2[1][1] = ((A2[1][0] * BB2[0][1]) % im2 + (A2[1][1] * BB2[1][1]) % im2 + (A2[1][2] * BB2[2][1]) % im2) % im2;
		CC2[1][2] = ((A2[1][0] * BB2[0][2]) % im2 + (A2[1][1] * BB2[1][2]) % im2 + (A2[1][2] * BB2[2][2]) % im2) % im2;
		CC2[2][0] = ((A2[2][0] * BB2[0][0]) % im2 + (A2[2][1] * BB2[1][0]) % im2 + (A2[2][2] * BB2[2][0]) % im2) % im2;
		CC2[2][1] = ((A2[2][0] * BB2[0][1]) % im2 + (A2[2][1] * BB2[1][1]) % im2 + (A2[2][2] * BB2[2][1]) % im2) % im2;
		CC2[2][2] = ((A2[2][0] * BB2[0][2]) % im2 + (A2[2][1] * BB2[1][2]) % im2 + (A2[2][2] * BB2[2][2]) % im2) % im2;

		for (i = 0; i < 3; i++) {
			for (j = 0; j < 3; j++) {
				BB1[i][j] = CC1[i][j];
				BB2[i][j] = CC2[i][j];
			}
		}

		/*
		C1[0][0] = A1[0][0] * B1[0][0] + A1[0][1] * B1[1][0] + A1[0][2] * B1[2][0];
		C1[0][1] = A1[0][0] * B1[0][1] + A1[0][1] * B1[1][1] + A1[0][2] * B1[2][1];
		C1[0][2] = A1[0][0] * B1[0][2] + A1[0][1] * B1[1][2] + A1[0][2] * B1[2][2];
		C1[1][0] = A1[1][0] * B1[0][0] + A1[1][1] * B1[1][0] + A1[1][2] * B1[2][0];
		C1[1][1] = A1[1][0] * B1[0][1] + A1[1][1] * B1[1][1] + A1[1][2] * B1[2][1];
		C1[1][2] = A1[1][0] * B1[0][2] + A1[1][1] * B1[1][2] + A1[1][2] * B1[2][2];
		C1[2][0] = A1[2][0] * B1[0][0] + A1[2][1] * B1[1][0] + A1[2][2] * B1[2][0];
		C1[2][1] = A1[2][0] * B1[0][1] + A1[2][1] * B1[1][1] + A1[2][2] * B1[2][1];
		C1[2][2] = A1[2][0] * B1[0][2] + A1[2][1] * B1[1][2] + A1[2][2] * B1[2][2];

		C2[0][0] = A2[0][0] * B2[0][0] + A2[0][1] * B2[1][0] + A2[0][2] * B2[2][0];
		C2[0][1] = A2[0][0] * B2[0][1] + A2[0][1] * B2[1][1] + A2[0][2] * B2[2][1];
		C2[0][2] = A2[0][0] * B2[0][2] + A2[0][1] * B2[1][2] + A2[0][2] * B2[2][2];
		C2[1][0] = A2[1][0] * B2[0][0] + A2[1][1] * B2[1][0] + A2[1][2] * B2[2][0];
		C2[1][1] = A2[1][0] * B2[0][1] + A2[1][1] * B2[1][1] + A2[1][2] * B2[2][1];
		C2[1][2] = A2[1][0] * B2[0][2] + A2[1][1] * B2[1][2] + A2[1][2] * B2[2][2];
		C2[2][0] = A2[2][0] * B2[0][0] + A2[2][1] * B2[1][0] + A2[2][2] * B2[2][0];
		C2[2][1] = A2[2][0] * B2[0][1] + A2[2][1] * B2[1][1] + A2[2][2] * B2[2][1];
		C2[2][2] = A2[2][0] * B2[0][2] + A2[2][1] * B2[1][2] + A2[2][2] * B2[2][2];

		for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
		lp1 = C1[i][j];
		lp1 = lp1 % im1;
		if (lp1 < 0) lp1 += im1;
		B1[i][j] = lp1;
		lp2 = C2[i][j];
		lp2 = lp2 % im2;
		if (lp2 < 0) lp2 += im2;
		B2[i][j] = lp2;
		}
		}
		*/

	}

	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			Bn1[i][j] = BB1[i][j];
			Bn2[i][j] = BB2[i][j];
		}
	}

	return;

}



double MRG32k3a_v2(unsigned int *seed)
{
	double f;
	long long lp1, lp2;

	lp1 = seed[1];
	lp2 = seed[2];
	lp1 = ia12*lp1 - ia13n*lp2;
	lp1 = lp1 % im1;
	if (lp1 < 0) lp1 += im1;

	seed[2] = seed[1]; seed[1] = seed[0];
	seed[0] = lp1;

	lp1 = seed[3];
	lp2 = seed[5];
	lp2 = ia21*lp1 - ia23n*lp2;
	lp2 = lp2 % im2;
	if (lp2 < 0) lp2 += im2;

	seed[5] = seed[4]; seed[4] = seed[3];
	seed[3] = lp2;

	if (seed[0] <= seed[3]) f = ((seed[0] - seed[3] + im1)*norm);
	else f = ((seed[0] - seed[3])*norm);

	return f;

}

double **dmatrix(int nrl, int nrh, int ncl, int nch)
/* allocate a double matrix with subscript range m[nrl..nrh][ncl..nch] */
{
	int i, nrow = nrh - nrl + 1, ncol = nch - ncl + 1;
	double **m;

	/* allocate pointers to rows */
	m = (double **)malloc((size_t)((nrow + NR_END)*sizeof(double*)));
	m += NR_END;
	m -= nrl;

	/* allocate rows and set pointers to them */
	m[nrl] = (double *)malloc((size_t)((nrow*ncol + NR_END)*sizeof(double)));
	m[nrl] += NR_END;
	m[nrl] -= ncl;

	for (i = nrl + 1; i <= nrh; i++) m[i] = m[i - 1] + ncol;

	/* return pointer to array of pointers to rows */
	return m;
}

void free_dmatrix(double **m, int nrl, int nrh, int ncl, int nch)
/* free a double matrix allocated by dmatrix() */
{
	free((FREE_ARG)(m[nrl] + ncl - NR_END));
	free((FREE_ARG)(m + nrl - NR_END));
}


unsigned int **uint_matrix(int nrl, int nrh, int ncl, int nch)
/* allocate an unsigned int matrix with subscript range m[nrl..nrh][ncl..nch] */
{
	int i, nrow = nrh - nrl + 1, ncol = nch - ncl + 1;
	unsigned int **m;

	/* allocate pointers to rows */
	m = (unsigned int **)malloc((size_t)((nrow + NR_END) * sizeof(unsigned int*)));
	m += NR_END;
	m -= nrl;

	/* allocate rows and set pointers to them */
	m[nrl] = (unsigned int *)malloc((size_t)((nrow*ncol + NR_END) * sizeof(unsigned int)));
	m[nrl] += NR_END;
	m[nrl] -= ncl;

	for (i = nrl + 1; i <= nrh; i++) m[i] = m[i - 1] + ncol;

	/* return pointer to array of pointers to rows */
	return m;
}

void free_uint_matrix(unsigned int **m, int nrl, int nrh, int ncl, int nch)
/* free a double matrix allocated by dmatrix() */
{
	free((FREE_ARG)(m[nrl] + ncl - NR_END));
	free((FREE_ARG)(m + nrl - NR_END));
}
