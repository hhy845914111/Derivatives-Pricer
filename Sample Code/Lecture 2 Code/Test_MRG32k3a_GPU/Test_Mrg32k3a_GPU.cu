// Test_MRG32k3a_GPU.cu 
//
//

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "stdafx.h"
#include "Test_MRG32k3a_kernel.cuh"

#include <cuda.h>
#include <curand.h>

//    These are required for Numerical Recipes in C

using namespace std;

/* Global Variables */
//	These variables are used by MRG32k3a
#define norm 2.328306549295728e-10
#define m1   4294967087.0
#define m2   4294944443.0
#define a12     1403580.0
#define a13n     810728.0
#define a21      527612.0
#define a23n    1370589.0

#define NR_END 1
#define FREE_ARG char*

double MRG32k3a(double *dseed);
void RollSeed_MRG32k3a(double *dseed);
double MRG32k3a_v2(unsigned int *seed);
void SkipAhead_MRG32k3a(int n, unsigned int **An1, unsigned int **An2);
unsigned int **uint_matrix(int nrl, int nrh, int ncl, int nch);
void free_uint_matrix(unsigned int **m, int nrl, int nrh, int ncl, int nch);

const unsigned int im1 = 4294967087;
const unsigned int im2 = 4294944443;
const unsigned int ia12 = 1403580;
const unsigned int ia13n = 810728;
const unsigned int ia21 = 527612;
const unsigned int ia23n = 1370589;

FILE *fout;

int main()
{
	int i, j, n, n1, n2, cudaNumber, ind, nSim, ii;
	long long k;
	unsigned int **An1, **An2, *seed, *h_seeds;
	unsigned long long lp1, lp2, seed1[3], seed2[3], sseed1[3], sseed2[3];
	double x, x2, *dseed;
	unsigned int s11, s12, s13, s21, s22, s23, ib1, ib2, ia_k2, ia_k1;
	float *h_MRG_rng1, float *h_MRG_rng2, float *h_MRG_rng4;


	float *d_MRG_rng;
	unsigned int *d_seeds;
	curandStateMRG32k3a *devMRGStates;

	double time0, time1, time2, time3, time4, time5, time6;
	struct _timeb timebuffer;
	errno_t errcheck;
	cudaError_t cudaStatus;

	errcheck = fopen_s(&fout, "Test_MRG32k3a_GPU.txt", "w");
	if (errcheck) printf(" Test_MRG32k3a_GPU.txt not opened \n");

//	cudaNumber = 3;
//	cudaStatus = cudaSetDevice(cudaNumber);

	fprintf(fout, " Test MRG32k3a random number generator on CPU and GPU \n");

	//	Read inputs from text file
	fprintf(fout, " Now reading input files \n");
	FILE *fin;
	errcheck = fopen_s(&fin, "Test_MRG32k3a_GPU_Parameters.txt", "r");
	if (errcheck) printf(" File Test_MRG32k3a_GPU_Parameters.txt not opened \n");

	fscanf_s(fin, " %i %i %i ", &ind, &n, &nSim);

	printf("  Inputs: n, simulations per path %i  nSim %i \n", n, nSim);

	An1 = uint_matrix(0, 2, 0, 2);
	An2 = uint_matrix(0, 2, 0, 2);
	seed = (unsigned int *)malloc(6 * sizeof(unsigned int));
	dseed = (double *)malloc(6 * sizeof(double));

//	Initial seeds for this test run
	s11 = 298193;  s12 = 104959;  s13 = 84736;
	s21 = 727366;  s22 = 94727;   s23 = 5928384;

	fscanf_s(fin, " %ui %ui %ui  %ui %ui %ui ", &s11, &s12, &s13, &s21, &s22, &s23);

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

	_ftime64_s(&timebuffer);
	time0 = timebuffer.time + timebuffer.millitm / 1000.0;

	for (i = 1; i <= n; i++) RollSeed_MRG32k3a(dseed);

	printf("   seeds after rolling %i times on CPU: %12.1lf %12.1lf %12.1lf %12.1lf %12.1lf %12.1lf \n",
		n, dseed[0], dseed[1], dseed[2], dseed[3], dseed[4], dseed[5]);

	_ftime64_s(&timebuffer);
	time1 = timebuffer.time + timebuffer.millitm / 1000.0;

//	Reset seeds
	dseed[0] = s11;
	dseed[1] = s12;
	dseed[2] = s13;
	dseed[3] = s21;
	dseed[4] = s22;
	dseed[5] = s23;

	for (i = 1; i <= n; i++) {
		x = MRG32k3a(dseed);
	}
	printf("   new seeds after %i simulations:      %12.1lf %12.1lf %12.1lf %12.1lf %12.1lf %12.1lf \n",
		n, dseed[0], dseed[1], dseed[2], dseed[3], dseed[4], dseed[5]);

	_ftime64_s(&timebuffer);
	time2 = timebuffer.time + timebuffer.millitm / 1000.0;

	//	Calculate matrices for skip ahead with MRG32k3a
	SkipAhead_MRG32k3a(n, An1, An2);

	for (i = 0; i < 3; i++) {
		sseed1[i] = 0.0;
		sseed2[i] = 0.0;
		for (j = 0; j < 3; j++) {
			sseed1[i] += (An1[i][j] * seed1[j]) % im1;
			sseed2[i] += (An2[i][j] * seed2[j]) % im2;
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

	printf("   new seeds after skip ahead %i:     %12llu %12llu %12llu %12llu %12llu %12llu \n",
		n, sseed1[0], sseed1[1], sseed1[2], sseed2[0], sseed2[1], sseed2[2]);

	_ftime64_s(&timebuffer);
	time3 = timebuffer.time + timebuffer.millitm / 1000.0;

	time3 = time3 - time2;
	time2 = time2 - time1;
	time1 = time1 - time0;

	printf("  time1 = %7.3lf   time2 = %7.3lf   time3 = %7.3lf  \n", time1, time2, time3);

	printf(" An1: \n");
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) printf(" %12u ", An1[i][j]);
		printf("  \n");
	}
	printf(" An2: \n");
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) printf(" %12u ", An2[i][j]);
		printf("  \n");
	}

	printf(" \n  Now running time test on GPU, n = %i simulations per path and nSim = %i separate simuation paths \n",
				n, nSim);

	_ftime64_s(&timebuffer);
	time0 = timebuffer.time + timebuffer.millitm / 1000.0;

	cudaNumber = 0;
	cudaStatus = cudaSetDevice(cudaNumber);

	h_seeds = (unsigned int *)malloc((nSim*6) * sizeof(unsigned int));
	h_MRG_rng1 = (float *)malloc(nSim*sizeof(float));
	h_MRG_rng2 = (float *)malloc(nSim*sizeof(float));
	h_MRG_rng4 = (float *)malloc(nSim*sizeof(float));
	//h_LC_rng = (float *)malloc(nSim*sizeof(float));

	cudaMalloc((void **)&d_seeds, (nSim*6)*sizeof(unsigned int));
	cudaMalloc((void **)&d_MRG_rng, nSim*sizeof(float));
//	cudaMalloc((void **)&d_LC_rng, nSim*sizeof(float));

	_ftime64_s(&timebuffer);
	time1 = timebuffer.time + timebuffer.millitm / 1000.0;

//	Step ahead forseeds to be passed to each GPU thread
		h_seeds[0] = s11;
		h_seeds[1] = s12;
		h_seeds[2] = s13;
		h_seeds[3] = s21;
		h_seeds[4] = s22;
		h_seeds[5] = s23;
		for (ii = 1; ii < nSim; ii++) {
			for (i = 0; i < 3; i++) {
				seed1[i] = h_seeds[(ii-1)*6 + i];
				seed2[i] = h_seeds[(ii-1)*6 + i + 3];
			}
			for (i = 0; i < 3; i++) {
				sseed1[i] = 0.0;
				sseed2[i] = 0.0;
				for (j = 0; j < 3; j++) {
					sseed1[i] += (An1[i][j] * seed1[j]) % im1;
					sseed2[i] += (An2[i][j] * seed2[j]) % im2;
				}
				lp1 = sseed1[i];
				lp1 = lp1 % im1;
				if (lp1 < 0) lp1 += im1;
				h_seeds[ii*6+i] = lp1;
				lp2 = sseed2[i];
				lp2 = lp2 % im2;
				if (lp2 < 0) lp2 += im2;
				h_seeds[ii*6+i+3] = lp2;
			}
		}
		cudaMemcpy(d_seeds, h_seeds, nSim*6*sizeof(unsigned int), cudaMemcpyHostToDevice);

	_ftime64_s(&timebuffer);
	time2 = timebuffer.time + timebuffer.millitm / 1000.0;

	ind = 1;

	SimulatePathGPU <<<(1 + nSim / 128), 128 >>>(
		n, nSim, ind, d_seeds, d_MRG_rng);

	cudaGetLastError();
	cudaDeviceSynchronize();

	//Read GPU simulations back to host CPU 
	cudaMemcpy(h_MRG_rng1, d_MRG_rng, nSim*sizeof(float), cudaMemcpyDeviceToHost);

	_ftime64_s(&timebuffer);
	time3 = timebuffer.time + timebuffer.millitm / 1000.0;
	
	ind = 2;

	SimulatePathGPU <<<(1 + nSim / 128), 128 >>>(
		n, nSim, ind, d_seeds, d_MRG_rng);

	cudaGetLastError();
	cudaDeviceSynchronize();

	//Read GPU simulations back to host CPU 
	cudaMemcpy(h_MRG_rng2, d_MRG_rng, nSim*sizeof(float), cudaMemcpyDeviceToHost);

	_ftime64_s(&timebuffer);
	time4 = timebuffer.time + timebuffer.millitm / 1000.0;
	
	cudaMalloc((void **)&devMRGStates, nSim*sizeof(curandStateMRG32k3a));

	setup_kernel <<<(1 + nSim / 64), 64 >>>(nSim, devMRGStates);

	_ftime64_s(&timebuffer);
	time5 = timebuffer.time + timebuffer.millitm / 1000.0;

	Test_curand_MRG32k3a <<<(1 + nSim / 64), 64 >>>(
		n, nSim, d_seeds, d_MRG_rng, devMRGStates);

	cudaGetLastError();
	cudaDeviceSynchronize();

	cudaMemcpy(h_MRG_rng4, d_MRG_rng, nSim*sizeof(float), cudaMemcpyDeviceToHost);

	_ftime64_s(&timebuffer);
	time6 = timebuffer.time + timebuffer.millitm / 1000.0;

	time6 = time6 - time5;
	time5 = time5 - time4;
	time4 = time4 - time3;
	time3 = time3 - time2;
	time2 = time2 - time1;
	time1 = time1 - time0;

	printf("  Setup times: Memory allocation %7.3lf  initialize seeds for MRG on host %7.3lf  initialize curand MRG %7.3lf \n", 
					time1, time2, time5);
	printf("  Run times: MRG(unsigned int) %7.3lf  MRG(double precision) %7.3lf  MRG(curand) %7.3lf \n", 
					time3, time4, time6);
	fprintf(fout, "  Setup times: Memory allocation %7.3lf  initialize seeds for MRG on host %7.3lf  initialize curand MRG %7.3lf \n",
		time1, time2, time5);
	fprintf(fout, "  Run times: MRG(unsigned int) %7.3lf  MRG(double precision) %7.3lf  MRG(curand) %7.3lf \n",
		time3, time4, time6);

	printf("  Print first 20 rows and last 2 rows of the MRG32k3a simulations on GPU \n");
	printf("  MRG(unsigned int)  MRG(double precision)  MRG(curand) \n");
	for (i = 0; i < 20; i++) printf("   %11.8f        %11.8f            %11.8f \n", 
					h_MRG_rng1[i], h_MRG_rng2[i], h_MRG_rng4[i]);
	printf("\n   %11.8f        %11.8f            %11.8f \n", 
					h_MRG_rng1[nSim - 2], h_MRG_rng2[nSim - 2], h_MRG_rng4[nSim - 2]);
	printf("\n   %11.8f        %11.8f            %11.8f \n \n", 
					h_MRG_rng1[nSim - 1], h_MRG_rng2[nSim - 1], h_MRG_rng4[nSim - 1]);

	fprintf(fout, "  Print first 20 rows and last 2 rows of the MRG32k3a simulations on GPU \n");
	fprintf(fout, "  MRG(unsigned int)  MRG(double precision)  MRG(curand) \n");
	for (i = 0; i < 20; i++) fprintf(fout, "   %11.8f        %11.8f            %11.8f \n",
		h_MRG_rng1[i], h_MRG_rng2[i], h_MRG_rng4[i]);
	fprintf(fout, "\n   %11.8f        %11.8f            %11.8f \n",
		h_MRG_rng1[nSim - 2], h_MRG_rng2[nSim - 2], h_MRG_rng4[nSim - 2]);
	fprintf(fout, "\n   %11.8f        %11.8f            %11.8f \n \n",
		h_MRG_rng1[nSim - 1], h_MRG_rng2[nSim - 1], h_MRG_rng4[nSim - 1]);


	_ftime64_s(&timebuffer);
	time4 = timebuffer.time + timebuffer.millitm / 1000.0;

	
	free(seed);
	free(dseed);
	free_uint_matrix(An1, 0, 2, 0, 2);
	free_uint_matrix(An2, 0, 2, 0, 2);

	free(h_seeds);
	free(h_MRG_rng1);
	free(h_MRG_rng2);
	free(h_MRG_rng4);

	cudaFree(devMRGStates);
	cudaFree(d_seeds);
	cudaFree(d_MRG_rng);

	cudaDeviceReset();

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
/*
void Gen_seed(unsigned int &ib1, unsigned int &ib2, unsigned int ib3, unsigned int ib4)
{
	//	unsigned _int64 ltem1, ltem2, ltem3;
	unsigned long long ltem1, ltem2, ltem3;
	const unsigned int I2_31 = 2147483648;
	const unsigned int I2_28 = 268435456;
	const unsigned int I13_6 = 4826809;
	const unsigned int I13_7 = 62748517;

	ltem1 = ib1;
	ltem2 = ib2;
	ltem3 = ltem1*ib4;
	ltem1 = ltem1*ib3;
	ltem2 = ltem2*ib3;

	ib1 = (ltem1 % I2_31);
	ib2 = ((ltem2 + ltem3 + (ltem1 >> 31)) % I2_28);

}

double rand_u01(unsigned int &ib1, unsigned int &ib2)
{
	//	unsigned _int64 ltem1, ltem2;
	unsigned long long ltem1, ltem2;
	double answer;
	const double  fI2_28 = 3.72529029846191e-09;
	const double  fI2_59 = 1.73472347597681e-18;
	const unsigned int I13_6 = 4826809;
	const unsigned int I13_7 = 62748517;
	const unsigned int I2_28 = 268435456;
	const unsigned int I2_31 = 2147483648;

	ltem1 = ib1;
	ltem2 = ib2;
	ltem1 = ltem1*I13_7;
	ltem2 = ltem2*I13_7;
	ib1 = (ltem1 % I2_31);
	ib2 = ((ltem2 + (ltem1 >> 31)) % I2_31);

	ltem1 = ib1;
	ltem2 = ib2;
	ltem1 = ltem1*I13_6;
	ltem2 = ltem2*I13_6;
	ib1 = (ltem1 % I2_31);
	ib2 = ((ltem2 + (ltem1 >> 31)) % I2_31);
	ib2 = ib2 % I2_28;

	answer = ib2*fI2_28 + ib1*fI2_59;

	return answer;
}


void roll_seed_CPU(unsigned int &ib1, unsigned int &ib2)
{
	//	unsigned _int64 ltem1, ltem2;
	unsigned long long ltem1, ltem2;
	const unsigned int I13_6 = 4826809;
	const unsigned int I13_7 = 62748517;
	const unsigned int I2_28 = 268435456;
	const unsigned int I2_31 = 2147483648;

	ltem1 = ib1;
	ltem2 = ib2;
	ltem1 = ltem1*I13_7;
	ltem2 = ltem2*I13_7;
	ib1 = (ltem1 % I2_31);
	ib2 = ((ltem2 + (ltem1 >> 31)) % I2_31);

	ltem1 = ib1;
	ltem2 = ib2;
	ltem1 = ltem1*I13_6;
	ltem2 = ltem2*I13_6;
	ib1 = (ltem1 % I2_31);
	ib2 = ((ltem2 + (ltem1 >> 31)) % I2_31);
	ib2 = ib2 % I2_28;

}
*/
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


#undef norm
#undef m1
#undef m2
#undef a12
#undef a13n
#undef a21
#undef a23n
#undef NR_END
#undef FREE_ARG

//#undef MAX_NUM_THREADS

/* (C)Copr. 1986-92 Numerical Recipes Software G2v#X):K. */

