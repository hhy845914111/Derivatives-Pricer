//			Sim_3m_LIBOROptions_GPU_v2.cu
//
//			Function to simulate the 3 Factor Hull-White model 
//			dll to be called from Visual Basic in Excel
//			This model prices 3m LIBOR Options (European style ED Options)
//			Input number of option expirations and maximum number of strikes 
//				for each expiration
//			Input the futures rate, number of strikes, expiration, and term
//			Must calculate A and B coefficients at expiration dates
//
//			Created January 2019, by Louis Scott
//

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "stdafx.h"
#include "DynamicMemoryAllocation.h"
#include "SimulationFunctions_MRG32k3a.h"
#include "ODE_Solver.h"
#include "Sim_3m_LIBOROptions_kernel.cuh"

#define MAX_NUM_THREADS 4
#define NEQ 1
#define nFactors 3

using namespace std;

/* prototype */
unsigned __stdcall RunThread(void *param);
void SimulateOISRates(int jThr);
void derivs(double t, double y[], double dydx[]);
void derivs2(double t, double y[], double dydx[]);

/*     Global Variables     */
const unsigned int I31_max = 2147483647;
const double DaysPerYear = 365.245;
int nMat, nSteps, nSim, maxStrikes, jMat;
int *nMat1, *nMat2, *nStrikes;
unsigned int **OptType;
double dSeeds_Threads[MAX_NUM_THREADS + 1][6];
double dt, sqdt, r0, y10, y20;
double kappa[4], theta[4], sigma[4], lambda[4];
double *A, *FutRate0, *FwdSpread, **Strikes, ***LIBOROption, ***StdErr;
double *Discount, *AConst, *B0, *B1, *B2;
unsigned int **An1, **An2;

FILE *fout;

/* create thread argument struct for Run_Thread() */
typedef struct _thread_data_t {
	int tid;
	//	double stuff;
} thread_data_t;

int main()
{
	const unsigned int im1 = 4294967087;
	const unsigned int im2 = 4294944443;
	short ifail = 0;
	int i, j, k, nrerror;
	double *A;
	int nMaturities, nThreadsToUse, nSimIn, nStepsPerDay, maxStrikesIn;
	double r0_in, y10_in, y20_in;
	double tem, dSeed_Init[6];
	int neq, nok, nbad;
	double eps, hstart, t1, t2, h1, hmin;
	double C1, C2, C01, C02, *ystart;
	double Atem, B0tem, B1tem, B2tem;
	int nSimsPerPath, nThreads;
	unsigned int **Bn1, **Bn2, iseed[6];
	unsigned long long lp1, lp2, seed1[3], seed2[3], sseed1[3], sseed2[3];
	double time0, time1;
	struct _timeb timebuffer;
	HANDLE threads[MAX_NUM_THREADS];
	errno_t errcheck;
	unsigned threadID[MAX_NUM_THREADS];
	// create a thread_data_t argument array 
	thread_data_t thr_data[MAX_NUM_THREADS];

	errcheck = fopen_s(&fout, "Check_3m_LIBOR_Options.txt", "w");

	FILE *fin;
	errcheck = fopen_s(&fin, "LIBOROption_Parameters.txt", "r");
	if (errcheck) printf(" File LIBOROption_Parameters.txt not opened \n");

	fscanf_s(fin, " %i %i %i %i %i ", &nSimIn, &nThreadsToUse, &nMaturities, &maxStrikesIn, &nStepsPerDay);

	printf("  Enter the number of threads and the number of simulations per thread \n");
	printf("  Enter negative numbers to use defaults \n");
	cin >> i;
	cin >> j;
	if (i > 0) nThreadsToUse = i;
	if (j > 0) nSimIn = j;

	nMat = nMaturities;
	nThreads = nThreadsToUse;
	if (nThreads > MAX_NUM_THREADS) nThreads = MAX_NUM_THREADS;
	nSim = nSimIn;
	nSteps = nStepsPerDay;
	maxStrikes = maxStrikesIn;
	nrerror = 0;
	neq = NEQ;

	A = (double *)malloc(25 * sizeof(double));
	LIBOROption = d3tensor(1, nMat, 1, maxStrikes, 0, nThreads);
	StdErr = d3tensor(1, nMat, 1, maxStrikes, 0, nThreads);
	Strikes = matrix_fp64(nMat + 1, maxStrikes + 1);
	OptType = uint_matrix(1, nMat, 1, maxStrikes);
	FutRate0 = (double *)malloc((nMat + 1) * sizeof(double));
	nStrikes = (int *)malloc((nMat + 1) * sizeof(int));
	FwdSpread = (double *)malloc((nMat + 1) * sizeof(double));

	nMat1 = (int *)malloc((nMat + 1) * sizeof(int));
	nMat2 = (int *)malloc((nMat + 1) * sizeof(int));

	An1 = uint_matrix(0, 2, 0, 2);
	An2 = uint_matrix(0, 2, 0, 2);
	Bn1 = uint_matrix(0, 2, 0, 2);
	Bn2 = uint_matrix(0, 2, 0, 2);

	printf("  Parameters \n");
	fprintf(fout, "  Parameters  \n");
	for (i = 1; i <= 10; i++) {
		fscanf_s(fin, " %lf ", &A[i]);
		printf(" %f ", A[i]);
		fprintf(fout, " %f ", A[i]);
	}
	printf(" \n");
	fprintf(fout, " \n");

	fscanf_s(fin, " %lf %lf %lf ", &r0_in, &y10_in, &y20_in);
	r0 = r0_in;
	y10 = y10_in;
	y20 = y20_in;
	printf("   Initial values for r, y1, y2: %f %f %f \n", r0, y10, y20);

	if (nMat > nThreads) nMat = nThreads;

	printf(" %i  %i  %i \n", nSim, nStepsPerDay, nThreads);
	fprintf(fout, " %i  %i  %i \n", nSim, nStepsPerDay, nThreads);
	printf("  The input maturity days, nMat = %i \n", nMat);
	fprintf(fout, "  The input maturity nMat, nMat = %i \n", nMat);

	printf("  The initial seeds for MRG32k3a \n");
	fprintf(fout, "  The initial seeds for MRG32k3a \n");
	for (i = 1; i <= 6; i++) {
		fscanf_s(fin, " %i ", &j);
		iseed[i - 1] = j;
		printf(" %i ", iseed[i - 1]);
		fprintf(fout, " %i ", iseed[i - 1]);
	}
	printf(" \n");
	fprintf(fout, "  \n");

	for (i = 1; i <= nMat; i++) {
		fscanf_s(fin, " %lf %i %i %i ", &FutRate0[i], &nMat1[i], &nMat2[i], &nStrikes[i]);
		for (j = 1; j <= nStrikes[i]; j++) {
			fscanf_s(fin, " %lf %i ", &Strikes[i][j], &OptType[i][j]);
		}
	}

	/*
		Parameter inputs

		A[1] = kappa[0]
		A[2] = sigma[0]
		A[3] = lambda[0]
		A[4] = kappa[1]
		A[5] = theta[1]
		A[6] = sigma[1]
		A[7] = lambda[1]
		A[8] = kappa[2]
		A[9] = sigma[2]
		A[10] = lambda[2]

	*/

	kappa[0] = A[1];
	sigma[0] = A[2];
	lambda[0] = A[3];
	kappa[1] = A[4];
	theta[1] = A[5];
	sigma[1] = A[6];
	lambda[1] = A[7];
	kappa[2] = A[8];
	sigma[2] = A[9];
	lambda[2] = A[10];

	r0 = r0_in;
	y10 = y10_in;
	y20 = y20_in;

	//    End of input section

	_ftime64_s(&timebuffer);
	time0 = timebuffer.time + timebuffer.millitm / 1000.0;

	dt = 1.0 / (DaysPerYear*nSteps);
	sqdt = sqrt(dt);

	//	Calculate initial discount function for payment dates 
	//		and annuity factors for swaption expirations

	AConst = (double *)malloc((nMat + 1) * sizeof(double));
	B0 = (double *)malloc((nMat + 1) * sizeof(double));
	B1 = (double *)malloc((nMat + 1) * sizeof(double));
	B2 = (double *)malloc((nMat + 1) * sizeof(double));

	ystart = (double *)malloc((NEQ + 1) * sizeof(double));
	Discount = (double *)malloc((nMat + 1) * sizeof(double));

	//	Use position i = 0 for initial discount function	
	//	Not certain that we need initial discount function
	hstart = 0.0;
	eps = 0.00000001;
	hmin = 0.0;

	//	Compute exp. affine coefficients for 3m, using nMat2, at each maturity/expiration
	//	NMat2 days past 2 days forward 
	for (i = 1; i <= nMat; i++) {
		ystart[1] = 0.0;
		t1 = 0.0;
		t2 = 2.0 / DaysPerYear;
		h1 = 0.5*(t2 - t1);
		odeint(ystart, neq, t1, t2, eps, h1, hmin, &nok, &nbad, &nrerror, derivs, rkqs);
		if (nrerror < 0) {
			ifail = -10 + nrerror;
			return ifail;
		}
		Atem = ystart[1];
		//	Need to calculate B coefficients for 2 day rate 
		tem = kappa[0] * t2;
		if (fabs(tem) < 1.0e-06) B0tem = t2 * (1.0 - 0.5*tem + tem * tem / 6.0 - tem * tem*tem / 24.0);
		else B0tem = (1.0 - exp(-tem)) / kappa[0];
		tem = kappa[1] * t2;
		if (fabs(tem) < 1.0e-06) C1 = t2 * (1.0 - 0.5*tem + tem * tem / 6.0 - tem * tem*tem / 24.0);
		else C1 = (1.0 - exp(-tem)) / kappa[1];
		tem = kappa[2] * t2;
		if (fabs(tem) < 1.0e-06) C2 = t2 * (1.0 - 0.5*tem + tem * tem / 6.0 - tem * tem*tem / 24.0);
		else C2 = (1.0 - exp(-tem)) / kappa[2];

		tem = (kappa[0] - kappa[1]) * t2;
		if (fabs(tem) < 1.0e-06) C01 = t2 * (1.0 - 0.5*tem + tem * tem / 6.0 - tem * tem*tem / 24.0);
		else C01 = (1.0 - exp(-tem)) / (kappa[0] - kappa[1]);
		tem = (kappa[0] - kappa[2]) * t2;
		if (fabs(tem) < 1.0e-06) C02 = t2 * (1.0 - 0.5*tem + tem * tem / 6.0 - tem * tem*tem / 24.0);
		else C02 = (1.0 - exp(-tem)) / (kappa[0] - kappa[2]);

		B1tem = C1 - exp(-kappa[1] * t2)*C01;
		B2tem = C2 - exp(-kappa[2] * t2)*C02;

//		Discount[i] = exp(-(AConst[0][i] + B0[0][i] * r0 + B1[0][i] * y10 + B2[0][i] * y20));
		t1 = t2;
		t2 = (2 + nMat2[i]) / DaysPerYear;
		h1 = 0.5*(t2 - t1);
		odeint(ystart, neq, t1, t2, eps, h1, hmin, &nok, &nbad, &nrerror, derivs, rkqs);
		if (nrerror < 0) {
			ifail = -10 + nrerror;
			return ifail;
		}
		AConst[i] = ystart[1] - Atem;

		//	Need to calculate B coefficients 
		tem = kappa[0] * t2;
		if (fabs(tem) < 1.0e-06) B0[i] = t2 * (1.0 - 0.5*tem + tem * tem / 6.0 - tem * tem*tem / 24.0);
		else B0[i] = (1.0 - exp(-tem)) / kappa[0];
		tem = kappa[1] * t2;
		if (fabs(tem) < 1.0e-06) C1 = t2 * (1.0 - 0.5*tem + tem * tem / 6.0 - tem * tem*tem / 24.0);
		else C1 = (1.0 - exp(-tem)) / kappa[1];
		tem = kappa[2] * t2;
		if (fabs(tem) < 1.0e-06) C2 = t2 * (1.0 - 0.5*tem + tem * tem / 6.0 - tem * tem*tem / 24.0);
		else C2 = (1.0 - exp(-tem)) / kappa[2];

		tem = (kappa[0] - kappa[1]) * t2;
		if (fabs(tem) < 1.0e-06) C01 = t2 * (1.0 - 0.5*tem + tem * tem / 6.0 - tem * tem*tem / 24.0);
		else C01 = (1.0 - exp(-tem)) / (kappa[0] - kappa[1]);
		tem = (kappa[0] - kappa[2]) * t2;
		if (fabs(tem) < 1.0e-06) C02 = t2 * (1.0 - 0.5*tem + tem * tem / 6.0 - tem * tem*tem / 24.0);
		else C02 = (1.0 - exp(-tem)) / (kappa[0] - kappa[2]);

		B1[i] = C1 - exp(-kappa[1] * t2)*C01;
		B2[i] = C2 - exp(-kappa[2] * t2)*C02;

		B0[i] = B0[i] - B0tem;
		B1[i] = B1[i] - B1tem;
		B2[i] = B2[i] - B2tem;

//		Discount[i] = exp(-(AConst[i] + B0[i] * r0 + B1[i] * y10 + B2[i] * y20));
	}

	//	Use previous coefficients and calculate MGF at each expiration date, tehn calculate spread between 
	//	3m LIBOR futures and 3m OIS futures (hypothetical) -> FwdSpread[i], calculated for annualized simple interest rate

	for (i = 1; i <= nMat; i++) {
		jMat = i;
		ystart[1] = 0.0;
		t1 = 0.0;
		t2 = nMat1[i] / DaysPerYear;
		h1 = 0.5*(t2 - t1);
		odeint(ystart, neq, t1, t2, eps, h1, hmin, &nok, &nbad, &nrerror, derivs2, rkqs);
		if (nrerror < 0) {
			ifail = -10 + nrerror;
			return ifail;
		}
		Atem = ystart[1];
		//	Need to calculate B coefficients for 2 day rate 
		B0tem = exp(-kappa[0] * t2)*B0[i];

		tem = (kappa[0] - kappa[1]) * t2;
		if (fabs(tem) < 1.0e-06) C01 = t2 * (1.0 - 0.5*tem + tem * tem / 6.0 - tem * tem*tem / 24.0);
		else C01 = (1.0 - exp(-tem)) / (kappa[0] - kappa[1]);
		tem = (kappa[0] - kappa[2]) * t2;
		if (fabs(tem) < 1.0e-06) C02 = t2 * (1.0 - 0.5*tem + tem * tem / 6.0 - tem * tem*tem / 24.0);
		else C02 = (1.0 - exp(-tem)) / (kappa[0] - kappa[2]);

		C1 = exp(-kappa[1] * t2);
		C2 = exp(-kappa[2] * t2);

		B1tem = C1 * (B1[i] + kappa[0] * B0[i] * C01);
		B2tem = C2 * (B2[i] + kappa[0] * B0[i] * C02);

		//	Now calcuate spread between 3m LIBOR futures and 3m OIS futures
		//	place the 3m OIS futures into tem
		tem = AConst[i] + Atem + B0tem * r0 + B1tem * y10 + B2tem * y20;
		tem = (exp(tem) - 1.0)*360.0 / nMat2[i];
		FwdSpread[i] = FutRate0[i] - tem;
		fprintf(fout, "  3m LIBOR - 3m OIS Futures Rate Spread %f     3m OIS Futures  %f  \n", FwdSpread[i], tem);
		fprintf(fout, "  OIS Futures Coeff  %f %f %f %f %f \n", AConst[i], Atem, B0tem, B1tem, B2tem);
	}

	//	Check and initialize seeds for each thread
	for (i = 0; i < 6; i++) dSeed_Init[i] = iseed[i];
	//	roll seeds 3 times for initiaization
	for (i = 1; i <= 3; i++) roll_seedCPU(dSeed_Init);
	for (i = 0; i < 3; i++) {
		seed1[i] = dSeed_Init[i];
		seed2[i] = dSeed_Init[i + 3];
	}
	for (i = 0; i < 6; i++) dSeeds_Threads[1][i] = dSeed_Init[i];

	printf("  The starting seeds for MRG32k3a \n");
	fprintf(fout, "  The starting seeds for MRG32k3a \n");
	for (i = 0; i < 6; i++) {
		printf(" %f ", dSeed_Init[i]);
		fprintf(fout, " %f ", dSeed_Init[i]);
	}
	printf(" \n");
	fprintf(fout, "  \n");

	nSimsPerPath = nMat1[nMat] * (nSteps * 3);
	SkipAhead_MRG32k3a(nSimsPerPath, An1, An2);
	SkipAhead2_MRG32k3a(nSim, An1, An2, Bn1, Bn2);

	if (nThreads > 1) {

		for (k = 1; k < nThreads; k++) {
			for (i = 0; i < 3; i++) {
				seed1[i] = dSeeds_Threads[k][i];
				seed2[i] = dSeeds_Threads[k][i + 3];
			}
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
			for (i = 0; i < 3; i++) {
				dSeeds_Threads[k + 1][i] = sseed1[i];
				dSeeds_Threads[k + 1][i + 3] = sseed2[i];
			}
		}

	}		//	end of if nThreads > 1

	//	Set up multi-threading here
	if (nThreads == 1) SimulateOISRates(0);
	else {
		for (i = 0; i < nThreads; i++) {
			thr_data[i].tid = i;
			threads[i] = (HANDLE)_beginthreadex(NULL, 0, RunThread, &thr_data[i], 0, &threadID[i]);
		}
		WaitForMultipleObjects(nThreads, threads, TRUE, INFINITE);
		for (i = 0; i < nThreads; i++) CloseHandle(threads[i]);
	}

	//	Average across the threads

	fprintf(fout, "  Completed simulations.  Now averaging across threads \n");

	for (k = 1; k <= nMat; k++) {
		for (i = 1; i <= nStrikes[k]; i++) {
			LIBOROption[k][i][0] = 0.0;
			StdErr[k][i][0] = 0.0;
			for (j = 1; j <= nThreads; j++) {
				LIBOROption[k][i][0] = LIBOROption[k][i][0] + LIBOROption[k][i][j];
				StdErr[k][i][0] = StdErr[k][i][0] + StdErr[k][i][j];
			}
			LIBOROption[k][i][0] = LIBOROption[k][i][0] / (nThreads*nSim);
			StdErr[k][i][0] = StdErr[k][i][0] / (nThreads*nSim) - LIBOROption[k][i][0] * LIBOROption[k][i][0];
			StdErr[k][i][0] = sqrt(StdErr[k][i][0] / (nThreads*nSim));
		}
	}

	_ftime64_s(&timebuffer);
	time1 = timebuffer.time + timebuffer.millitm / 1000.0;
	time1 = time1 - time0;

	//   Print results 
	printf("  European Options Prices (in percent) from Monte Carlo Simulation using %i simulations across %i threads \n", nSim*nThreads, nThreads);
	for (i = 1; i <= nMat; i++) {
		for (j = 1; j <= nStrikes[i]; j++) {
			printf("  %i  %10.8f  %10.8f   %10.8f \n", OptType[i][j], Strikes[i][j] * 100, LIBOROption[i][j][0] * 100, StdErr[i][j][0] * 100);
		}
	}

	printf("  Compute time %f \n", time1);
	fprintf(fout, "  Compute time %f \n", time1);

	fclose(fout);

	//   free the work arrays

	free(A);
	free(FutRate0);
	free(nStrikes);
	free(nMat1);
	free(nMat2);
	free(FwdSpread);
	free(ystart);
	free(Discount);
	free_matrix_fp64(Strikes);
	free_uint_matrix(OptType, 1, nMat, 1, maxStrikes);
	free(AConst);
	free(B0);
	free(B1);
	free(B2);
	free_d3tensor(LIBOROption, 1, nMat, 1, maxStrikes, 0, nThreads);
	free_d3tensor(StdErr, 1, nMat, 1, maxStrikes, 0, nThreads);
	free_uint_matrix(An1, 0, 2, 0, 2);
	free_uint_matrix(An2, 0, 2, 0, 2);
	free_uint_matrix(Bn1, 0, 2, 0, 2);
	free_uint_matrix(Bn2, 0, 2, 0, 2);

	return 0;
}               //    End of Sim_3m_LIBOROptions_GPU_v2

unsigned __stdcall RunThread(void *param)
{
	int iThread;
	thread_data_t *data = (thread_data_t *)param;
	iThread = data->tid;
	SimulateOISRates(iThread);

	return 1;
}

void SimulateOISRates(int iThread)
{
	const unsigned int im1 = 4294967087;
	const unsigned int im2 = 4294944443;
	int i, j, k, kk, jk, jThr, jstart, jend, cudaNumber;
	double dseed[6], y1rho, tem, tem1, temopt;
	double **sum, **sumStdErr;
	unsigned long long lp1, lp2, seed1[3], seed2[3], sseed1[3], sseed2[3];
	unsigned int *h_seeds;
	int *h_nMat1, *h_nMat2;
	float *h_FwdSpread, *h_AConst, *h_B0, *h_B1, *h_B2;
	float h_r0, h_y10, h_y20;
	float y1const, *h_sigz, *h_lamsig, *h_temexp;
	double temkappa[3], tem2kappa[3], sigz[3], lamsig[3], temexp[3];
	double *h_SimDiscount, *h_SimLIBOR;
	int *d_nMat1, *d_nMat2;
	float *d_FwdSpread, *d_AConst, *d_B0, *d_B1, *d_B2;
	unsigned int *d_seeds;
	float *d_sigz, *d_lamsig, *d_temexp;
	double *d_SimDiscount, *d_SimLIBOR;
	cudaError_t cudaStatus;

	cudaNumber = 0;
	jThr = iThread + 1;

	printf("   Now running simulation for thread %i \n", jThr);
//	fprintf(fout, "   Now running simulation for thread %i \n", jThr);

	cudaStatus = cudaSetDevice(cudaNumber);

	sum = matrix_fp64(nMat + 1, maxStrikes + 1);
	sumStdErr = matrix_fp64(nMat + 1, maxStrikes + 1);

	for (k = 1; k <= nMat; k++) {
		for (j = 1; j <= nStrikes[k]; j++) {
			sum[k][j] = 0.0;
			sumStdErr[k][j] = 0.0;
		}
	}

	h_seeds = (unsigned int *)malloc(6 * nSim * sizeof(unsigned int));

	h_r0 = r0;
	h_y10 = y10;
	h_y20 = y20;
	h_nMat1 = (int *)malloc(nMat * sizeof(int));
	h_nMat2 = (int *)malloc(nMat * sizeof(int));

	h_FwdSpread = (float *)malloc(nMat * sizeof(float));
	h_AConst = (float *)malloc(nMat * sizeof(float));
	h_B0 = (float *)malloc(nMat * sizeof(float));
	h_B1 = (float *)malloc(nMat * sizeof(float));
	h_B2 = (float *)malloc(nMat * sizeof(float));

	h_sigz = (float *)malloc(3 * sizeof(float));
	h_lamsig = (float *)malloc(3 * sizeof(float));
	h_temexp = (float *)malloc(3 * sizeof(float));
	h_SimDiscount = (double *)malloc(nMat*nSim*sizeof(double));
	h_SimLIBOR = (double *)malloc(nMat*nSim * sizeof(double));

	cudaMalloc((void **)&d_seeds, 6 * nSim * sizeof(unsigned int));
	cudaMalloc((void **)&d_nMat1, nMat * sizeof(int));
	cudaMalloc((void **)&d_nMat2, nMat * sizeof(int));

	cudaMalloc((void **)&d_FwdSpread, nMat * sizeof(float));
	cudaMalloc((void **)&d_AConst, nMat * sizeof(float));
	cudaMalloc((void **)&d_B0, nMat * sizeof(float));
	cudaMalloc((void **)&d_B1, nMat * sizeof(float));
	cudaMalloc((void **)&d_B2, nMat * sizeof(float));

	cudaMalloc((void **)&d_sigz, 3 * sizeof(float));
	cudaMalloc((void **)&d_lamsig, 3 * sizeof(float));
	cudaMalloc((void **)&d_temexp, 3 * sizeof(float));
	cudaMalloc((void **)&d_SimDiscount, nMat*nSim*sizeof(double));
	cudaMalloc((void **)&d_SimLIBOR, nMat*nSim * sizeof(double));

//	set seeds for the start of each path; this requires the most time
	for (i = 0; i < 6; i++) h_seeds[i] = dSeeds_Threads[jThr][i];
	for (i = 0; i < 3; i++) {
		seed1[i] = dSeeds_Threads[jThr][i];
		seed2[i] = dSeeds_Threads[jThr][i + 3];
	}
	for (k = 1; k < nSim; k++) {
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

		for (i = 0; i < 3; i++) {
			h_seeds[i + k * 6] = sseed1[i];
			h_seeds[i + 3 + k * 6] = sseed2[i];
		}

		for (i = 0; i < 3; i++) {
			seed1[i] = sseed1[i];
			seed2[i] = sseed2[i];
		}
	}

//	Move values to GPU device arrays
	cudaMemcpy(d_seeds, h_seeds, 6 * nSim * sizeof(unsigned int), cudaMemcpyHostToDevice);

	for (j = 0; j < 3; j++) {
		temexp[j] = exp(-kappa[j] * dt);
		tem = kappa[j] * dt;
		if (fabs(tem) < 1.0e-06) temkappa[j] = 1.0 - 0.5*tem + tem*tem / 6.0 - tem*tem*tem / 24.0 + tem*tem*tem*tem / 120.0;
		else temkappa[j] = (1.0 - exp(-kappa[j] * dt)) / tem;
		tem = 2.0*kappa[j]*dt;
		if (fabs(tem) < 1.0e-06) tem2kappa[j] = 1.0 - 0.5*tem + tem*tem / 6.0 - tem*tem*tem / 24.0 + tem*tem*tem*tem / 120.0;
		else tem2kappa[j] = (1.0 - temexp[j]*temexp[j]) / tem;
		lamsig[j] = lambda[j] * sigma[j] * sigma[j] * dt*temkappa[j];
		sigz[j] = sigma[j] * sqrt(dt*tem2kappa[j]);
	}

	for (i = 0;i < 3;i++) {
		h_sigz[i] = sigz[i];
		h_lamsig[i] = lamsig[i];
		h_temexp[i] = temexp[i];
	}

//	Pass y1const
	y1const = theta[1] * (1.0 - temexp[1]);

	for (i = 0;i < nMat;i++) {
		h_nMat1[i] = nMat1[i + 1];
		h_nMat2[i] = nMat2[i + 1];
	}

	for (i = 0;i < nMat;i++) {
		h_FwdSpread[i] = FwdSpread[i+1];
	}
	for (j = 0;j < nMat;j++) {
		h_AConst[j] = AConst[j + 1];
		h_B0[j] = B0[j + 1];
		h_B1[j] = B1[j + 1];
		h_B2[j] = B2[j + 1];
	}

	cudaMemcpy(d_nMat1, h_nMat1, nMat * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nMat2, h_nMat2, nMat * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_FwdSpread, h_FwdSpread, nMat * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_AConst, h_AConst, nMat * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B0, h_B0, nMat * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B1, h_B1, nMat * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B2, h_B2, nMat * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sigz, h_sigz, 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_lamsig, h_lamsig, 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_temexp, h_temexp, 3 * sizeof(float), cudaMemcpyHostToDevice);

	SimulatePathGPU <<<(1 + nSim / 128), 128 >>> (
		nSim, nMat, nSteps, dt, y1const, h_r0, h_y10, h_y20, d_nMat1, 
		d_nMat2, d_seeds, d_AConst, d_B0, d_B1, d_B2, d_FwdSpread,  
		d_temexp, d_lamsig, d_sigz,d_SimDiscount, d_SimLIBOR);

	cudaGetLastError();
	cudaDeviceSynchronize();

//Read GPU simulations back to host CPU 
	cudaMemcpy(h_SimDiscount, d_SimDiscount, nMat*nSim*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_SimLIBOR, d_SimLIBOR, nMat*nSim * sizeof(double), cudaMemcpyDeviceToHost);

//	Compute option payoffs fromsimulated discounts and simulated LIBOR's
	for (i = 0; i < nSim; i++) {
		for (j = 0; j < nMat; j++) {
			tem1 = h_SimDiscount[i*nMat + j] ;
			tem = h_SimLIBOR[i*nMat + j] ;
//	Calculate discounted option payoffs
			for (k = 1; k <= nStrikes[j+1]; k++) {
				if (OptType[j+1][k] == 1) temopt = max(0.0, tem - Strikes[j+1][k]);
				else temopt = max(0.0, Strikes[j+1][k] - tem);
				sum[j+1][k] += tem1 * temopt;
				sumStdErr[j+1][k] += tem1 * tem1*temopt*temopt;
			}
		}		//	End of loop on j for nMat maturities
	}                //     end of loop on i for independent simulations

	for (k = 1; k <= nMat; k++) {
		for (j = 1; j <= nStrikes[k]; j++) {
			LIBOROption[k][j][jThr] = sum[k][j];
			StdErr[k][j][jThr] = sumStdErr[k][j];
		}
	}

//		Release memory allocation
	free_matrix_fp64(sum);
	free_matrix_fp64(sumStdErr);
	free(h_seeds);
	free(h_nMat1);
	free(h_nMat2);
	free(h_FwdSpread);
	free(h_AConst);
	free(h_B0);
	free(h_B1);
	free(h_B2);
	free(h_sigz);
	free(h_lamsig);
	free(h_temexp);
	free(h_SimDiscount);
	free(h_SimLIBOR);

	cudaFree(d_seeds);
	cudaFree(d_nMat1);
	cudaFree(d_nMat2);	
	cudaFree(d_FwdSpread);
	cudaFree(d_AConst);
	cudaFree(d_B0);
	cudaFree(d_B1);
	cudaFree(d_B2);
	cudaFree(d_sigz);
	cudaFree(d_lamsig);
	cudaFree(d_temexp);
	cudaFree(d_SimDiscount);
	cudaFree(d_SimLIBOR);

	cudaDeviceReset();

	printf("   Finished with simulation function and GPU \n");

}


/*     The function for computing the derivatives    */
void derivs(double t, double y[], double dydx[])
{
	double  B0tem, B1tem, B2tem, tem, C1, C2, C01, C02;

	//	No Need to calculate B1 
	tem = kappa[0] * t;
	if (fabs(tem) < 1.0e-06) B0tem = t * (1.0 - 0.5*tem + tem * tem / 6.0 - tem * tem*tem / 24.0);
	else B0tem = (1.0 - exp(-tem)) / kappa[0];

	tem = kappa[1] * t;
	if (fabs(tem) < 1.0e-06) C1 = t * (1.0 - 0.5*tem + tem * tem / 6.0 - tem * tem*tem / 24.0);
	else C1 = (1.0 - exp(-tem)) / kappa[1];
	tem = kappa[2] * t;
	if (fabs(tem) < 1.0e-06) C2 = t * (1.0 - 0.5*tem + tem * tem / 6.0 - tem * tem*tem / 24.0);
	else C2 = (1.0 - exp(-tem)) / kappa[2];

	tem = (kappa[0] - kappa[1]) * t;
	if (fabs(tem) < 1.0e-06) C01 = t * (1.0 - 0.5*tem + tem * tem / 6.0 - tem * tem*tem / 24.0);
	else C01 = (1.0 - exp(-tem)) / (kappa[0] - kappa[1]);
	tem = (kappa[0] - kappa[2]) * t;
	if (fabs(tem) < 1.0e-06) C02 = t * (1.0 - 0.5*tem + tem * tem / 6.0 - tem * tem*tem / 24.0);
	else C02 = (1.0 - exp(-tem)) / (kappa[0] - kappa[2]);

	B1tem = C1 - exp(-kappa[1] * t)*C01;
	B2tem = C2 - exp(-kappa[2] * t)*C02;

	dydx[1] = kappa[1] * theta[1] * B1tem - 0.5*(sigma[0] * sigma[0] * B0tem*B0tem + sigma[1] * sigma[1] * B1tem*B1tem + sigma[2] * sigma[2] * B2tem*B2tem)
		- (lambda[0] * sigma[0] * sigma[0] * B0tem + lambda[1] * sigma[1] * sigma[1] * B1tem + lambda[2] * sigma[2] * sigma[2] * B2tem);

}

void derivs2(double t, double y[], double dydx[])
{
	double  B0tem, B1tem, B2tem, tem, C1, C2, C01, C02;

	B0tem = exp(-kappa[0] * t)*B0[jMat];

	tem = (kappa[0] - kappa[1]) * t;
	if (fabs(tem) < 1.0e-06) C01 = t * (1.0 - 0.5*tem + tem * tem / 6.0 - tem * tem*tem / 24.0);
	else C01 = (1.0 - exp(-tem)) / (kappa[0] - kappa[1]);
	tem = (kappa[0] - kappa[2]) * t;
	if (fabs(tem) < 1.0e-06) C02 = t * (1.0 - 0.5*tem + tem * tem / 6.0 - tem * tem*tem / 24.0);
	else C02 = (1.0 - exp(-tem)) / (kappa[0] - kappa[2]);

	C1 = exp(-kappa[1] * t);
	C2 = exp(-kappa[2] * t);

	B1tem = C1 * (B1[jMat] + kappa[0] * B0[jMat] * C01);
	B2tem = C2 * (B2[jMat] + kappa[0] * B0[jMat] * C02);

	dydx[1] = kappa[1] * theta[1] * B1tem - 0.5*(sigma[0] * sigma[0] * B0tem*B0tem + sigma[1] * sigma[1] * B1tem*B1tem + sigma[2] * sigma[2] * B2tem*B2tem)
		- (lambda[0] * sigma[0] * sigma[0] * B0tem + lambda[1] * sigma[1] * sigma[1] * B1tem + lambda[2] * sigma[2] * sigma[2] * B2tem);

}

/* (C) Copr. 1986-92 Numerical Recipes Software G2v#X):K. */

#undef MAXSTP
#undef TINY
#undef SAFETY
#undef PGROW
#undef PSHRNK
#undef ERRCON

#undef NR_END
#undef FREE_ARG
#undef MAX_NUM_THREADS 
#undef NEQ
#undef nFactors