#include <iostream>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include <sys/timeb.h>
//#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <windows.h>
#include <wchar.h>
#include <locale>
#include <process.h>
#include <stdio.h> 
#include <malloc.h> 
#include <iostream>
#include <fstream>

#include "DynamicMemoryAllocation.h"
#include "SimulationFunctions_MRG32k3a.h"
#include "FX_SV_Options_kernel.cuh"

#define MAX_NUM_THREADS 4

using namespace std;

#define DLLEXPORT extern "C" __declspec(dllexport)

/* define functions */
unsigned __stdcall RunThread(void *param);
void SimulateFX(int jThr);

/*     Global Variables     */
//const unsigned int I31_max = 2147483647;
const double DaysPerYear = 365.245;

int nMat, nSteps, nSim, nStrikes;
int *Maturity, **OptType;
unsigned int **An1, **An2;
double dSeeds_Threads[MAX_NUM_THREADS][6];
double dt, sqdt, FX0, v0, yinit;
double kappa, theta, sigma, lambda, rho;
double **FXOptPrice, **FXStdErr, *rd, *rf, **Strikes;

FILE *fout;

/* create thread argument struct for Run_Thread() */
typedef struct _thread_data_t {
	int tid;
	//	double stuff;
} thread_data_t;

//	Note:  you need r_dom and either r_for or fwdFX
//	Removed fwdFX
DLLEXPORT void FX_SV_Model_GPU(int nThreadsIn, int nSimIn, int nTimeStepsIn, int nExpIn,
	int nStrikesPerExp, double spotFX, double spotV, int *daysExp, double *r_dom,
	double *r_for, double *params, int *OptTypeIn, double *StrikesIn,
	double *OptPriceOut, double *StdErrOut)
{
	const unsigned int im1 = 4294967087;
	const unsigned int im2 = 4294944443;
	int i, j, k, iThr;
	double dSeed_Init[6];
	int nSimsPerPath, nThreads;
	unsigned int **Bn1, **Bn2, iseed[6];
	unsigned long long lp1, lp2, seed1[3], seed2[3], sseed1[3], sseed2[3];
	HANDLE threads[MAX_NUM_THREADS];
	errno_t errcheck;
	unsigned threadID[MAX_NUM_THREADS];
	// create a thread_data_t argument array 
	thread_data_t thr_data[MAX_NUM_THREADS];

//	errcheck = fopen_s(&fout, "Check_FX_SV_Model_Simulations.txt", "w");

	nThreads = nThreadsIn;
	nSim = nSimIn;
	nSteps = nTimeStepsIn;
	nMat = nExpIn;
	nStrikes = nStrikesPerExp;
	FX0 = spotFX;
	v0 = spotV;
	yinit = log(v0*v0);
	if (nThreads > MAX_NUM_THREADS) nThreads = MAX_NUM_THREADS;
	//	take iseed from inuts
	iseed[0] = 23949459;
	iseed[1] = 948758;
	iseed[2] = 505938;
	iseed[3] = 23949459;
	iseed[4] = 948758;
	iseed[5] = 505938;

	//fprintf(fout, "  spot %f  vol %f \n", FX0, v0);
	//fprintf(fout, " nThreads %i nSim %i nSteps %i nMat %i nStrikes %i \n", nThreads, nSim, nSteps, nMat, nStrikes);

	Maturity = (int *)malloc((nMat + 1) * sizeof(int));
	rd = (double *)malloc((daysExp[nMat] + 1) * sizeof(double));
	rf = (double *)malloc((daysExp[nMat] + 1) * sizeof(double));

	OptType = int_matrix(0, nMat, 0, nStrikes);
	FXOptPrice = matrix_fp64((nMat + 1)*(nStrikes + 1), nThreads + 1);
	FXStdErr = matrix_fp64((nMat + 1)*(nStrikes + 1), nThreads + 1);
	Strikes = matrix_fp64(nMat + 1, nStrikes + 1);
	An1 = uint_matrix(0, 2, 0, 2);
	An2 = uint_matrix(0, 2, 0, 2);
	Bn1 = uint_matrix(0, 2, 0, 2);
	Bn2 = uint_matrix(0, 2, 0, 2);

	kappa = params[1];
	theta = params[2];
	sigma = params[3];
	lambda = params[4];
	rho = params[5];

	dt = 1.0 / (DaysPerYear*nSteps);
	sqdt = sqrt(dt);

		//	Move remaining inuts to global memory
	for (i = 1; i <= nMat; i++) {
		Maturity[i] = daysExp[i];
		for (j = 1; j <= nStrikes; j++) {
			k = i * (nStrikes + 1) + j;
			OptType[i][j] = OptTypeIn[k];
			Strikes[i][j] = StrikesIn[k];
		}
	}
	for (i = 1; i <= Maturity[nMat]; i++) {
		rd[i] = r_dom[i];
		rf[i] = r_for[i];
	}

	//	Check and initialize seeds for each thread
	for (i = 0; i < 6; i++) dSeed_Init[i] = iseed[i];
	//	roll seeds 3 times for initiaization
	for (i = 1; i <= 3; i++) roll_seed(dSeed_Init);
	for (i = 0; i < 3; i++) {
		seed1[i] = dSeed_Init[i];
		seed2[i] = dSeed_Init[i + 3];
	}
	for (i = 0; i < 6; i++) dSeeds_Threads[0][i] = dSeed_Init[i];

	if (nThreads > 1) {
		nSimsPerPath = Maturity[nMat] * nSteps * 2;
		SkipAhead_MRG32k3a(nSimsPerPath, An1, An2);
		SkipAhead2_MRG32k3a(nSim, An1, An2, Bn1, Bn2);
		for (k = 1; k < nThreads; k++) {
			for (i = 0; i < 3; i++) {
				seed1[i] = dSeeds_Threads[k - 1][i];
				seed2[i] = dSeeds_Threads[k - 1][i + 3];
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
				dSeeds_Threads[k][i] = sseed1[i];
				dSeeds_Threads[k][i + 3] = sseed2[i];
			}
		}
	}		//	end of if nThreads > 1

//	Set up multi-threading here

	if (nThreads == 1) SimulateFX(0);
	else {
		for (i = 0; i < nThreads; i++) {
			thr_data[i].tid = i;
			threads[i] = (HANDLE)_beginthreadex(NULL, 0, RunThread, &thr_data[i], 0, &threadID[i]);
		}
		WaitForMultipleObjects(nThreads, threads, TRUE, INFINITE);
		for (i = 0; i < nThreads; i++) CloseHandle(threads[i]);
	}

	//	Average across the threads

	OptPriceOut[0] = 10.0;

	for (i = 1; i <= nMat; i++) {
		for (j = 1; j <= nStrikes; j++) {
			k = i * (nStrikes + 1) + j;
			FXOptPrice[k][0] = 0.0;
			FXStdErr[k][0] = 0.0;
			for (iThr = 1; iThr <= nThreads; iThr++) {
				FXOptPrice[k][0] = FXOptPrice[k][0] + FXOptPrice[k][iThr];
				FXStdErr[k][0] = FXStdErr[k][0] + FXStdErr[k][iThr];
			}
			FXOptPrice[k][0] = FXOptPrice[k][0] / (nThreads*nSim);
			FXStdErr[k][0] = FXStdErr[k][0] / (nThreads*nSim) - FXOptPrice[k][0] * FXOptPrice[k][0];
			FXStdErr[k][0] = sqrt(FXStdErr[k][0] / (nThreads*nSim));
			OptPriceOut[k] = FXOptPrice[k][0];
			StdErrOut[k] = FXStdErr[k][0];
		}
	}

//	fclose(fout);

	//   free the work arrays
	free(Maturity);
	free(rd);
	free(rf);

	free_int_matrix(OptType, 0, nMat, 0, nStrikes);
	free_matrix_fp64(FXOptPrice);
	free_matrix_fp64(FXStdErr);
	free_matrix_fp64(Strikes);

	free_uint_matrix(An1, 0, 2, 0, 2);
	free_uint_matrix(An2, 0, 2, 0, 2);
	free_uint_matrix(Bn1, 0, 2, 0, 2);
	free_uint_matrix(Bn2, 0, 2, 0, 2);

}


unsigned __stdcall RunThread(void *param)
{
	int iThread;
	thread_data_t *data = (thread_data_t *)param;
	iThread = data->tid;
	SimulateFX(iThread);

	return 1;
}

void SimulateFX(int iThread)
{
	const unsigned int im1 = 4294967087;
	const unsigned int im2 = 4294944443;
	int i, j, k, iT, jj, kk, jThr, jstart, jend, cudaNumber, nMat2;
	double dseed[6];
	double logFX0, FX, x, y, v, BCoef, ACoef, sigCoef;
	double RInt, tem, tem1, cfexp, ynew, yavg, vnew, vavg;
	double **sum, **SStdErr, *Discount;
	unsigned int *h_seeds;

	float *h_SimFX, *h_rd, *h_rf ;
	unsigned long long lp1, lp2, seed1[3], seed2[3], sseed1[3], sseed2[3];
	float h_FX0, h_v0, h_dt, h_sqdt;
	float h_BCoef, float h_ACoef, float h_sigCoef, h_rho;
	int *d_Maturity;
	unsigned int *d_seeds;
	float *d_SimFX, *d_rd, *d_rf;

	cudaError_t cudaStatus;

	sum = matrix_fp64((nMat + 1), (nStrikes + 1));
	SStdErr = matrix_fp64((nMat + 1), (nStrikes + 1));
	Discount = (double *)malloc((size_t)((nMat + 1) * sizeof(double)));

	jThr = iThread + 1;

	cudaNumber = 0;
	cudaStatus = cudaSetDevice(cudaNumber);

	nMat2 = Maturity[nMat];

	h_seeds = (unsigned int *)malloc(6 * nSim * sizeof(unsigned int));

	h_FX0 = FX0;
	h_v0 = v0;
	h_rho = rho;
	h_rd = (float *)malloc((nMat2 + 1) * sizeof(float));
	h_rf = (float *)malloc((nMat2 + 1) * sizeof(float));
	h_SimFX = (float *)malloc((nMat+1)*nSim * sizeof(float));

	cudaMalloc((void **)&d_seeds, 6 * nSim * sizeof(unsigned int));
	cudaMalloc((void **)&d_Maturity, (nMat+1) * sizeof(int));
	cudaMalloc((void **)&d_SimFX, (nMat+1)*nSim * sizeof(float));
	cudaMalloc((void **)&d_rd, (nMat2 + 1) * sizeof(float));
	cudaMalloc((void **)&d_rf, (nMat2 + 1) * sizeof(float));

	//	set seeds for the start of each path; this requires the most time
	for (i = 0; i < 6; i++) h_seeds[i] = dSeeds_Threads[iThread][i];
	for (i = 0; i < 3; i++) {
		seed1[i] = dSeeds_Threads[iThread][i];
		seed2[i] = dSeeds_Threads[iThread][i + 3];
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

	tem = kappa * dt;
	BCoef = exp(-tem);
	if (fabs(tem) < 1.0e-06) cfexp = dt * (1.0 - 0.5*tem + tem * tem / 6.0 - tem * tem*tem / 24.0 + tem * tem*tem*tem / 120.0);
	else cfexp = (1.0 - BCoef) / kappa;
	ACoef = theta * (1 - BCoef) - lambda * sigma*sigma*cfexp;
	tem = 2.0*kappa*dt;
	if (fabs(tem) < 1.0e-06) tem1 = dt * (1.0 - 0.5*tem + tem * tem / 6.0 - tem * tem*tem / 24.0 + tem * tem*tem*tem / 120.0);
	else tem1 = (1.0 - exp(-tem)) / (2.0*kappa);
	sigCoef = sigma * sqrt(tem1);

	h_BCoef = BCoef;
	h_ACoef = ACoef;
	h_sigCoef = sigCoef;

	for (i = 1; i <= nMat; i++) {
		for (j = 1; j <= nStrikes; j++) {
			sum[i][j] = 0.0;
			SStdErr[i][j] = 0.0;
		}
	}

	RInt = 0.0;
	jstart = 1;
	for (i = 1; i <= nMat; i++) {
		for (j = jstart; j <= Maturity[i]; j++) {
			RInt += rd[j];
		}
		Discount[i] = exp(-RInt / DaysPerYear);
		jstart = Maturity[i] + 1;
	}

	h_dt = dt;
	h_sqdt = sqdt;

	h_rd[0] = 0.0;
	h_rf[0] = 0.0;
	for (i = 1; i <= nMat2; i++) {
		h_rd[i] = rd[i];
		h_rf[i] = rf[i];
	}

	//	Move values to GPU device arrays
	cudaMemcpy(d_seeds, h_seeds, 6 * nSim * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Maturity, Maturity, (nMat+1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rd, h_rd, (nMat2+1) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rf, h_rf, (nMat2 + 1) * sizeof(float), cudaMemcpyHostToDevice);

	SimulatePathGPU <<<(1 + nSim / 128), 128 >>> (
		nSim, nMat, nSteps, h_dt, h_sqdt, h_FX0, h_v0, h_BCoef, h_ACoef, 
		h_sigCoef, h_rho, d_seeds, d_Maturity, d_rd, d_rf, d_SimFX);

	cudaGetLastError();
	cudaDeviceSynchronize();

	//Read GPU simulations back to host CPU 
	cudaMemcpy(h_SimFX, d_SimFX, (nMat+1)*nSim * sizeof(float), cudaMemcpyDeviceToHost);

	for (iT = 0; iT < nSim; iT++) {
		for (j = 1; j <= nMat; j++) {
//	Value options, i = 1, ..., nStrikes, fpor Maturity[j]
			FX = h_SimFX[iT*(nMat + 1) + j];
			for (k = 1; k <= nStrikes; k++) {
				if (OptType[j][k] == 1) {
					tem = Discount[j] * max(0.0, FX - Strikes[j][k]);
					sum[j][k] += tem;
					SStdErr[j][k] += tem * tem;
				}
				else {
					tem = Discount[j] * max(0.0, Strikes[j][k] - FX);
					sum[j][k] += tem;
					SStdErr[j][k] += tem * tem;
				}
			}
			jstart = jend + 1;
		}		//	End of loop on j for nMat maturities

	}                //     end of loop on iT for independent simulations

	for (i = 1; i <= nMat; i++) {
		for (j = 1; j <= nStrikes; j++) {
			k = i * (nStrikes + 1) + j;
			FXOptPrice[k][jThr] = sum[i][j];
			FXStdErr[k][jThr] = SStdErr[i][j];
		}
	}

	//		Release memory allocation
	free(Discount);
	free_matrix_fp64(sum);
	free_matrix_fp64(SStdErr);

	free(h_seeds);
	free(h_SimFX);
	free(h_rd);
	free(h_rf);

	cudaFree(d_seeds);
	cudaFree(d_Maturity);
	cudaFree(d_SimFX);
	cudaFree(d_rd);
	cudaFree(d_rf);

}

#undef NR_END
#undef FREE_ARG
#undef MAX_NUM_THREADS 
