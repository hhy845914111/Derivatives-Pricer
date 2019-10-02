#include <iostream>
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

#define MAX_NUM_THREADS 10

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
double dSeeds_Threads[MAX_NUM_THREADS][6];
double dt, sqdt, FX0, v0, yinit;
double kappa, theta, sigma, lambda, rho;
double **FXOptPrice, **FXStdErr, *rd, *rf, **Strikes ;

FILE *fout;

/* create thread argument struct for Run_Thread() */
typedef struct _thread_data_t {
	int tid;
	//	double stuff;
} thread_data_t;

//	Note:  you need r_dom and either r_for or fwdFX
//	Removed fwdFX
DLLEXPORT void FX_SV_Model_CPU(int nThreadsIn, int nSimIn, int nTimeStepsIn, int nExpIn, 
	int nStrikesPerExp, double spotFX, double spotV, int *daysExp, double *r_dom, 
	double *r_for, double *params, int *OptTypeIn, double *StrikesIn, 
	double *OptPriceOut, double *StdErrOut) 
{
	const unsigned int im1 = 4294967087;
	const unsigned int im2 = 4294944443;
	int i, j, k, iThr;
	double dSeed_Init[6];
	int nSimsPerPath, nThreads;
	unsigned int **An1, **An2, **Bn1, **Bn2, iseed[6];
	unsigned long long lp1, lp2, seed1[3], seed2[3], sseed1[3], sseed2[3];
	HANDLE threads[MAX_NUM_THREADS];
	errno_t errcheck;
	unsigned threadID[MAX_NUM_THREADS];
	// create a thread_data_t argument array 
	thread_data_t thr_data[MAX_NUM_THREADS];

	errcheck = fopen_s(&fout, "Check_FX_SV_Model_Simulations.txt", "w");

	nThreads = nThreadsIn;
	nSim = nSimIn;
	nSteps = nTimeStepsIn;
	nMat = nExpIn;
	nStrikes = nStrikesPerExp;
	FX0 = spotFX;
	v0 = spotV;
	yinit = log(v0*v0);
	fprintf(fout, " spot FX %f   spot V %f  yinit %f  \n", FX0, v0, yinit);
	if (nThreads > MAX_NUM_THREADS) nThreads = MAX_NUM_THREADS;
//	take iseed from inuts
	iseed[0] = 23949459;
	iseed[1] = 948758;
	iseed[2] = 505938;
	iseed[3] = 23949459;
	iseed[4] = 948758;
	iseed[5] = 505938;

	fprintf(fout, "  spot %f  vol %f \n", FX0, v0);
	fprintf(fout, " nThreads %i nSim %i nSteps %i nMat %i nStrikes %i \n", nThreads, nSim, nSteps, nMat, nStrikes);

	Maturity = (int *)malloc((nMat + 1) * sizeof(int));
	rd = (double *)malloc((daysExp[nMat] + 1) * sizeof(double));
	rf = (double *)malloc((daysExp[nMat] + 1) * sizeof(double));
	
	OptType = int_matrix(0, nMat, 0, nStrikes);
	FXOptPrice = matrix_fp64((nMat+1)*(nStrikes+1), nThreads+1);
	FXStdErr = matrix_fp64((nMat + 1)*(nStrikes + 1), nThreads+1);
	Strikes = matrix_fp64(nMat+1, nStrikes+1);
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

	fprintf(fout, "  params  %f %f %f %f  %f \n", params[1], params[2], params[3], params[4], params[5]);

	//	Move remaining inuts to global memory
	for (i = 1; i <= nMat; i++) {
		Maturity[i] = daysExp[i];
		fprintf(fout, "  Mat %i \n", Maturity[i]);
		for (j = 1; j <= nStrikes; j++) {
			k = i * (nStrikes+1) + j;
			OptType[i][j] = OptTypeIn[k];
			Strikes[i][j] = StrikesIn[k];
			fprintf(fout, " %i %f    \n", OptType[i][j], Strikes[i][j]);
		}
		fprintf(fout, " \n");
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
	//	printf("  finished seeds for thread 0 \n");

	if (nThreads > 1) {
		nSimsPerPath = Maturity[nMat] * nSteps * 2;
		//		fprintf(fout, " set up seeds for second thread, thread 1, nSimsPerPath %i \n", nSimsPerPath);
		SkipAhead_MRG32k3a(nSimsPerPath, An1, An2);
		//		fprintf(fout, " finished with first Skip ahead \n");
		SkipAhead2_MRG32k3a(nSim, An1, An2, Bn1, Bn2);
		//		fprintf(fout, " finished with second Skip ahead \n");

		for (k = 1; k < nThreads; k++) {
			for (i = 0; i < 3; i++) {
				seed1[i] = dSeeds_Threads[k-1][i];
				seed2[i] = dSeeds_Threads[k-1][i + 3];
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
			//			fprintf(fout, "  finished seeds for thread %i \n", k);
		}

	}		//	end of if nThreads > 1

//	fprintf(fout, "  Seeds for stepping ahead have been generated.  Now running simulations on %i threads \n", nThreads);

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

//	fprintf(fout, "  Completed simulations.  Now averaging across threads \n");

	OptPriceOut[0] = 10.0;

	for (i = 1; i <= nMat; i++) {
		for (j = 1; j <= nStrikes; j++) {
			k = i * (nStrikes+1) + j;
			FXOptPrice[k][0] = 0.0;
			FXStdErr[k][0] = 0.0;
			for (iThr = 1; iThr <= nThreads; iThr++) {
				FXOptPrice[k][0] = FXOptPrice[k][0] + FXOptPrice[k][iThr];
				FXStdErr[k][0] = FXStdErr[k][0] + FXStdErr[k][iThr];
			}
			FXOptPrice[k][0] = FXOptPrice[k][0] / (nThreads*nSim);
			FXStdErr[k][0] = FXStdErr[k][0] / (nThreads*nSim) - FXOptPrice[k][0] * FXOptPrice[k][0];
			FXStdErr[k][0] = sqrt(FXStdErr[k][0] / (nThreads*nSim));
			fprintf(fout, "  Exp %i  Strike %i Opt Price %f \n", i, j, FXOptPrice[k][0]);
			OptPriceOut[k] = FXOptPrice[k][0];
			StdErrOut[k] = FXStdErr[k][0];
		}
	}

	fclose(fout);

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
//	printf("  Running thread %i from RunThread \n", iThread);
	SimulateFX(iThread);

	return 1;
}

void SimulateFX(int iThread)
{
	int i, j, k, iT, jj, kk, jThr, jstart, jend;
	double dseed[6];
	double logFX0, FX, x, y, v, BCoef, ACoef, sigCoef, z1, z2;
	double RInt, tem, tem1, cfexp, ynew, yavg, vnew, vavg, rhosq;
	double **sum, **SStdErr, *Discount;

	sum = matrix_fp64((nMat + 1), (nStrikes + 1));
	SStdErr = matrix_fp64((nMat + 1), (nStrikes + 1));
	Discount = (double *)malloc((size_t)((nMat + 1) * sizeof(double)));

	jThr = iThread + 1;

	for (k = 0; k < 6; k++) dseed[k] = dSeeds_Threads[iThread][k];

//	fprintf(fout, "  Thread %i  seeds %f %f %f %f %f %f \n", jThr, dseed[0], dseed[1], dseed[2], dseed[3], dseed[4], dseed[5]);

	tem = kappa*dt;
	BCoef = exp(-tem);

	if (fabs(tem) < 1.0e-06) cfexp = dt*(1.0 - 0.5*tem + tem*tem/6.0 - tem*tem*tem / 24.0 + tem*tem*tem*tem/120.0);
	else cfexp = (1.0 - BCoef) / kappa;
	ACoef = theta*(1.0 - BCoef) - lambda*sigma*sigma*cfexp;
	tem = 2.0*kappa*dt;
	if (fabs(tem) < 1.0e-06) tem1 = dt*(1.0 - 0.5*tem + tem*tem / 6.0 - tem*tem*tem / 24.0 + tem*tem*tem*tem / 120.0);
	else tem1 = (1.0 - exp(-tem)) / (2.0*kappa);
	sigCoef = sigma*sqrt(tem1);
	fprintf(fout, " A B sigCoef v0 theta %f %f %10.6e %f %f   dt sqdt %f %f \n", ACoef, BCoef, sigCoef, v0, theta, dt, sqdt);

//	fprintf(fout, " Now running simulation in thread %i with seeds %12.1lf %12.1lf %12.1lf %12.1lf %12.1lf %12.1lf \n", jThr,
//		dseed[0], dseed[1], dseed[2], dseed[3], dseed[4], dseed[5]);

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
//		fprintf(fout, " Exp %i  Discount %f \n", i, Discount[i]);
		jstart = Maturity[i] + 1;
	}

	logFX0 = log(FX0);
	rhosq = sqrt(1.0 - rho * rho);
	for (iT = 1; iT <= nSim; iT++) {
		x = logFX0;
		y = yinit;
		v = v0;
		jstart = 1;
		for (j = 1; j <= nMat; j++) {
			jend = Maturity[j];
			for (jj = jstart; jj <= jend; jj++) {
				for (kk = 1; kk <= nSteps; kk++) {
					z2 = sninvdev(dseed);
					z1 = rhosq*sninvdev(dseed) + rho*z2;
					//	Simulate y
					y = BCoef*y + ACoef + sigCoef * z2;
					vnew = exp(0.5*y);
//					vavg = 0.5*(v + vnew);
					x = x + ((rd[jj] - rf[jj]) - 0.5*v*v)*dt + v*sqdt*z1;
					v = vnew;
				}		//	end of loop on kk for time steps within a day

			}		//	End of loop on jj
//	Value options, i = 1, ..., nStrikes, fpor Maturity[j]
			FX = exp(x);
			for (k = 1; k <= nStrikes; k++) {
				if (OptType[j][k] == 1) {
					tem = Discount[j]*max(0, FX - Strikes[j][k]);
					sum[j][k] += tem;
					SStdErr[j][k] += tem*tem;
				}
				else {
					tem = Discount[j] * max(0, Strikes[j][k]-FX);
					sum[j][k] += tem;
					SStdErr[j][k] += tem*tem;
				}
			}
			jstart = jend + 1;
		}		//	End of loop on j for nMat maturities

	}                //     end of loop on iT for independent simulations

	for (i = 1; i <= nMat; i++) {
		for (j = 1; j <= nStrikes; j++) {
			k = i*(nStrikes+1) + j;
			FXOptPrice[k][jThr] = sum[i][j];
			FXStdErr[k][jThr] = SStdErr[i][j];
		}
	}

	//		Release memory allocation
	free(Discount);
	free_matrix_fp64(sum);
	free_matrix_fp64(SStdErr);

}

#undef NR_END
#undef FREE_ARG
#undef MAX_NUM_THREADS 
