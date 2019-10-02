// Test_FD_LIBOROption.cpp : Defines the entry point for the console application.
//
//			Test_FD_LIBOROption.cpp
//
//			Function to price LIBOR options in the 3 Factor Hull-White model 
//			Using finite difference method
//
//			Created January 2019, by Louis Scott
//
//			Add American optionn pricing
//			Calculate futures exp. affine coefficient at each time step and store 
//				in global memory
//
//			set up code to run 1 expiration for European or American calls & puts
//
#include "pch.h"
#include <iostream>
#include "DynamicMemoryAllocation.h"
#include "ODE_Solver.h"

#define MAX_NUM_THREADS 8
#define NEQ 1
using namespace std;

/* prototype */
unsigned __stdcall RunThread(void *param);
void FiniteDiff_LIBOROptions(int jThr);
void ExplicitSolution1(int ny_grid, int jmin, int jmax, int kmin, int kmax,
	int j_y1min, int j_y1max, int k_y2min, int k_y2max, double dt,
	double v_lamsig[], double rgrid[], double y1grid[], double y2grid[],
	double dtdr, double dtdy1, double dtdy2, double ***V1, double ***V2);
void ExplicitSolution2(int ny_grid, int jmin, int jmax, int kmin, int kmax,
	int j_y1min, int j_y1max, int k_y2min, int k_y2max, double dt,
	double v_lamsig[], double rgrid[], double y1grid[], double y2grid[],
	double dtdr, double dtdy1, double dtdy2, double ***V1, double ***V2);
void ImplicitSolution(int ny_grid, int jmin, int jmax, int kmin, int kmax, double dt,
	double dr, double v_lamsig[], double rgrid[], double y1grid[], double y2grid[],
	double dtdr, double dtdy1, double dtdy2, double *wkVa, double *wkVb,
	double ***V1, double ***V2);
void derivs(double t, double y[], double dydx[]);
void derivs2(double t, double y[], double dydx[]);

/*     Global Variables     */
const double DaysPerYear = 365.245;
int nMat, *nSteps_vector, nStepsPerDaY, iMat, jMat;
int *nMat1, *nMat2, **OptType, **ExerciseType;
double *dt_vector, r0, y10, y20;
double kappa[4], theta[4], sigma[4], lambda[4];
double **LIBOROption, **Strikes, *FutRate0, *FwdSpread;
double **AFut, **B0Fut, **B1Fut, **B2Fut;

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
	int nMaturities, nThreadsToUse, nThreads, n_maxStrikes, nmaxSteps, *nStrikes;
	double r0_in, y10_in, y20_in;
	int nSteps, neq, nok, nbad;
	double dt, eps, hstart, t1, t2, h1, hmin, tem;
	double C1, C2, C01, C02, *ystart;
	double Atem, B0tem, B1tem, B2tem;
	double time0, time1;
	struct _timeb timebuffer;

	HANDLE threads[MAX_NUM_THREADS];
	errno_t errcheck;
	unsigned threadID[MAX_NUM_THREADS];
	// create a thread_data_t argument array 
	thread_data_t thr_data[MAX_NUM_THREADS];

	errcheck = fopen_s(&fout, "Check_HW_3F_FiniteDifference.txt", "w");

	FILE *fin;
	errcheck = fopen_s(&fin, "FD_LIBOROption_Parameters.txt", "r");
	if (errcheck) printf(" File FD_LIBOROption_Parameters.txt not opened \n");

	fscanf_s(fin, " %i %i %i %i ", &nMaturities, &nThreadsToUse, &n_maxStrikes, &nmaxSteps);

	nMat = nMaturities;
	nThreads = nThreadsToUse;
	if (nThreads > MAX_NUM_THREADS) nThreads = MAX_NUM_THREADS;

	printf("   Now running model with %i maturities and %i threads \n", nMat, nThreads);
	fprintf(fout, "   Now running model with %i maturities and %i threads \n", nMat, nThreads);

	A = (double *)malloc(25 * sizeof(double));
	LIBOROption = matrix_fp64(nMat + 1, n_maxStrikes + 1);
	Strikes = matrix_fp64(nMat + 1, n_maxStrikes + 1);
	OptType = int_matrix(1, nMat, 1, n_maxStrikes);
	ExerciseType = int_matrix(1, nMat, 1, n_maxStrikes);

	FutRate0 = (double *)malloc((nMat + 1) * sizeof(double));
	FwdSpread = (double *)malloc((nMat + 1) * sizeof(double));
	nMat1 = (int *)malloc((nMat + 1) * sizeof(int));
	nMat2 = (int *)malloc((nMat + 1) * sizeof(int));
	dt_vector = (double *)malloc((nMat + 1) * sizeof(double));
	nSteps_vector = (int *)malloc((nMat + 1) * sizeof(int));
	nStrikes = (int *)malloc((nMat + 1) * sizeof(int));
	AFut = matrix_fp64(nMat + 1, nmaxSteps + 1);
	B0Fut = matrix_fp64(nMat + 1, nmaxSteps + 1);
	B1Fut = matrix_fp64(nMat + 1, nmaxSteps + 1);
	B2Fut = matrix_fp64(nMat + 1, nmaxSteps + 1);
	ystart = (double *)malloc((NEQ + 1) * sizeof(double));

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

	for (i = 1; i <= nMat; i++) {
		fscanf_s(fin, " %i %i %i %lf %i ", &nStrikes[i], &nMat1[i], &nMat2[i], &FutRate0[i], &nSteps_vector[i]);
		printf("  Maturity %i:  %i strikes  %i days to expiration with tenor of %i days  Futures %f   %i time steps \n",
			i, nStrikes[i], nMat1[i], nMat2[i], FutRate0[i], nSteps_vector[i]);
		for (j = 1; j <= nStrikes[i]; j++) {
			fscanf_s(fin, " %i %i %lf ", &OptType[i][j], &ExerciseType[i][j], &Strikes[i][j]);
			printf(" %i %i %i %f \n", i, OptType[i][j], ExerciseType[i][j], Strikes[i][j]);
			fprintf(fout, " %i %i %i %f \n", i, OptType[i][j], ExerciseType[i][j], Strikes[i][j]);
		}
		if (nStrikes[i] > nThreads) nStrikes[i] = nThreads;
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

	//    End of input section

	for (i = 1; i <= nMat; i++) dt_vector[i] = (1.0*nMat1[i]) / (365.0*nSteps_vector[i]);

	hstart = 0.0;
	eps = 0.00000001;
	hmin = 0.0;
	nrerror = 0;
	neq = NEQ;
	//	Compute exp. affine coefficients for 3m, using nMat2, at each maturity/expiration
	//	NMat2 days past 2 days forward 
	for (i = 1; i <= nMat; i++) {
		nSteps = nSteps_vector[i];
		iMat = i;
		jMat = nSteps;
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

		t1 = t2;
		t2 = (2 + nMat2[i]) / DaysPerYear;
		h1 = 0.5*(t2 - t1);
		odeint(ystart, neq, t1, t2, eps, h1, hmin, &nok, &nbad, &nrerror, derivs, rkqs);
		if (nrerror < 0) {
			ifail = -10 + nrerror;
			return ifail;
		}
		AFut[i][nSteps] = ystart[1] - Atem;

		//	Need to calculate B coefficients 
		tem = kappa[0] * t2;
		if (fabs(tem) < 1.0e-06) B0Fut[i][nSteps] = t2 * (1.0 - 0.5*tem + tem * tem / 6.0 - tem * tem*tem / 24.0);
		else B0Fut[i][nSteps] = (1.0 - exp(-tem)) / kappa[0];
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

		B1Fut[i][nSteps] = C1 - exp(-kappa[1] * t2)*C01;
		B2Fut[i][nSteps] = C2 - exp(-kappa[2] * t2)*C02;

		B0Fut[i][nSteps] = B0Fut[i][nSteps] - B0tem;
		B1Fut[i][nSteps] = B1Fut[i][nSteps] - B1tem;
		B2Fut[i][nSteps] = B2Fut[i][nSteps] - B2tem;

//		printf("  Coefficients for %i %f %f %f %f \n", nSteps, AFut[i][nSteps], B0Fut[i][nSteps], B1Fut[i][nSteps], B2Fut[i][nSteps]);

//	*************   Compute futures/forward spread       ***********
//	*************   Calculate futures on 3m OIS and then calculate spread from input futures rates
//	Use previous coefficients and calculate MGF at each expiration or exercise date, 
//	then calculate spread between 3m LIBOR futures and 3m OIS futures:
//	(hypothetical) -> FwdSpread[i], calculated for annualized simple interest rate

		ystart[1] = AFut[i][nSteps];
		dt = dt_vector[i];
		t1 = 0.0;
		for (j = 1; j <= nSteps; j++) {
			k = nSteps - j;
			t2 = t1 + dt;
			h1 = 0.5*dt;
			odeint(ystart, neq, t1, t2, eps, h1, hmin, &nok, &nbad, &nrerror, derivs2, rkqs);
			if (nrerror < 0) {
				ifail = -10 + nrerror;
				return ifail;
			}
			AFut[i][k] = ystart[1];
			//	Need to calculate B coefficients for 2 day rate 
			B0Fut[i][k] = exp(-kappa[0] * t2)*B0Fut[i][nSteps];

			tem = (kappa[0] - kappa[1]) * t2;
			if (fabs(tem) < 1.0e-06) C01 = t2 * (1.0 - 0.5*tem + tem * tem / 6.0 - tem * tem*tem / 24.0);
			else C01 = (1.0 - exp(-tem)) / (kappa[0] - kappa[1]);
			tem = (kappa[0] - kappa[2]) * t2;
			if (fabs(tem) < 1.0e-06) C02 = t2 * (1.0 - 0.5*tem + tem * tem / 6.0 - tem * tem*tem / 24.0);
			else C02 = (1.0 - exp(-tem)) / (kappa[0] - kappa[2]);

			C1 = exp(-kappa[1] * t2);
			C2 = exp(-kappa[2] * t2);

			B1Fut[i][k] = C1 * (B1Fut[i][nSteps] + kappa[0] * B0Fut[i][nSteps] * C01);
			B2Fut[i][k] = C2 * (B2Fut[i][nSteps] + kappa[0] * B0Fut[i][nSteps] * C02);

			//			printf("  Coefficients for %i %f %f %f %f \n", k, AFut[i][k], B0Fut[i][k], B1Fut[i][k], B2Fut[i][k]);

			t1 = t2;

		}		//	enf of loop j

	//	Now calcuate spread between 3m LIBOR futures and 3m OIS futures
	//	place the 3m OIS futures into tem
		tem = AFut[i][0] + B0Fut[i][0] * r0 + B1Fut[i][0] * y10 + B2Fut[i][0] * y20;
		tem = (exp(tem) - 1.0)*360.0 / nMat2[i];
		FwdSpread[i] = FutRate0[i] - tem;

		printf("  3m LIBOR - 3m OIS Futures Rate Spread %f     3m OIS Futures  %f  \n", FwdSpread[i], tem);
		fprintf(fout, "  3m LIBOR - 3m OIS Futures Rate Spread %f     3m OIS Futures  %f  \n", FwdSpread[i], tem);

		//		fprintf(fout, "  OIS Futures Coeff  %f %f %f %f %f \n", AConst[i], Atem, B0tem, B1tem, B2tem);

	}		//	end of loop i

	fprintf(fout, "  Initial setup complete, now running the finitie difference algorithm \n");

	_ftime64_s(&timebuffer);
	time0 = timebuffer.time + timebuffer.millitm / 1000.0;

	//	Set up multi-threading here
	for (i = 1; i <= nMat; i++) {
		iMat = i;
		k = nThreads;
		if (k > nStrikes[i]) k = nStrikes[i];
		if (k == 1) FiniteDiff_LIBOROptions(0);
		else {
			for (j = 0; j < k; j++) {
				thr_data[j].tid = j;
				threads[j] = (HANDLE)_beginthreadex(NULL, 0, RunThread, &thr_data[j], 0, &threadID[j]);
			}
			WaitForMultipleObjects(nThreads, threads, TRUE, INFINITE);
			for (j = 0; j < k; j++) CloseHandle(threads[j]);
		}

		printf("  LIBOROption  Maturity %i days \n", nMat1[i]);
		fprintf(fout, "  LIBOROption  Maturity %i days \n", nMat1[i]);
		for (j = 1; j <= nStrikes[i]; j++) {
			printf("  Type %i  Exercise %i  Strike %f  LIBOR Option %12.8f \n", OptType[i][j], ExerciseType[i][j], Strikes[i][j], LIBOROption[i][j]);
			fprintf(fout, "  Type %i  Exercise %i  Strike %f  LIBOR Option %12.8f \n", OptType[i][j], ExerciseType[i][j], Strikes[i][j], LIBOROption[i][j]);
		}

	}		//	end of loop i

	//	Average across the threads

	fprintf(fout, "  Completed finite difference algorithm \n");

	_ftime64_s(&timebuffer);
	time1 = timebuffer.time + timebuffer.millitm / 1000.0;
	time1 = time1 - time0;

	printf("  FD compute time %f \n", time1);
	fprintf(fout, "  FD compute time %f \n", time1);

	fclose(fout);
	fclose(fin);

	//   free the work arrays
	free(A);
	free_matrix_fp64(LIBOROption);
	free_matrix_fp64(Strikes);
	free_int_matrix(OptType, 1, nMat, 1, n_maxStrikes);
	free_int_matrix(ExerciseType, 1, nMat, 1, n_maxStrikes);
	free(FutRate0);
	free(FwdSpread);
	free(nMat1);
	free(nMat2);
	free(dt_vector);
	free(nSteps_vector);
	free(nStrikes);
	free_matrix_fp64(AFut);
	free_matrix_fp64(B0Fut);
	free_matrix_fp64(B1Fut);
	free_matrix_fp64(B2Fut);
	free(ystart);

	return 0;
}


unsigned __stdcall RunThread(void *param)
{
	int iThread;
	thread_data_t *data = (thread_data_t *)param;
	iThread = data->tid;
	printf("  Running thread %i from RunThread \n", iThread);
	FiniteDiff_LIBOROptions(iThread);

	return 1;
}

void FiniteDiff_LIBOROptions(int iThread)
{
	int i, j, k, kk, jThr, nSteps, iT, ny_grid_LB, ny_grid_UB, icount, iStep;
	int jmin, jmax, kmin, kmax, jmin2, jmax2, j_y1max, j_y1min, k_y2max, k_y2min;
	int ny_grid, ipos_r0, jpos_y10, kpos_y20;
	double *r_grid, *y1_grid, *y2_grid, *wkVa, *wkVb;
	double ***V1, ***V2;
	double dr, dy1, dy2;
	double dtdr, dtdy1, dtdy2, y1min, y2min, y1max, y2max;
	double dt, tem, temopt;
	double sigz[3], lamsig[3], temkappa[3], tem2kappa[3], temexp[3];

	jThr = iThread + 1;
	dt = dt_vector[iMat];
	nSteps = nSteps_vector[iMat];

	printf(" Now running the finite difference algorithm for the discount function in thread %i with %i time steps \n", jThr, nSteps);
	fprintf(fout, " Now running the finite difference algorithm for the discount function in thread %i with %i time steps \n", jThr, nSteps);

	//	Compute the step sizes for the 3 state variables

	dr = sigma[0] * sqrt(1.5*dt);
	dy1 = sigma[1] * sqrt(1.5*dt);
	dy2 = sigma[2] * sqrt(1.5*dt);

	ny_grid = 1 + 2 * nSteps;
	ipos_r0 = 1 + nSteps;
	jpos_y10 = 1 + nSteps;
	kpos_y20 = 1 + nSteps;

	printf("  Thread %i:  dt %12.6e  dr %12.6e  dy1 %12.6e  dy2 %12.6e \n", jThr, dt, dr, dy1, dy2);

	r_grid = (double *)malloc((size_t)((ny_grid + 1) * sizeof(double)));
	y1_grid = (double *)malloc((size_t)((ny_grid + 1) * sizeof(double)));
	y2_grid = (double *)malloc((size_t)((ny_grid + 1) * sizeof(double)));
	wkVa = (double *)malloc((size_t)((ny_grid + 1) * sizeof(double)));
	wkVb = (double *)malloc((size_t)((ny_grid + 1) * sizeof(double)));

	V1 = d3Tensor(ny_grid + 1, ny_grid + 1, ny_grid + 1);
	V2 = d3Tensor(ny_grid + 1, ny_grid + 1, ny_grid + 1);

	r_grid[ipos_r0] = r0;
	y1_grid[jpos_y10] = y10;
	y2_grid[kpos_y20] = y20;

	dtdr = dt / dr;
	dtdy1 = dt / dy1;
	dtdy2 = dt / dy2;

	printf("  Thread %i:  dtdr %f  dtdy1 %f  dtdy2 %f  \n", jThr, dtdr, dtdy1, dtdy2);

	for (j = 0; j < 3; j++) {
		temexp[j] = exp(-kappa[j] * dt);
		tem = kappa[j] * dt;
		if (fabs(tem) < 1.0e-06) temkappa[j] = 1.0 - 0.5*tem + tem * tem / 6.0 - tem * tem*tem / 24.0 + tem * tem*tem*tem / 120.0;
		else temkappa[j] = (1.0 - temexp[j]) / tem;
		tem = 2.0*kappa[j] * dt;
		if (fabs(tem) < 1.0e-06) tem2kappa[j] = 1.0 - 0.5*tem + tem * tem / 6.0 - tem * tem*tem / 24.0 + tem * tem*tem*tem / 120.0;
		else tem2kappa[j] = (1.0 - temexp[j] * temexp[j]) / tem;
		lamsig[j] = lambda[j] * sigma[j] * sigma[j] * dt*temkappa[j];
		sigz[j] = sigma[j] * sqrt(dt*tem2kappa[j]);
	}

	//	The initial positions are at 1+nSteps
	r_grid[ipos_r0] = r0;
	y1_grid[jpos_y10] = y10;
	y2_grid[kpos_y20] = y20;
	for (kk = 1; kk <= nSteps; kk++) {
		r_grid[ipos_r0 + kk] = r0 + kk * dr;
		r_grid[ipos_r0 - kk] = r0 - kk * dr;
		y1_grid[jpos_y10 + kk] = y10 + kk * dy1;
		y1_grid[jpos_y10 - kk] = y10 - kk * dy1;
		y2_grid[kpos_y20 + kk] = y20 + kk * dy2;
		y2_grid[kpos_y20 - kk] = y20 - kk * dy2;
	}
	printf("   Min and max for r on grid %f %f \n", r_grid[1], r_grid[ny_grid]);
	//	fprintf(fout, "   Min and max for r on grid %f %f \n", r_grid[1], r_grid[ny_grid]);
	printf("   Min and max for y1 on grid %f %f \n", y1_grid[1], y1_grid[ny_grid]);
	//	fprintf(fout, "   Min and max for y1 on grid %f %f \n", y1_grid[1], y1_grid[ny_grid]);
	printf("   Min and max for y2 on grid %f %f \n", y2_grid[1], y2_grid[ny_grid]);
	//	fprintf(fout, "   Min and max for y2 on grid %f %f \n", y2_grid[1], y2_grid[ny_grid]);

	y1max = ((kappa[1] * theta[1] - lamsig[1]) + 2.0 / (3.0*dtdy1)) / kappa[1];
	y1min = ((kappa[1] * theta[1] - lamsig[1]) - 2.0 / (3.0*dtdy1)) / kappa[1];
	printf("    y1min  %f    y1max %f   \n", y1min, y1max);
	//	fprintf(fout, "    y1min  %f    y1max %f   \n", y1min, y1max);

	y2max = (-lamsig[2] + 2.0 / (3.0*dtdy2)) / kappa[2];
	y2min = (-lamsig[1] - 2.0 / (3.0*dtdy2)) / kappa[2];
	printf("    y2min  %f    y2max %f   \n", y2min, y2max);
	//	fprintf(fout, "    y2min  %f    y2max %f   \n", y2min, y2max);

	j_y1max = floor((y1max - y10) / dy1) + jpos_y10 + 1;
	j_y1min = floor((y1min - y10) / dy1) + jpos_y10;
	k_y2max = floor((y2max - y20) / dy2) + kpos_y20 + 1;
	k_y2min = floor((y2min - y20) / dy2) + kpos_y20;

	//	compute 3m LIBOR rate and option payoffs
	//	First compute 3m OIS on simulatin path
	//	printf("   Setting LIBOR Optionvalues at expiration for thread %i  ny_grid = %i \n", jThr, ny_grid);
	for (i = 1; i <= ny_grid; i++) {
		for (j = 1; j <= ny_grid; j++) {
			for (k = 1; k <= ny_grid; k++) {
				tem = (exp(AFut[iMat][nSteps] + B0Fut[iMat][nSteps] * r_grid[i] + B1Fut[iMat][nSteps] * y1_grid[j] + B2Fut[iMat][nSteps] * y2_grid[k]) - 1.0)*360.0 / nMat2[iMat];
				tem = tem + FwdSpread[iMat];
				if (OptType[iMat][jThr] == 1) V2[i][j][k] = max(0.0, tem - Strikes[iMat][jThr]);
				else V2[i][j][k] = max(0.0, Strikes[iMat][jThr] - tem);
			}
		}
	}

	ny_grid_LB = 1;
	ny_grid_UB = ny_grid;
	for (iT = 1; iT < nSteps; iT++) {
		if ((iT % 10) == 0) printf("  Thread %i  Running day %i \n", jThr, iT);
		iStep = nSteps - iT;
		ny_grid_LB = ny_grid_LB + 1;
		ny_grid_UB = ny_grid_UB - 1;
		jmin = ny_grid_LB;
		jmax = ny_grid_UB;
		kmin = ny_grid_LB;
		kmax = ny_grid_UB;
		if (jmin < j_y1min) jmin = j_y1min;
		if (jmax > j_y1max) jmax = j_y1max;
		if (kmin < k_y2min) kmin = k_y2min;
		if (kmax > k_y2max) kmax = k_y2max;
		jmin2 = jmin - 1;
		jmax2 = jmax + 1;
		if (jmin2 < j_y1min) jmin2 = j_y1min;
		if (jmax2 > j_y1max) jmax2 = j_y1max;

		ExplicitSolution1(ny_grid, jmin2, jmax2, kmin, kmax, j_y1min, j_y1max, k_y2min, k_y2max,
			dt, lamsig, r_grid, y1_grid, y2_grid, dtdr, dtdy1, dtdy2, V1, V2);
		ExplicitSolution2(ny_grid, jmin, jmax, kmin, kmax, j_y1min, j_y1max, k_y2min, k_y2max,
			dt, lamsig, r_grid, y1_grid, y2_grid, dtdr, dtdy1, dtdy2, V2, V1);
		ImplicitSolution(ny_grid, jmin, jmax, kmin, kmax, dt, dr,
			lamsig, r_grid, y1_grid, y2_grid, dtdr, dtdy1, dtdy2, wkVa, wkVb, V1, V2);
//	Check American options for early exercise
		if (ExerciseType[iMat][jThr] == 1) {
			kk = ny_grid / 2;
			for (k = kmin; k <= kmax; k++) {
				for (j = jmin; j <= jmax; j++) {

					if (OptType[iMat][jThr] == 2) {
						//	put options on 3m LIBOR futures rate
						icount = 0;
						for (i = 1; i <= ny_grid; i++) {
							if (icount < kk) {
								tem = (exp(AFut[iMat][iStep] + B0Fut[iMat][iStep] * r_grid[i] + B1Fut[iMat][iStep] * y1_grid[j] + B2Fut[iMat][iStep] * y2_grid[k]) - 1.0)*360.0 / nMat2[iMat];
								tem = tem + FwdSpread[iMat];
								temopt = max(0.0, Strikes[iMat][jThr] - tem);
								if (V2[i][j][k] < temopt) V2[i][j][k] = temopt;
								else icount += 1;
							}

						}
					}
					else {
						// call options on 3m LIBOR futures rate
						icount = 0;
						for (i = ny_grid; i > 0; i--) {
							if (icount < kk) {
								tem = (exp(AFut[iMat][iStep] + B0Fut[iMat][iStep] * r_grid[i] + B1Fut[iMat][iStep] * y1_grid[j] + B2Fut[iMat][iStep] * y2_grid[k]) - 1.0)*360.0 / nMat2[iMat];
								tem = tem + FwdSpread[iMat];
								temopt = max(0.0, tem - Strikes[iMat][jThr]);
								if (V2[i][j][k] < temopt) V2[i][j][k] = temopt;
								else icount += 1;
							}

						}
					}

				}	//	end of loop on j
			}	//	end of loop on k
		}

	}

//	One more loop from time 1 back to time 0
	iStep = 0;
	ny_grid_LB = ny_grid_LB + 1;
	ny_grid_UB = ny_grid_UB - 1;
	jmin = ny_grid_LB;
	jmax = ny_grid_UB;
	kmin = ny_grid_LB;
	kmax = ny_grid_UB;
	if (jmin < j_y1min) jmin = j_y1min;
	if (jmax > j_y1max) jmax = j_y1max;
	if (kmin < k_y2min) kmin = k_y2min;
	if (kmax > k_y2max) kmax = k_y2max;
	jmin2 = jmin - 1;
	jmax2 = jmax + 1;
	if (jmin2 < j_y1min) jmin2 = j_y1min;
	if (jmax2 > j_y1max) jmax2 = j_y1max;
	printf("  Start date: jmin jmax kmin kmax %i %i %i %i \n", jmin, jmax, kmin, kmax);
	fprintf(fout, "  Start date: jmin jmax kmin kmax %i %i %i %i \n", jmin, jmax, kmin, kmax);

	ExplicitSolution1(ny_grid, jmin2, jmax2, kmin, kmax, j_y1min, j_y1max, k_y2min, k_y2max,
		dt, lamsig, r_grid, y1_grid, y2_grid, dtdr, dtdy1, dtdy2, V1, V2);
	ExplicitSolution2(ny_grid, jmin, jmax, kmin, kmax, j_y1min, j_y1max, k_y2min, k_y2max,
		dt, lamsig, r_grid, y1_grid, y2_grid, dtdr, dtdy1, dtdy2, V2, V1);
	ImplicitSolution(ny_grid, jmin, jmax, kmin, kmax, dt, dr,
		lamsig, r_grid, y1_grid, y2_grid, dtdr, dtdy1, dtdy2, wkVa, wkVb, V1, V2);

	if (ExerciseType[iMat][jThr] == 1) {
		//	Check for early exercise on start date
		i = ipos_r0;
		j = jpos_y10;
		k = kpos_y20;
		tem = (exp(AFut[iMat][iStep] + B0Fut[iMat][iStep] * r_grid[i] + B1Fut[iMat][iStep] * y1_grid[j] + B2Fut[iMat][iStep] * y2_grid[k]) - 1.0)*360.0 / nMat2[iMat];
		tem = tem + FwdSpread[iMat];
		if (OptType[iMat][jThr] == 1) {
			temopt = max(0.0, tem - Strikes[iMat][jThr]);
			if (V1[i][j][k] < temopt) V1[i][j][k] = temopt;
		}
		else {
			temopt = max(0.0, Strikes[iMat][jThr] - tem);
			if (V1[i][j][k] < temopt) V1[i][j][k] = temopt;
		}
	}

	LIBOROption[iMat][jThr] = V1[ipos_r0][jpos_y10][kpos_y20];

	//		fprintf(fout, " Finished the finite difference algorithm for thread %i \n", jThr);

	//		Release memory allocation

	free(r_grid);
	free(y1_grid);
	free(y2_grid);
	free(wkVa);
	free(wkVb);

	free_d3Tensor(V1);
	free_d3Tensor(V2);

}

void ExplicitSolution1(int ny_grid, int jmin, int jmax, int kmin, int kmax,
	int j_y1min, int j_y1max, int k_y2min, int k_y2max, double dt,
	double lamsig[], double rgrid[], double y1grid[], double y2grid[],
	double dtdr, double dtdy1, double dtdy2, double ***V1, double ***V2)
{
	//	This function calculates explicit solution in y2 dimension
	int i, j, k;
	double drift, cfu, cf0, cfd, cfuu, cfdd, tem1_3, tem2_3;

	tem1_3 = 1.0 / 3.0;
	tem2_3 = 2.0 / 3.0;

	for (k = kmin; k <= kmax; k++) {
		drift = (-kappa[2] * y2grid[k] - lamsig[2])*dtdy2;
		if ((k > k_y2min) && (k < k_y2max)) {
			cf0 = tem1_3;
			cfu = cf0 + 0.5*drift;
			cfd = cf0 - 0.5*drift;
			for (j = jmin; j <= jmax; j++) {
				for (i = 1; i <= ny_grid; i++) {
					V1[i][j][k] = cfu * V2[i][j][k + 1] + cf0 * V2[i][j][k] + cfd * V2[i][j][k - 1];
				}
			}
		}
		else {
			if (k <= k_y2min) {
				//				printf("   Computing on y2 min  k = %i   \n", k);
				cf0 = 1.0 + tem1_3 - drift;
				cfu = drift - tem2_3;
				cfuu = tem1_3;
				for (j = jmin; j <= jmax; j++) {
					for (i = 1; i <= ny_grid; i++) {
						V1[i][j][k] = cf0 * V2[i][j][k] + cfu * V2[i][j][k + 1] + cfuu * V2[i][j][k + 2];
					}
				}
			}
			else {
				cf0 = 1.0 + tem1_3 + drift;
				cfd = -drift - tem2_3;
				cfdd = tem1_3;
				for (j = jmin; j <= jmax; j++) {
					for (i = 1; i <= ny_grid; i++) {
						V1[i][j][k] = cf0 * V2[i][j][k] + cfd * V2[i][j][k - 1] + cfdd * V2[i][j][k - 2];
					}
				}
			}
		}
	}
}

void ExplicitSolution2(int ny_grid, int jmin, int jmax, int kmin, int kmax,
	int j_y1min, int j_y1max, int k_y2min, int k_y2max, double dt,
	double lamsig[], double rgrid[], double y1grid[], double y2grid[],
	double dtdr, double dtdy1, double dtdy2, double ***V1, double ***V2)
{
	//	This function calculates explicit solution in y1 dimension
	int i, j, k;
	double drift, cfu, cf0, cfd, cfuu, cfdd, tem1_3, tem2_3;

	tem1_3 = 1.0 / 3.0;
	tem2_3 = 2.0 / 3.0;

	for (j = jmin; j <= jmax; j++) {
		if ((j > j_y1min) && (j < j_y1max)) {
			drift = (kappa[1] * (theta[1] - y1grid[j]) - lamsig[1])*dtdy1;
			cf0 = tem1_3;
			cfu = cf0 + 0.5*drift;
			cfd = cf0 - 0.5*drift;
			for (k = kmin; k <= kmax; k++) {
				for (i = 1; i <= ny_grid; i++) {
					V1[i][j][k] = cfu * V2[i][j + 1][k] + cf0 * V2[i][j][k] + cfd * V2[i][j - 1][k];
				}
			}
		}
		else {
			if (j <= j_y1min) {
				printf("   Computing on y1 min  j = %i   \n", j);
				cf0 = 1.0 + tem1_3 - drift;
				cfu = drift - tem2_3;
				cfuu = tem1_3;
				for (k = kmin; j <= kmax; k++) {
					for (i = 1; i <= ny_grid; i++) {
						V1[i][j][k] = cf0 * V2[i][j][k] + cfu * V2[i][j + 1][k] + cfuu * V2[i][j + 2][k];
					}
				}
			}
			else {
				printf("   Computing on y1 max  j = %i   \n", j);
				cf0 = 1.0 + tem1_3 + drift;
				cfd = -drift - tem2_3;
				cfdd = tem1_3;
				for (k = kmin; k <= kmax; k++) {
					for (i = 1; i <= ny_grid; i++) {
						V1[i][j][k] = cf0 * V2[i][j][k] + cfd * V2[i][j - 1][k] + cfdd * V2[i][j - 2][k];
					}
				}
			}
		}
	}
}

void ImplicitSolution(int ny_grid, int jmin, int jmax, int kmin, int kmax, double dt,
	double dr, double lamsig[], double rgrid[], double y1grid[], double y2grid[],
	double dtdr, double dtdy1, double dtdy2, double *wkVa, double *wkVb,
	double ***V1, double ***V2)
{
	//	This function calculates implicit solution in the r dimension
	int i, j, k, ii;
	double a, b, c, drift_r, tem, tem1_3, tem2_3;;

	tem1_3 = 1.0 / 3.0;
	tem2_3 = 2.0 / 3.0;

	for (j = jmin; j <= jmax; j++) {
		for (k = kmin; k <= kmax; k++) {
			i = 1;
			drift_r = (kappa[0] * (y1grid[j] + y2grid[k] - rgrid[i]) - lamsig[0])*dtdr;
			b = 1.0 + rgrid[i] * dt + tem2_3 + (0.5*drift_r - tem1_3)*(1.0 + rgrid[i] * dt) / (1.0 + (rgrid[1] - dr)*dt);
			c = -0.5*drift_r - tem1_3;
			wkVa[i] = c / b;
			wkVb[i] = V2[i][j][k] / b;
			for (i = 2; i < ny_grid; i++) {
				drift_r = (kappa[0] * (y1grid[j] + y2grid[k] - rgrid[i]) - lamsig[0])*dtdr;
				b = 1.0 + rgrid[i] * dt + tem2_3;
				a = 0.5*drift_r - tem1_3;
				c = -0.5*drift_r - tem1_3;
				tem = (b - a * wkVa[i - 1]);
				wkVa[i] = c / tem;
				wkVb[i] = (V2[i][j][k] - a * wkVb[i - 1]) / tem;
			}
			i = ny_grid;
			drift_r = (kappa[0] * (y1grid[j] + y2grid[k] - rgrid[i]) - lamsig[0])*dtdr;
			b = 1.0 + rgrid[i] * dt + tem2_3 + (-0.5*drift_r - tem1_3)*(1.0 + rgrid[i] * dt) / (1.0 + (rgrid[ny_grid] + dr)*dt);
			a = 0.5*drift_r - tem1_3;
			tem = (b - a * wkVa[i - 1]);
			wkVb[i] = (V2[i][j][k] - a * wkVb[i - 1]) / tem;

			V1[i][j][k] = wkVb[i];
			for (ii = 1; ii <= (ny_grid - 1); ii++) {
				i = i - 1;
				V1[i][j][k] = wkVb[i] - wkVa[i] * V1[i + 1][j][k];
			}
		}
	}

	for (j = jmin; j <= jmax; j++) {
		for (k = kmin; k <= kmax; k++) {
			//		Move results from V1 to V2 in the y(j) dimension
			for (i = 1; i <= ny_grid; i++) V2[i][j][k] = V1[i][j][k];
		}
	}

}

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

	B0tem = exp(-kappa[0] * t)*B0Fut[iMat][jMat];

	tem = (kappa[0] - kappa[1]) * t;
	if (fabs(tem) < 1.0e-06) C01 = t * (1.0 - 0.5*tem + tem * tem / 6.0 - tem * tem*tem / 24.0);
	else C01 = (1.0 - exp(-tem)) / (kappa[0] - kappa[1]);
	tem = (kappa[0] - kappa[2]) * t;
	if (fabs(tem) < 1.0e-06) C02 = t * (1.0 - 0.5*tem + tem * tem / 6.0 - tem * tem*tem / 24.0);
	else C02 = (1.0 - exp(-tem)) / (kappa[0] - kappa[2]);

	C1 = exp(-kappa[1] * t);
	C2 = exp(-kappa[2] * t);

	B1tem = C1 * (B1Fut[iMat][jMat] + kappa[0] * B0Fut[iMat][jMat] * C01);
	B2tem = C2 * (B2Fut[iMat][jMat] + kappa[0] * B0Fut[iMat][jMat] * C02);

	dydx[1] = kappa[1] * theta[1] * B1tem - 0.5*(sigma[0] * sigma[0] * B0tem*B0tem + sigma[1] * sigma[1] * B1tem*B1tem + sigma[2] * sigma[2] * B2tem*B2tem)
		- (lambda[0] * sigma[0] * sigma[0] * B0tem + lambda[1] * sigma[1] * sigma[1] * B1tem + lambda[2] * sigma[2] * sigma[2] * B2tem);

}

#undef NR_END
#undef FREE_ARG
#undef MAX_NUM_THREADS 
#undef MAXSTP
#undef TINY
#undef SAFETY
#undef PGROW
#undef PSHRNK
#undef ERRCON
#undef NEQ
