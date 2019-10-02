// SVJump_Model_MonteCarlo_MRG32k3a_64.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
//			SVJump_Model_MonteCarlo_MRG32k3a_64.cpp
//
//
//			Created April 2016, by Louis Scott
//			Version 2, full dynamic model, multiple expirations
//			Stochastic volatility 
//			Stock price + 1 volatility state variables
//			
//			With multi-threading
//
//			Day of the week and opening period effects
//
//			Trading Day Indicator, iTradeDaysa: must be 0 for trading days
//				and 1 for Saturday, 2 for Sunday, and 3 for weekday/holidays
//
//			Version 2, extreme jump probability is fixed
//			Version 3, replace linear congruent random number generator with MRG32k3a
//

#include "pch.h"
//#include <iostream>
#include "DynamicMemoryAllocation.h"
#include "SimulationFunctions_MRG32k3a.h"

#define MAX_NUM_THREADS 10

using namespace std;

/* prototype */
unsigned __stdcall RunThread(void *param);
int MonteCarlo(int jThread);

/* Global Variables */
const unsigned int I31_max = 2147483647;
double x0, logP0, rho, rhoJ, muJ1, sigmaJ1, muJ1exp;
double kappa, muJ2, sigmaJ2;
double kappa2, theta2, sigma2, sm0, kappa_dif_tol;
double kappa3, sigma3, x30;
double kappa4, theta4, sigma4, v40;
double *theta, *sigma, *lambdaJ;
double rho12, rho13, rho14, rho23, rho24, rho34;
double kappa5, theta5, sigma5, lambdaJ0, lambdaJ3;
double rhoJ34, muJ3, sigmaJ3, muJ3exp, muJ4, sigmaJ4;
double **SPX_OptPrice, **SPX_StdErr, *SPX_Strike, Discount_SPX;
double **drift_logP, *ForRate;
double DaysPerYear, TradeDaysPerYear, RemTradeDate;
double dt[5], B_drift[5], sqrt_dt[5], B_x[5];
double DT_open, DT_weekend, DT_holiday;
int nSim, nTimeStepsPerDay[5], nSPXExp, nGrid_MC, nThreads;
int nKSPXMaxDim, iTest[MAX_NUM_THREADS];
double dSeeds_Threads[MAX_NUM_THREADS][6];
int nKSPX, *SPX_Ind, *Opt_Exp_Ind;
int nTotalDays, iT_Exp, *iTradeDayInd, nSPXOpt;
double *DailyDiscreteDiv, *DailyDiscreteDY, *DailyDivYield;

HANDLE threads[MAX_NUM_THREADS];
FILE *fout;

/* create thread argument struct for Run_Thread() */
typedef struct _thread_data_t {
	int tid;
	//	double stuff;
} thread_data_t;

//int _tmain(int argc, _TCHAR* argv[])
int main()
{
	const unsigned int im1 = 4294967087;
	const unsigned int im2 = 4294944443;
	short ifail = 0;
	double *A, P0, v0, PV_Div, PV_DivIn, dSeed_Init[6];
	int i, j, k;
	int nTotalDaysPlus, nTimeStepsPerDayIn;
	double *Discount, *DiscreteDiv1, *DiscreteDiv2;
	int *iExDivDay1, *iExDivDay2;
	double tem, sqrtv, sqrtv4, tem2, tem3, temDiv;
	int nSimsPerPath, StepsPerPath, nDiscreteDiv1, nDiscreteDiv2;
	unsigned int **An1, **An2, **Bn1, **Bn2, iseed[6];
	unsigned int ia_kn1, ia_kn2, is1, is2;
	unsigned long long lp1, lp2, seed1[3], seed2[3], sseed1[3], sseed2[3];
	double time0, time1;
	unsigned threadID[MAX_NUM_THREADS];
	struct _timeb timebuffer;
	errno_t errcheck;
	// create a thread_data_t argument array 
	thread_data_t thr_data[MAX_NUM_THREADS];

	errcheck = fopen_s(&fout, "CheckMC_DP_Simulations.txt", "w");
	if (errcheck) printf(" CheckMC_DP_Simulations.txt not opened \n");

	fprintf(fout, " SVJump Model Using Monte Carlo Simulation in Double Precision \n");
	//	Read inputs from text file
	fprintf(fout, " Now reading input files \n");

	FILE *fin;
	errcheck = fopen_s(&fin, "SVJump_Model_MC_DP_Parameters.txt", "r");
	if (errcheck) printf(" File SVJump_Model_MC_DP_Parameters.txt not opened \n");
	//errcheck = fopen_s(&fin, "SVJump_Model_Parameters.txt", "r");
	//if (errcheck) printf(" File SVJump_Model_Parameters.txt not opened \n");

	FILE *fin2;
	errcheck = fopen_s(&fin2, "DailyData.txt", "r");
	if (errcheck) printf(" File DailyData.txt not opened \n");

	fscanf_s(fin, " %lf %lf %i %i %i %i %i %lf ", &RemTradeDate, &DT_holiday, &nSim, &nTimeStepsPerDayIn, &nThreads, &nSPXOpt, &iT_Exp, &PV_DivIn);
	fscanf_s(fin, " %i %i %i %i %i %i ", &iseed[0], &iseed[1], &iseed[2], &iseed[3], &iseed[4], &iseed[5]);

	//	printf(" Enter the number of threads to use, the number of simulations per thread, and a starting seed \n");
	printf(" Enter the number of threads to use and the number of simulations per thread, \n");
	printf("  Use negative numbers to accept defaults \n");
	cin >> i;
	cin >> j;
	//	std::cin >> i;
	//	std::cin >> j;

	if (i > 0) nThreads = i;
	if (j > 0) nSim = j;

	printf(" %f %f %i %i %i %i %i \n", RemTradeDate, DT_holiday, nSim, nTimeStepsPerDayIn,
		nThreads, nSPXOpt, iT_Exp);
	printf(" %i %i %i %i %i %i \n", iseed[0], iseed[1], iseed[2], iseed[3], iseed[4], iseed[5]);

	fprintf(fout, " %f %f %i %i %i %i %i \n", RemTradeDate, DT_holiday, nSim, nTimeStepsPerDayIn,
		nThreads, nSPXOpt, iT_Exp);
	fprintf(fout, " %i %i %i %i %i %i \n", iseed[0], iseed[1], iseed[2], iseed[3], iseed[4], iseed[5]);

	if (nThreads > MAX_NUM_THREADS) return -25;

	_ftime64_s(&timebuffer);
	time0 = timebuffer.time + timebuffer.millitm / 1000.0;

	//	Set constants to control volatility time for trading days, weekends, and holidays
	DT_open = 0.40;
	DT_weekend = 0.30;
	//	DT_holiday = 0.05 ;
	//	DT_holiday = 0.30 ;

	nSPXExp = 1;
	nTimeStepsPerDay[1] = nTimeStepsPerDayIn;
	nTimeStepsPerDay[2] = nTimeStepsPerDay[1] * (1.0 - DT_open) / DT_open;
	if (nTimeStepsPerDay[2] < 1) nTimeStepsPerDay[2] = 1;
	//	RemTradeDate = RemTradeDateIn;

	//	Use 365.245 or 366.0 if you are crossing over a leap day
	DaysPerYear = 365.0;
	TradeDaysPerYear = 252.0;

	nTotalDaysPlus = iT_Exp + 7;
	nTotalDays = iT_Exp;

	A = (double *)malloc(45 * sizeof(double));
	ForRate = (double *)malloc((nTotalDaysPlus + 1) * sizeof(double));
	DailyDiscreteDiv = (double *)malloc((nTotalDaysPlus + 1) * sizeof(double));
	DailyDiscreteDY = (double *)malloc((nTotalDaysPlus + 1) * sizeof(double));
	DailyDivYield = (double *)malloc((nTotalDaysPlus + 1) * sizeof(double));
	Discount = (double *)malloc((nTotalDaysPlus + 1) * sizeof(double));
	drift_logP = dmatrix(1, nTotalDaysPlus, 0, 4);
	iTradeDayInd = (int *)malloc((nTotalDaysPlus + 1) * sizeof(int));
	SPX_Ind = (int *)malloc((nSPXOpt + 1) * sizeof(int));
	Opt_Exp_Ind = (int *)malloc((nSPXOpt + 1) * sizeof(int));
	SPX_OptPrice = dmatrix(0, nSPXOpt, 0, nThreads);
	SPX_StdErr = dmatrix(0, nSPXOpt, 0, nThreads);
	SPX_Strike = (double *)malloc((nSPXOpt + 1) * sizeof(double));
	theta = (double *)malloc((nTotalDays + 1) * sizeof(double));
	sigma = (double *)malloc((nTotalDays + 1) * sizeof(double));
	lambdaJ = (double *)malloc((nTotalDays + 1) * sizeof(double));

	An1 = uint_matrix(0, 2, 0, 2);
	An2 = uint_matrix(0, 2, 0, 2);
	Bn1 = uint_matrix(0, 2, 0, 2);
	Bn2 = uint_matrix(0, 2, 0, 2);

	if (ifail < 0) return -5;

	for (i = 1; i <= 33; i++) {
		fscanf_s(fin, " %lf ", &A[i]);
	}

	Discount[0] = 1.0;

	j = 0;
	for (i = 1; i <= nTotalDaysPlus; i++) {
		fscanf_s(fin2, " %lf %i ", &ForRate[i], &iTradeDayInd[i]);
		//	For testing
		//		iTradeDayInd[i] = 0;
		Discount[i] = Discount[i - 1] * exp(-ForRate[i] / DaysPerYear);
		if (iTradeDayInd[i] > 3) return -25;
		if (iTradeDayInd[i] < 0) return -25;
	}

	fscanf_s(fin, " %i %i ", &nDiscreteDiv1, &nDiscreteDiv2);
	printf("  Dividends:  PVDivIn = %f, Number of Discrete Dividends %i, Number of Discrete Dividend Yields %i \n",
		PV_DivIn, nDiscreteDiv1, nDiscreteDiv2);
	fprintf(fout, "  Dividends:  PVDivIn = %f, Number of Discrete Dividends %i, Number of Discrete Dividend Yields %i \n",
		PV_DivIn, nDiscreteDiv1, nDiscreteDiv2);

	DiscreteDiv1 = (double *)malloc((nDiscreteDiv1 + 1) * sizeof(double));
	DiscreteDiv2 = (double *)malloc((nDiscreteDiv2 + 1) * sizeof(double));
	iExDivDay1 = (int *)malloc((nDiscreteDiv1 + 1) * sizeof(int));
	iExDivDay2 = (int *)malloc((nDiscreteDiv2 + 1) * sizeof(int));
	if (nDiscreteDiv1 > 0) {
		for (i = 1; i <= nDiscreteDiv1; i++) fscanf_s(fin, " %i %lf ", &iExDivDay1[i], &DiscreteDiv1[i]);
	}
	if (nDiscreteDiv2 > 0) {
		for (i = 1; i <= nDiscreteDiv2; i++) fscanf_s(fin, " %i %lf ", &iExDivDay2[i], &DiscreteDiv2[i]);
	}

	for (j = 1; j <= nSPXOpt; j++) {
		fscanf_s(fin, " %i %lf %i ", &SPX_Ind[j], &SPX_Strike[j], &Opt_Exp_Ind[j]);
	}

	i = nSim * nThreads;
	printf(" Setting up to run %i simulations for %i options, using %i simulations on each of %i threads \n",
		i, nSPXOpt, nSim, nThreads);
	fprintf(fout, " Setting up to run %i simulations for %i options, using %i simulations on each of %i threads \n",
		i, nSPXOpt, nSim, nThreads);
	fprintf(fout, " P/C Strike Expiration (0 = on open) \n");

	for (j = 1; j <= nSPXOpt; j++) {
		printf(" %i %8.1f  %i \n", SPX_Ind[j], SPX_Strike[j], Opt_Exp_Ind[j]);
		fprintf(fout, " %i %8.1f  %i \n", SPX_Ind[j], SPX_Strike[j], Opt_Exp_Ind[j]);
	}

	fprintf(fout, " Monte Carlo simulations with the following parameter values \n");
	for (j = 1; j <= 33; j++) {
		fprintf(fout, " %f ", A[j]);
		if ((j == 9) || (j == 18) || (j == 27)) fprintf(fout, "\n");
	}
	fprintf(fout, "\n");

	i = iT_Exp;
	if (iTradeDayInd[i + 1] == 0) Discount_SPX = Discount[i + 1];
	else {
		if (iTradeDayInd[i + 2] == 0) Discount_SPX = Discount[i + 2];
		else {
			if (iTradeDayInd[i + 3] == 0) Discount_SPX = Discount[i + 3];
			else {
				if (iTradeDayInd[i + 4] == 0) Discount_SPX = Discount[i + 4];
				else Discount_SPX = Discount[i + 5];
			}
		}
	}

	//    End of inputs

	/*
	Parameter inputs
	A[1]  =  kappa
	A[2]  =  rho
	A[3]  =  muJ1
	A[4]  =  sigmaJ1
	A[5]  =  muJ2
	A[6]  =  sigmaJ2
	A[7]  =  rhoJ
	A[8]  =  kappa2
	A[9]  =  sqrt(exp(theta2))
	A[10] =  sigma2
	A[11] =  kappa3
	A[12] =  sigma3
	A[13] =  kappa4
	A[14] =  sqrt(theta4)
	A[15] =  sigma4
	A[16] =  rho14
	A[17] =  rho24
	A[18] =  rho34
	A[19] =  kappa5
	A[20] =  theta5
	A[21] =  sigma5
	A[22] =  P0
	A[23] =  sqrt(v0)
	A[24] =  sqrt(exp(sm0))
	A[25] =  x30
	A[26] =  sqrt(v40)
	A[27] =  y50 = lambdaJ0
	A[28] =  lambdaJ3 (jump intensity coefficient for crash)
	A[29] =  muJ3
	A[30] =  sigmaJ3
	A[31] =  rhoJ34
	A[32] =  muJ4
	A[33] =  sigmaJ4

	*/

	kappa = A[1];
	rho = A[2];
	muJ1 = A[3];
	sigmaJ1 = A[4];
	muJ2 = A[5];
	sigmaJ2 = A[6];
	rhoJ = A[7];
	kappa2 = A[8];
	theta2 = log(A[9] * A[9]);
	sigma2 = A[10];
	kappa3 = A[11];
	sigma3 = A[12];
	kappa4 = A[13];
	theta4 = A[14] * A[14];
	sigma4 = A[15];
	rho14 = A[16];
	rho24 = A[17];
	rho34 = A[18];

	kappa5 = A[19];
	theta5 = A[20];
	sigma5 = A[21];
	P0 = A[22];
	sqrtv = A[23];
	sm0 = log(A[24] * A[24]);
	x30 = A[25];
	sqrtv4 = A[26];
	lambdaJ0 = A[27];

	v0 = sqrtv * sqrtv;
	x0 = log(v0);
	logP0 = log(P0);
	v40 = sqrtv4 * sqrtv4;

	lambdaJ3 = A[28];
	muJ3 = A[29];
	sigmaJ3 = A[30];
	rhoJ34 = A[31];
	muJ4 = A[32];
	sigmaJ4 = A[33];

	if (kappa < 0.0000001) return -101;

	nTimeStepsPerDay[0] = nTimeStepsPerDay[1];
	nTimeStepsPerDay[3] = nTimeStepsPerDay[1] * DT_weekend / DT_open;
	nTimeStepsPerDay[4] = nTimeStepsPerDay[1] * DT_holiday / DT_open;
	if (nTimeStepsPerDay[3] < 1) nTimeStepsPerDay[3] = 1;
	if (nTimeStepsPerDay[4] < 1) nTimeStepsPerDay[4] = 1;

	dt[0] = RemTradeDate / (DaysPerYear*nTimeStepsPerDay[0]);
	dt[1] = DT_open / (DaysPerYear*nTimeStepsPerDay[1]);
	dt[2] = (1.0 - DT_open) / (DaysPerYear*nTimeStepsPerDay[2]);
	dt[3] = DT_weekend / (DaysPerYear*nTimeStepsPerDay[3]);
	dt[4] = DT_holiday / (DaysPerYear*nTimeStepsPerDay[4]);
	for (j = 0; j < 5; j++) {
		sqrt_dt[j] = sqrt(dt[j]);
		B_x[j] = exp(-kappa * dt[j]);
		B_drift[j] = 1.0 - B_x[j];
	}

	muJ1exp = exp(muJ1 + rhoJ * muJ2 + 0.5*(sigmaJ1*sigmaJ1 + pow(rhoJ*sigmaJ2, 2))) - 1.0;
	muJ3exp = exp(muJ3 + rhoJ34 * muJ4 + 0.5*(sigmaJ3*sigmaJ3 + pow(rhoJ34*sigmaJ4, 2))) - 1.0;

	//	Calculate theta, sigma, and lambdaJ over time
	theta[0] = sm0;
	sigma[0] = sqrtv4;
	lambdaJ[0] = lambdaJ0;
	tem = exp(-kappa2 / DaysPerYear);
	tem2 = exp(-kappa4 / DaysPerYear);
	tem3 = exp(-kappa5 / DaysPerYear);
	for (i = 1; i <= nTotalDays; i++) {
		theta[i] = tem * theta[i - 1] + (1.0 - tem)*theta2;
		sigma[i] = sqrt(tem2*sigma[i - 1] * sigma[i - 1] + (1.0 - tem2)*theta4);
		lambdaJ[i] = tem3 * lambdaJ[i - 1] + (1.0 - tem3)*theta5;
	}

	//	Dividend treatment, extracting from vector for Div
	//		subtract daily dividend, dividend yield, or
	//		subtract PV(All Deterministic Dividends) and include
	//		PV of remaining dividends in stock price
	//
	//		The program currently produces a daily dividend yield
	//		extracted from the PV(Div's) for the SPX option
	//		expirations.  The daily vector Drift[] contains the 
	//		daily (calendar days) differential for r-d
	//
	for (i = 1; i <= iT_Exp; i++) {
		DailyDivYield[i] = 0.0;
		DailyDiscreteDiv[i] = 0.0;
		DailyDiscreteDY[i] = 0.0;
	}
	if (nDiscreteDiv1 > 0) {
		for (i = 1; i <= nDiscreteDiv1; i++) DailyDiscreteDiv[iExDivDay1[i]] = DiscreteDiv1[i];
	}
	if (nDiscreteDiv2 > 0) {
		for (i = 1; i <= nDiscreteDiv2; i++) DailyDiscreteDY[iExDivDay2[i]] = DiscreteDiv2[i];
	}

	if (PV_DivIn > 0.0) {
		//	The following lines extract a daily dividend yield (c.c.)
		//	DailyDivYield[] contains daily dividend yield for indexes like SPX 
		temDiv = -log(1.0 - PV_DivIn / P0)*DaysPerYear / iT_Exp;
		for (i = 1; i <= iT_Exp; i++) {
			DailyDivYield[i] = temDiv;
			DailyDiscreteDiv[i] = 0.0;
			DailyDiscreteDY[i] = 0.0;
		}
	}
	else {
		//	Discrete dividends must be subtracted from P0 before simulating logP
		PV_Div = 0.0;
		for (i = 1; i <= iT_Exp; i++) {
			if (DailyDiscreteDiv[i] > 0.0) PV_Div += Discount[i] * DailyDiscreteDiv[i];
		}
		if (PV_Div > 0.0) {
			P0 = P0 - PV_Div;
			if (P0 > 0.0) logP0 = log(P0);
			else return -500;
		}
		//	Convert daily discrete dividend yields to dividend yields for logP
		for (i = 1; i <= iT_Exp; i++) {
			temDiv = DailyDiscreteDY[i];
			if (temDiv > 0.0) DailyDiscreteDY[i] = -log(1.0 - temDiv);
		}
	}

	for (k = 0; k <= 2; k++) {
		for (i = 1; i <= iT_Exp; i++) drift_logP[i][k] = (ForRate[i] - DailyDivYield[i])*dt[k];
	}
	for (k = 3; k <= 4; k++) {
		for (i = 1; i <= iT_Exp; i++) drift_logP[i][k] = 0.0;
	}

	//	Place the non-trading day drift (ForRate[i]-DailyDivYield[i])/DaysPerYear in ForRate[i]
	//	Combines daily forward rate and daily dividend yield into one vector for simulations paths
	for (i = 1; i <= nTotalDays; i++) ForRate[i] = (ForRate[i] - DailyDivYield[i]) / DaysPerYear;

	//	Calculate simulations per path for this SV model
	StepsPerPath = 0;
	if (RemTradeDate > 0.0) StepsPerPath += nTimeStepsPerDay[0];
	for (i = 1; i <= iT_Exp; i++) {
		if (iTradeDayInd[i] == 0) StepsPerPath += nTimeStepsPerDay[1] + nTimeStepsPerDay[2];
		else {
			if (iTradeDayInd[i] == 2) StepsPerPath += nTimeStepsPerDay[3];
			if (iTradeDayInd[i] == 3) StepsPerPath += nTimeStepsPerDay[4];
		}
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
	printf("  finished seeds for thread 0 \n");

	if (nThreads > 1) {
		nSimsPerPath = StepsPerPath * 8;
		printf(" set up seeds for second thread, thread 1, nSimsPerPath %i \n", nSimsPerPath);
		SkipAhead_MRG32k3a(nSimsPerPath, An1, An2);
		printf(" finished with first Skip ahead \n");
		SkipAhead2_MRG32k3a(nSim, An1, An2, Bn1, Bn2);
		printf(" finished with second Skip ahead \n");

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
			printf("  finished seeds for thread %i \n", k);
		}

	}		//	end of if nThreads > 1

	if (nThreads > 1) {
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
	}

	_ftime64_s(&timebuffer);
	time1 = timebuffer.time + timebuffer.millitm / 1000.0;
	time1 = time1 - time0;

	printf("  \n");
	fprintf(fout, "  \n");
	j = nSim * nThreads;
	printf(" Monte Carlo Solutions %i simulations across %i threads \n", j, nThreads);
	fprintf(fout, " Monte Carlo Solutions %i simulations across %i threads \n", j, nThreads);

	printf(" Set up time for program: %8.3f \n", time1);
	fprintf(fout, " Set up time for program: %8.3f \n", time1);

	_ftime64_s(&timebuffer);
	time0 = timebuffer.time + timebuffer.millitm / 1000.0;

	//	Set up multi-threading here
	for (i = 0; i < nThreads; i++) {
		thr_data[i].tid = i;
		threads[i] = (HANDLE)_beginthreadex(NULL, 0, RunThread, &thr_data[i], 0, &threadID[i]);
	}
	WaitForMultipleObjects(nThreads, threads, TRUE, INFINITE);
	for (i = 0; i < nThreads; i++) CloseHandle(threads[i]);

	//	Average across the threads
	for (k = 1; k <= nSPXOpt; k++) {
		SPX_OptPrice[k][0] = 0.0;
		SPX_StdErr[k][0] = 0.0;
	}

	for (j = 1; j <= nThreads; j++) {
		for (k = 1; k <= nSPXOpt; k++) {
			SPX_OptPrice[k][0] += SPX_OptPrice[k][j];
			SPX_StdErr[k][0] += SPX_StdErr[k][j];
		}
	}

	for (k = 1; k <= nSPXOpt; k++) {
		SPX_OptPrice[k][0] = SPX_OptPrice[k][0] / nThreads;
		SPX_StdErr[k][0] = SPX_StdErr[k][0] / nThreads - SPX_OptPrice[k][0] * SPX_OptPrice[k][0];
		SPX_StdErr[k][0] = sqrt(SPX_StdErr[k][0] / (nSim*nThreads));
	}

	_ftime64_s(&timebuffer);
	time1 = timebuffer.time + timebuffer.millitm / 1000.0;
	time1 = time1 - time0;

	printf(" Simulation time: %8.3f \n", time1);
	fprintf(fout, " Simulation time: %8.3f \n", time1);

	printf(" The results \n");
	for (k = 1; k <= nSPXOpt; k++) printf(" %i  %f  %f %f \n", SPX_Ind[k], SPX_Strike[k], SPX_OptPrice[k][0], SPX_StdErr[k][0]);
	fprintf(fout, " The results \n");
	for (k = 1; k <= nSPXOpt; k++) fprintf(fout, " %i %8.1f  %f %f \n", SPX_Ind[k], SPX_Strike[k], SPX_OptPrice[k][0], SPX_StdErr[k][0]);

	//	Print results to screen and to output file

	fclose(fout);
	fclose(fin);
	fclose(fin2);

	printf("  Now free memory and close \n");

	//   free the work arrays
	free(A);
	free(ForRate);
	free(DailyDiscreteDiv);
	free(DailyDiscreteDY);
	free(DailyDivYield);
	free(Discount);
	free_dmatrix(drift_logP, 1, nTotalDaysPlus, 0, 4);
	free(iTradeDayInd);
	free(SPX_Ind);
	free(Opt_Exp_Ind);
	free_dmatrix(SPX_OptPrice, 0, nSPXOpt, 0, nThreads);
	free_dmatrix(SPX_StdErr, 0, nSPXOpt, 0, nThreads);
	free(SPX_Strike);
	free(theta);
	free(sigma);
	free(lambdaJ);
	free(DiscreteDiv1);
	free(DiscreteDiv2);
	free(iExDivDay1);
	free(iExDivDay2);
	free_uint_matrix(An1, 0, 2, 0, 2);
	free_uint_matrix(An2, 0, 2, 0, 2);
	free_uint_matrix(Bn1, 0, 2, 0, 2);
	free_uint_matrix(Bn2, 0, 2, 0, 2);

	ifail = nThreads;

	return 0;
}


unsigned __stdcall RunThread(void *param)
{
	int icheck, iThread;
	thread_data_t *data = (thread_data_t *)param;
	iThread = data->tid;
	printf("  Running thread %i from RunThread \n", iThread);
	icheck = MonteCarlo(iThread);
	iTest[iThread] = icheck;
	return 1;
}

int MonteCarlo(int jThread)
{
	int j, k;
	double *SPX_Sum, *SPX_Err;
	double SPX, temOptPrice, SPX_Open;
	double dseed[6];
	double x, bz00, bz01, dx;
	double z0, z1, z2, temz1, PrJump, temPr;
	double logP, v, sqrtv, SimJump;
	double logP_Open, x_Open;
	double lambdaJ1, Pr_lambdaJ3_dt[5];
	int jt, iDay, iSim;

	SPX_Sum = (double *)malloc((nSPXOpt + 1) * sizeof(double));
	SPX_Err = (double *)malloc((nSPXOpt + 1) * sizeof(double));

	bz00 = sqrt(1.0 - rho * rho);
	bz01 = rho;

	for (j = 1; j <= 4; j++) {
		temPr = exp(-lambdaJ3 * dt[j]);
		Pr_lambdaJ3_dt[j] = 1.0 - temPr;
	}

	//	Value the options by Monte Carlo simulation
	//	Simulate path and save logP, x, and sm for each day along path
	//	Then calculate payoffs on VIX and SPX options
	//	x_Day and sm_Day are saved at the end of the opening period

	for (k = 1; k <= nSPXOpt; k++) {
		SPX_Sum[k] = 0.0;
		SPX_Err[k] = 0.0;
	}

	for (k = 0; k < 6; k++) dseed[k] = dSeeds_Threads[jThread][k];

	printf(" Now running simulation in thread %i with seeds %12.1lf %12.1lf %12.1lf %12.1lf %12.1lf %12.1lf \n", jThread,
		dseed[0], dseed[1], dseed[2], dseed[3], dseed[4], dseed[5]);

	for (iSim = 0; iSim < nSim; iSim++) {
		x = x0;
		logP = logP0;
		if (RemTradeDate > 0.0) {
			temPr = exp(-lambdaJ3 * dt[0]);
			Pr_lambdaJ3_dt[0] = 1.0 - temPr;
			for (j = 1; j <= nTimeStepsPerDay[0]; j++) {
				sqrtv = exp(0.5*x);
				v = sqrtv * sqrtv;
				lambdaJ1 = lambdaJ[0];
				z0 = sninvdev(dseed);
				z1 = sninvdev(dseed);
				dx = sqrt_dt[0] * z1;
				temz1 = sqrt_dt[0] * bz00*z0 + bz01 * dx;
				dx = sigma[0] * dx;
				logP = logP + (drift_logP[1][0] - (0.5*v + lambdaJ1 * muJ1exp + lambdaJ3 * muJ3exp)*dt[0])
					+ sqrtv * temz1;
				x = x * B_x[0] + B_drift[0] * theta[0] + dx;
				if (lambdaJ1 > 0.0) {
					temPr = exp(-lambdaJ1 * dt[0]);
					PrJump = 1.0 - temPr;
					SimJump = rand_u01(dseed);
					if (SimJump <= PrJump) {
						z2 = muJ2 + sigmaJ2 * sninvdev(dseed);
						z1 = rhoJ * z2 + muJ1 + sigmaJ1 * sninvdev(dseed);
						x = x + z2;
						logP = logP + z1;
					}			//     end of first if on jump process
					else {
						roll_seed(dseed);
						roll_seed(dseed);
					}
				}			 //     end of if on lambdaJ1 > 0.0
				if (lambdaJ3 > 0.0) {
					SimJump = rand_u01(dseed);
					if (SimJump <= Pr_lambdaJ3_dt[0]) {
						z2 = muJ4 + sigmaJ4 * sninvdev(dseed);
						z1 = rhoJ34 * z2 + muJ3 + sigmaJ3 * sninvdev(dseed);
						x = x + z2;
						logP = logP + z1;
					}			//     end of second if on jump process	
					else {
						roll_seed(dseed);
						roll_seed(dseed);
					}
				}			//	end of if on lambdaJ3 > 0
			}            //		end of loop on j for simulation over time steps for TradeDate
		}		//	end of if on RemTradeDate

		for (iDay = 1; iDay <= iT_Exp; iDay++) {
			if (DailyDiscreteDY[iDay] > 0.0) logP = logP - DailyDiscreteDY[iDay];
			//		Check for Trading Days
			if (iTradeDayInd[iDay] == 0) {
				for (j = 1; j <= nTimeStepsPerDay[1]; j++) {
					sqrtv = exp(0.5*x);
					v = sqrtv * sqrtv;
					lambdaJ1 = lambdaJ[iDay];
					z0 = sninvdev(dseed);
					z1 = sninvdev(dseed);
					dx = sqrt_dt[1] * z1;
					temz1 = sqrt_dt[1] * bz00*z0 + bz01 * dx;
					dx = sigma[iDay] * dx;
					logP = logP + (drift_logP[iDay][1] - (0.5*v + lambdaJ1 * muJ1exp + lambdaJ3 * muJ3exp)*dt[1])
						+ sqrtv * temz1;
					x = x * B_x[1] + B_drift[1] * theta[iDay] + dx;

					if (lambdaJ1 > 0.0) {
						temPr = exp(-lambdaJ1 * dt[1]);
						PrJump = 1.0 - temPr;
						SimJump = rand_u01(dseed);
						if (SimJump <= PrJump) {
							z2 = muJ2 + sigmaJ2 * sninvdev(dseed);
							z1 = rhoJ * z2 + muJ1 + sigmaJ1 * sninvdev(dseed);
							x = x + z2;
							logP = logP + z1;
						}			//     end of first if on jump process
						else {
							roll_seed(dseed);
							roll_seed(dseed);
						}
					}			 //     end of if on lambdaJ1 > 0.0
					if (lambdaJ3 > 0.0) {
						SimJump = rand_u01(dseed);
						if (SimJump <= Pr_lambdaJ3_dt[1]) {
							z2 = muJ4 + sigmaJ4 * sninvdev(dseed);
							z1 = rhoJ34 * z2 + muJ3 + sigmaJ3 * sninvdev(dseed);
							x = x + z2;
							logP = logP + z1;
						}			//     end of second if on jump process		
						else {
							roll_seed(dseed);
							roll_seed(dseed);
						}
					}			//	end of if on lambdaJ3 > 0


				}            //		end of loop on j for simulation over time steps for opening period
				logP_Open = logP;
				x_Open = x;

				for (j = 1; j <= nTimeStepsPerDay[2]; j++) {
					sqrtv = exp(0.5*x);
					v = sqrtv * sqrtv;
					lambdaJ1 = lambdaJ[iDay];
					z0 = sninvdev(dseed);
					z1 = sninvdev(dseed);
					dx = sqrt_dt[2] * z1;
					temz1 = sqrt_dt[2] * bz00*z0 + bz01 * dx;
					dx = sigma[iDay] * dx;
					logP = logP + (drift_logP[iDay][2]
						- (0.5*v + lambdaJ1 * muJ1exp + lambdaJ3 * muJ3exp)*dt[2]) + sqrtv * temz1;
					x = x * B_x[2] + B_drift[2] * theta[iDay] + dx;

					if (lambdaJ1 > 0.0) {
						temPr = exp(-lambdaJ1 * dt[2]);
						PrJump = 1.0 - temPr;
						SimJump = rand_u01(dseed);
						if (SimJump <= PrJump) {
							z2 = muJ2 + sigmaJ2 * sninvdev(dseed);
							z1 = rhoJ * z2 + muJ1 + sigmaJ1 * sninvdev(dseed);
							x = x + z2;
							logP = logP + z1;
						}			//     end of first if on jump process
						else {
							roll_seed(dseed);
							roll_seed(dseed);
						}
					}			 //     end of if on lambdaJ1 > 0.0
					if (lambdaJ3 > 0.0) {
						SimJump = rand_u01(dseed);
						if (SimJump <= Pr_lambdaJ3_dt[2]) {
							z2 = muJ4 + sigmaJ4 * sninvdev(dseed);
							z1 = rhoJ34 * z2 + muJ3 + sigmaJ3 * sninvdev(dseed);
							x = x + z2;
							logP = logP + z1;
						}			//     end of second if on jump process		
						else {
							roll_seed(dseed);
							roll_seed(dseed);
						}
					}			//	end of if on lambdaJ3 > 0


				}            //		end of loop on j for simulation over time steps for period from after opening to close
			}			//	end of if for trading day
			else {
				logP_Open = logP;
				//	ForRate[iDay] includes the interest rate and the dividend yield, but no volatility
				logP = logP + ForRate[iDay];
				if (iTradeDayInd[iDay] > 1) {
					jt = iTradeDayInd[iDay] + 1;
					for (j = 1; j <= nTimeStepsPerDay[jt]; j++) {
						sqrtv = exp(0.5*x);
						v = sqrtv * sqrtv;
						lambdaJ1 = lambdaJ[iDay];
						z0 = sninvdev(dseed);
						z1 = sninvdev(dseed);
						dx = sqrt_dt[jt] * z1;
						temz1 = sqrt_dt[jt] * bz00*z0 + bz01 * dx;
						dx = sigma[iDay] * dx;
						//	drift_logP[iDay][jt] = 0.0 for weekends and holidays
						logP = logP + (drift_logP[iDay][jt]
							- (0.5*v + lambdaJ1 * muJ1exp + lambdaJ3 * muJ3exp)*dt[jt]) + sqrtv * temz1;
						x = x * B_x[jt] + B_drift[jt] * theta[iDay] + dx;

						if (lambdaJ1 > 0.0) {
							temPr = exp(-lambdaJ1 * dt[jt]);
							PrJump = 1.0 - temPr;
							SimJump = rand_u01(dseed);
							if (SimJump <= PrJump) {
								z2 = muJ2 + sigmaJ2 * sninvdev(dseed);
								z1 = rhoJ * z2 + muJ1 + sigmaJ1 * sninvdev(dseed);
								x = x + z2;
								logP = logP + z1;
							}			//     end of first if on jump process
							else {
								roll_seed(dseed);
								roll_seed(dseed);
							}
						}			 //     end of if on lambdaJ1 > 0.0
						if (lambdaJ3 > 0.0) {
							SimJump = rand_u01(dseed);
							if (SimJump <= Pr_lambdaJ3_dt[jt]) {
								z2 = muJ4 + sigmaJ4 * sninvdev(dseed);
								z1 = rhoJ34 * z2 + muJ3 + sigmaJ3 * sninvdev(dseed);
								x = x + z2;
								logP = logP + z1;
							}			//     end of second if on jump process	
							else {
								roll_seed(dseed);
								roll_seed(dseed);
							}
						}			//	end of if on lambdaJ3 > 0

					}            //		end of loop on j for simulation over time steps for weekend/holiday
				}		//	end of if (iTradeDayInd[iDay] > 1)

				x_Open = x;

			}			//	end of else for trading day

		}			//	end of loop on iDay

		//		Calculate the payoffs for the SPX option expiration
		SPX = exp(logP);
		SPX_Open = exp(logP_Open);
		for (k = 1; k <= nSPXOpt; k++) {
			if (Opt_Exp_Ind[k] == 0) {
				if (SPX_Ind[k] == 1) {
					if (SPX_Open > SPX_Strike[k]) temOptPrice = SPX_Open - SPX_Strike[k];
					else temOptPrice = 0.0;
				}
				else {
					if (SPX_Strike[k] > SPX_Open) temOptPrice = SPX_Strike[k] - SPX_Open;
					else temOptPrice = 0.0;
				}
				if (temOptPrice > 0.0) {
					SPX_Sum[k] += temOptPrice;
					SPX_Err[k] += temOptPrice * temOptPrice;
				}
			}
			else {
				if (SPX_Ind[k] == 1) {
					if (SPX > SPX_Strike[k]) temOptPrice = SPX - SPX_Strike[k];
					else temOptPrice = 0.0;
				}
				else {
					if (SPX_Strike[k] > SPX) temOptPrice = SPX_Strike[k] - SPX;
					else temOptPrice = 0.0;
				}
				if (temOptPrice > 0.0) {
					SPX_Sum[k] += temOptPrice;
					SPX_Err[k] += temOptPrice * temOptPrice;
				}
			}
		}

	}                //     end of loop on iSim for independent simulations

	printf(" Finished with thread %i \n", jThread);
	printf(" In thread %i the ending seed valuess %12.1lf %12.1lf %12.1lf %12.1lf %12.1lf %12.1lf \n", jThread,
		dseed[0], dseed[1], dseed[2], dseed[3], dseed[4], dseed[5]);

	for (k = 1; k <= nSPXOpt; k++) {
		SPX_OptPrice[k][jThread + 1] = Discount_SPX * SPX_Sum[k] / nSim;
		SPX_Err[k] = Discount_SPX * Discount_SPX * SPX_Err[k] / nSim;
		SPX_StdErr[k][jThread + 1] = SPX_Err[k];
	}

	//		Release memory allocation

	free(SPX_Sum);
	free(SPX_Err);

	return 1;
}

#undef MAX_NUM_THREADS
#undef NR_END
#undef FREE_ARG
#undef IM
#undef IQ
#undef IR
#undef I2_28 
#undef I2_29 
#undef I2_30
#undef I2_31 
#undef IM13  
#undef IQ13 
#undef pi
#undef dsqr2 
#undef dsqr2pi 
#undef sqrpi
#undef dsqrpi
#undef half_ln_2pi 
#undef I13_6 
#undef I13_7 

/* (C)Copr. 1986-92 Numerical Recipes Software G2v#X):K. */