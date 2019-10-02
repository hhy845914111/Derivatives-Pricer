//	GPU device code
//	Test MRG32k3a
//

#include "Test_MRG32k3a_kernel.cuh"

#include <stdio.h>

__device__ inline float MRG32k3a(unsigned int seed[])
{
	const unsigned int im1 = 4294967087;
	const unsigned int im2 = 4294944443;
	const unsigned int ia12 = 1403580;
	const unsigned int ia13n = 810728;
	const unsigned int ia21 = 527612;
	const unsigned int ia23n = 1370589;
	const double  norm = 2.328306549295728e-10;
//	Using norm in single precision does not improve performance on GPU
//	const float  norm = 2.328306549295728e-10;
	float f;
	long long lp1, lp2;
	int mp1, mp2;

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

__device__ inline float MRG32k3a_dp(double dseed[])
{
	//	This code is an exact copy of the C code in L'Ecuyer (Operations Research 1999)
	const double norm = 2.328306549295728e-10;
	const double m1 = 4294967087.0;
	const double m2 = 4294944443.0;
	const double a12 = 1403580.0;
	const double a13n = 810728.0;
	const double a21 = 527612.0;
	const double a23n = 1370589.0;
	int k;
	double p1, p2;
	float ans;

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

	if (p1 <= p2) ans = ((p1 - p2 + m1)*norm);
	else ans = ((p1 - p2)*norm);

	return ans;
}


///////////////////////////////////////////////////////////////////////////////
// Main functions for simulating random numbers
///////////////////////////////////////////////////////////////////////////////
__device__ inline void SimulateBodyGPU1(int iSim, int n, int nSim, unsigned int *seeds,
					float *d_MRG_rng)
{
	unsigned int iseed[6];
	int i;
	float f;

	for (i = 0; i < 6; i++) iseed[i] = seeds[iSim * 6 + i];
	for (i = 0; i < n; i++) {
		f = MRG32k3a(iseed);
	}

	d_MRG_rng[iSim] = f;

}

__device__ inline void SimulateBodyGPU2(int iSim, int n, int nSim, unsigned int *seeds,
	float *d_MRG_rng)
{
	int i;
	float f;
	double f2, dseed[6];

	for (i = 0; i < 6; i++) dseed[i] = seeds[iSim * 6 + i];

	for (i = 0; i < n; i++) {
		f2 = MRG32k3a_dp(dseed);
	}

	f = f2;
	d_MRG_rng[iSim] = f;

}


////////////////////////////////////////////////////////////////////////////////
//	Process an array of nSim simulations of the simulation model on GPU
//	with n simulations per path, using Cuda C code
////////////////////////////////////////////////////////////////////////////////
__global__ void SimulatePathGPU(
	int n, int nSim, int ind, unsigned int *d_seeds, float *d_MRG_rng
	)
{
	int iSim = blockDim.x * blockIdx.x + threadIdx.x;
	//Thread index
	if (iSim < nSim) {

		if (ind == 2) SimulateBodyGPU2(iSim, n, nSim, d_seeds, d_MRG_rng);
		else SimulateBodyGPU1(iSim, n, nSim, d_seeds, d_MRG_rng);

	}
}


////////////////////////////////////////////////////////////////////////////////
//	Process an array of nSim simulations of the simulation model on GPU
//	with n simulations per path, using call to curand function
////////////////////////////////////////////////////////////////////////////////

__global__ void Test_curand_MRG32k3a(
	int n, int nSim, unsigned int *d_seeds, float *d_MRG_rng, curandStateMRG32k3a *state
	)
{
	int iSim = blockDim.x * blockIdx.x + threadIdx.x;
	//Thread index
	if (iSim < nSim) {
		int i;
		unsigned long long n_skip;
		float f;
		curandStateMRG32k3a localState = state[iSim];

/*
//	Do the skip ahead on host and pass the seeds, instead of using curand skipahead
		localState.s1[0] = d_seeds[iSim * 6 + 2];
		localState.s1[1] = d_seeds[iSim * 6 + 1];
		localState.s1[2] = d_seeds[iSim * 6];
		localState.s2[0] = d_seeds[iSim * 6 + 5];
		localState.s2[1] = d_seeds[iSim * 6 + 4];
		localState.s2[2] = d_seeds[iSim * 6 + 3];
*/

//	This runs, but requires extra compute time for skip ahead
		localState.s1[0] = d_seeds[2];
		localState.s1[1] = d_seeds[1];
		localState.s1[2] = d_seeds[0];
		localState.s2[0] = d_seeds[5];
		localState.s2[1] = d_seeds[4];
		localState.s2[2] = d_seeds[3];
		n_skip = n*iSim;
		skipahead(n_skip, &localState);
	
		for (i = 0; i < n; i++) {
			f = curand_uniform(&localState);
		}

		d_MRG_rng[iSim] = f;

	}

}


__global__ void setup_kernel(int nSim, curandStateMRG32k3a *state)
{
	int iSim = threadIdx.x + blockIdx.x * blockDim.x;
// Each thread gets same seed, a different sequence number, no offset
	if (iSim < nSim) {
		curand_init(0, iSim, 0, &state[iSim]);
	}
}
