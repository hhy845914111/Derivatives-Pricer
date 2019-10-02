//	GPU device code
//	Test MRG32k3a
//

//#include <cuda.h>
//#include <curand_kernel.h>
#include "Test_MRG32k3a_Normal_kernel.cuh"

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

////////////////////////////////////////////////////////////////////////////////
//	Process an array of nSim simulations of the simulation model on GPU
//	with n simulations per path, using call to curand function
////////////////////////////////////////////////////////////////////////////////

__global__ void Test_curand_MRG32k3a(
	int n, int nSim, float r, float sigma, float dt, float log_S0, unsigned int *d_seeds, 
	float *d_MRG_rng, curandStateMRG32k3a *state
	)
{
	int iSim;
	//Thread index
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	//No matter how small is execution grid or how large nSim is,
	//exactly nSim paths will be processed with perfect memory coalescing
	iSim = tid;
	if (iSim < nSim) {
		int i;
		//		unsigned long long n_skip;
		float f, log_S, sq_dt;
		curandStateMRG32k3a localState = state[iSim];

//	Do the skip ahead on host and pass the seeds, instead of using curand skipahead
//	This is actually faster
		localState.s1[0] = d_seeds[iSim * 6 + 2];
		localState.s1[1] = d_seeds[iSim * 6 + 1];
		localState.s1[2] = d_seeds[iSim * 6];
		localState.s2[0] = d_seeds[iSim * 6 + 5];
		localState.s2[1] = d_seeds[iSim * 6 + 4];
		localState.s2[2] = d_seeds[iSim * 6 + 3];

		log_S = log_S0;
		sq_dt = sqrtf(dt);
		for (i = 0; i < n; i++) {
			f = curand_normal(&localState);
			log_S += r*dt + sigma*sq_dt*f;
		}

		d_MRG_rng[iSim] = expf(log_S);

	}

}

__global__ void Test_curand_MRG32k3a_NormInv(
	int n, int nSim, float r, float sigma, float dt, float log_S0, unsigned int *d_seeds, 
	float *d_MRG_rng, curandStateMRG32k3a *state
	)
{
	int iSim;
	//Thread index
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	//No matter how small is execution grid or how large nSim is,
	//exactly nSim paths will be processed with perfect memory coalescing
	iSim = tid;
	if (iSim < nSim) {
		int i;
		float f, log_S, sq_dt;
		curandStateMRG32k3a localState = state[iSim];

		localState.s1[0] = d_seeds[iSim * 6 + 2];
		localState.s1[1] = d_seeds[iSim * 6 + 1];
		localState.s1[2] = d_seeds[iSim * 6];
		localState.s2[0] = d_seeds[iSim * 6 + 5];
		localState.s2[1] = d_seeds[iSim * 6 + 4];
		localState.s2[2] = d_seeds[iSim * 6 + 3];

		log_S = log_S0;
		sq_dt = sqrtf(dt);
		for (i = 0; i < n; i++) {
			f = curand_uniform(&localState);
			log_S += r*dt + sigma*sq_dt*normcdfinvf(f);
		}
		d_MRG_rng[iSim] = expf(log_S);

	}
}



__global__ void setup_kernel(int nSim, curandStateMRG32k3a *state)
{
	int iSim;
//	int id = threadIdx.x + blockIdx.x * 64;
	/* Each thread gets same seed, a different sequence
	number, no offset */

	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	iSim = tid;
	if (iSim < nSim) {
		curand_init(0, iSim, 0, &state[iSim]);
	}

}


__global__ void Test_MRG32k3a_NormInv(
	int n, int nSim, float r, float sigma, float dt, float log_S0, unsigned int *d_seeds,
	float *d_MRG_rng
)
{
	int iSim;
	//Thread index
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	//No matter how small is execution grid or how large nSim is,
	//exactly nSim paths will be processed with perfect memory coalescing
	iSim = tid;
	if (iSim < nSim) {
		int i;
		float f, log_S, sq_dt;
//		double dseed[6];
		unsigned int iseed[6];

		for (i = 0; i < 6; i++) {
//			dseed[i] = d_seeds[iSim * 6 + i];
			iseed[i] = d_seeds[iSim * 6 + i];
		}

		log_S = log_S0;
		sq_dt = sqrtf(dt);
		for (i = 0; i < n; i++) {
//			f = MRG32k3a_dp(dseed);
			f = MRG32k3a(iseed);
			log_S += r * dt + sigma * sq_dt*normcdfinvf(f);
		}
		d_MRG_rng[iSim] = expf(log_S);

	}
}

