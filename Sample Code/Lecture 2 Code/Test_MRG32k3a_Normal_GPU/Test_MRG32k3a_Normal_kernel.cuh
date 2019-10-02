//	GPU device code:  Header Files

#include <cuda.h>
#include <curand_kernel.h>

#include <stdio.h>
//#include <windows.h>

//__device__ inline float rand_u01GPU(unsigned int &ib1, unsigned int &ib2);

//__device__ inline float sninvdev(unsigned int &ib1, unsigned int &ib2);

//__device__ inline void roll_seed(unsigned int &ib1, unsigned int &ib2);

__device__ inline float MRG32k3a(unsigned int seed[]);

__device__ inline float MRG32k3a_dp(double dseed[]);

__global__ void Test_curand_MRG32k3a(
	int n, int nSim, float r, float sigma, float dt, float log_S0, unsigned int *d_seeds, float *d_MRG_rng, curandStateMRG32k3a *state
	);

__global__ void Test_curand_MRG32k3a_NormInv(
	int n, int nSim, float r, float sigma, float dt, float log_S0, unsigned int *d_seeds, float *d_MRG_rng, curandStateMRG32k3a *state
	);

__global__ void setup_kernel(int nSim, curandStateMRG32k3a *state);

__global__ void Test_MRG32k3a_NormInv(
	int n, int nSim, float r, float sigma, float dt, float log_S0, unsigned int *d_seeds,
	float *d_MRG_rng);