//	GPU device code:  Header Files

#include <cuda.h>
#include <curand_kernel.h>

#include <stdio.h>

__device__ inline float MRG32k3a(unsigned int seed[]);

__device__ inline float MRG32k3a_dp(double dseed[]);

__device__ inline void SimulateBodyGPU1(int iSim, int nSim, unsigned int *seeds, 
			float *d_MRG_rng);

__device__ inline void SimulateBodyGPU2(int iSim, int nSim, unsigned int *seeds,
	float *d_MRG_rng);

__global__ void SimulatePathGPU(
	int n, int nSim, int ind, unsigned int *d_seeds, float *d_MRG_rng
	);

__device__ inline void SimulateBodyGPU(int iSim, int n, int nSim, unsigned int *seeds,
	float *d_MRG_rng, curandStateMRG32k3a *state);

__global__ void Test_curand_MRG32k3a(
	int n, int nSim, unsigned int *d_seeds, float *d_MRG_rng, curandStateMRG32k3a *state
	);

__global__ void setup_kernel(int nSim, curandStateMRG32k3a *state);