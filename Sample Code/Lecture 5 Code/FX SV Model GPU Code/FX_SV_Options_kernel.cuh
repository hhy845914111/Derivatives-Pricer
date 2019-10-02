//	GPU device code:  Header Files

__device__ inline float sninvdev(double dseed[]);

__global__ void SimulatePathGPU(
	int nSim, int nMat, int nSteps, float dt, float sqdt, float FX0, float v0,
	float BCoef, float ACoef, float sigCoef, float rho, unsigned int *seeds, 
	int *Maturity, float *rd, float *rf, float *d_SimFX);