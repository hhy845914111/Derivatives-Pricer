//	GPU device code:  Header Files

__device__ inline float sninvdev(double dseed[]);

__device__ inline void SimulateBodyGPU(
	int iSim, int nSim, int nMat, int nSteps, double dt, float y1const,
	float r0, float y10, float y20, int *nMat1, int *nMat2,
	unsigned int *seeds, float *AConst, float *B0, float *B1, float *B2,
	float *FwdSpread, float *temexp, float *lamsig, float *sigz,
	double *SimDiscount, double *SimLIBOR);

__global__ void SimulatePathGPU(
	int nSim, int nMat, int nSteps, double dt, float y1const, float r0,
	float y10, float y20, int *d_nMat1, int *d_nMat2,
	unsigned int *d_seeds, float *d_AConst, float *d_B0, float *d_B1,
	float *d_B2, float *d_FwdSpread, float *d_temexp,
	float *d_lamsig, float *d_sigz, double *d_SimDiscount, double *d_SimLIBOR);