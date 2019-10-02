//	GPU device code:  Header Files

__global__ void ExpirationValuesGPU(
	int ny_grid, int OptType, int nMat2, float Strike, float AFut, float B0Fut,
	float B1Fut, float B2Fut, float FwdSpread, float *r_grid, float*y1_grid,
	float *y2_grid, float *V2);

__global__ void ExplicitSolution1_GPU(int nr_grid, int jmin, int jmax, int kmin,
	int kmax, int j_y1min, int j_y1max, int k_y2min, int k_y2max, float kappa2,
	float lamsig2, float *rgrid, float *y1grid, float *y2grid, float dtdy2,
	float *V1, float *V2);

__global__ void ExplicitSolution2_GPU(int nr_grid, int jmin, int jmax, int kmin,
	int kmax, int j_y1min, int j_y1max, int k_y2min, int k_y2max, float kappa1,
	float theta1, float lamsig1, float *rgrid, float *y1grid, float *y2grid,
	float dtdy1, float *V1, float *V2);

__global__ void ImplicitSolution_GPU(
	int nr_grid, int jmin, int jmax, int kmin, int kmax, float dt, float dr,
	float kappa0, float lamsig0, float *rgrid, float *y1grid, float *y2grid,
	float dtdr, float *wkVa, float *V1, float *V2);

__global__ void Check_EarlyExercise_GPU(
	int ny_grid, int jmin, int jmax, int kmin, int kmax, int nMat2,
	int OptType, float Strike, float AFut, float B0Fut, float B1Fut, float B2Fut,
	float FwdSpread, float *r_grid, float *y1_grid,
	float *y2_grid, float *V2);

__global__ void Check_EarlyExercise2_GPU(
	int ny_grid, int jmin, int jmax, int kmin, int kmax, int nMat2,
	int OptType, float Strike, float AFut, float B0Fut, float B1Fut, float B2Fut,
	float FwdSpread, float *r_grid, float *y1_grid,
	float *y2_grid, float *V2);

//__global__ void Move_V1to_V2_GPU(int nr_grid, int jmin, int jmax, int kmin,
//	int kmax, float *V1, float *V2);
