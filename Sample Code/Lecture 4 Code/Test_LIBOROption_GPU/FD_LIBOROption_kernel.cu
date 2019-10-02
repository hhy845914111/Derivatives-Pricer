//	GPU device code
//	LIBOR options, 3Factor Hull-White Model, Finite Difference
//	Version 1, January 2019

#include <stdio.h>

////////////////////////////////////////////////////////////////////////////////
//Process the calculations at expiration on GPU
////////////////////////////////////////////////////////////////////////////////
__global__ void ExpirationValuesGPU(
	int ny_grid, int OptType, int nMat2, float Strike, float AFut, float B0Fut, 
	float B1Fut, float B2Fut, float FwdSpread, float *r_grid, float*y1_grid, 
	float *y2_grid, float *V2)
{
	int ijk;
	float tem;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	if ((i < (ny_grid + 1)) && (j < (ny_grid + 1))  && (k < (ny_grid + 1))) {
				ijk = i * (ny_grid+1)*(ny_grid+1) + j * (ny_grid+1) + k;
				tem = (exp(AFut + B0Fut * r_grid[i] + B1Fut * y1_grid[j] + B2Fut * y2_grid[k]) - 1.0)*360.0 / nMat2;
				tem = tem + FwdSpread;
				if (OptType == 1) V2[ijk] = max(0.0, tem - Strike);
				else V2[ijk] = max(0.0, Strike - tem);
	}

}


////////////////////////////////////////////////////////////////////////////////
//Process the explicit solutions on GPU
////////////////////////////////////////////////////////////////////////////////

__global__ void ExplicitSolution1_GPU(int nr_grid, int jmin, int jmax, int kmin, 
	int kmax, int j_y1min, int j_y1max, int k_y2min, int k_y2max, float kappa2, 
	float lamsig2, float *rgrid, float *y1grid, float *y2grid, float dtdy2, 
	float *V1, float *V2)
{
//	This function calculates explicit solution in y2 dimension
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	if (i < nr_grid && j < (1 + jmax - jmin) && k < (1 + kmax - kmin)) {

		int ijk, jp, kp;
		float drift, cfu, cf0, cfd, cfuu, cfdd, tem1_3, tem2_3;

		jp = jmin + j;
		kp = kmin + k;
		ijk = i * (nr_grid + 1)*(nr_grid + 1) + jp * (nr_grid + 1) + kp;
		tem1_3 = 1.0 / 3.0;
		tem2_3 = 2.0 / 3.0;
		drift = (-kappa2 * y2grid[kp] - lamsig2)*dtdy2;
		if ((kp > k_y2min) && (kp < k_y2max)) {
			if ((kp > k_y2min) && (kp < k_y2max)) {
				cf0 = tem1_3;
				cfu = cf0 + 0.5*drift;
				cfd = cf0 - 0.5*drift;
				V1[ijk] = cfu * V2[ijk + 1] + cf0 * V2[ijk] + cfd * V2[ijk - 1];
			}
			else {
				if (kp <= k_y2min) {
		//				printf("   Computing on y2 min  k = %i   \n", k);
					cf0 = 1.0 + tem1_3 - drift;
					cfu = drift - tem2_3;
					cfuu = tem1_3;
					V1[ijk] = cf0 * V2[ijk] + cfu * V2[ijk + 1] + cfuu * V2[ijk + 2];
				}
				else {
		//				printf("   Computing on y2 max  k = %i   \n", k);
					cf0 = 1.0 + tem1_3 + drift;
					cfd = -drift - tem2_3;
					cfdd = tem1_3;
					V1[ijk] = cf0 * V2[ijk] + cfd * V2[ijk - 1] + cfdd * V2[ijk - 2];
				}
			}
		}

	}	//	end of GPU if on i, j, k

}


__global__ void ExplicitSolution2_GPU(int nr_grid, int jmin, int jmax, int kmin, 
		int kmax, int j_y1min, int j_y1max, int k_y2min, int k_y2max, float kappa1, 
		float theta1, float lamsig1, float *rgrid, float *y1grid, float *y2grid,
		float dtdy1, float *V1, float *V2)
{
//	This function calculates explicit solution in y1 dimension
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	if (i < nr_grid && j < (1 + jmax - jmin) && k < (1 + kmax - kmin)) {

		int ijk, jp, kp, ijk1u, ijk1d, ijk2u, ijk2d;
		float drift, cfu, cf0, cfd, cfuu, cfdd, tem1_3, tem2_3;

		jp = jmin + j;
		kp = kmin + k;
		ijk = i * (nr_grid + 1)*(nr_grid + 1) + jp * (nr_grid + 1) + kp;
		tem1_3 = 1.0 / 3.0;
		tem2_3 = 2.0 / 3.0;
		drift = (kappa1 * (theta1 - y1grid[jp]) - lamsig1)*dtdy1;

		if ((jp > j_y1min) && (jp < j_y1max)) {
			cf0 = tem1_3;
			cfu = cf0 + 0.5*drift;
			cfd = cf0 - 0.5*drift;
			ijk1u = i * (nr_grid + 1)*(nr_grid + 1) + (jp + 1) * (nr_grid + 1) + kp;
			ijk1d = i * (nr_grid + 1)*(nr_grid + 1) + (jp - 1) * (nr_grid + 1) + kp;
			V1[ijk] = cfu * V2[ijk1u] + cf0 * V2[ijk] + cfd * V2[ijk1d];
		}
		else {
			if (jp <= j_y1min) {
				//				printf("   Computing on y1 min  jp = %i   \n", jp);
				cf0 = 1.0 + tem1_3 - drift;
				cfu = drift - tem2_3;
				cfuu = tem1_3;
				ijk1u = i * (nr_grid + 1)*(nr_grid + 1) + (jp + 1) * (nr_grid + 1) + kp;
				ijk2u = i * (nr_grid + 1)*(nr_grid + 1) + (jp + 2) * (nr_grid + 1) + kp;
				V1[ijk] = cf0 * V2[ijk] + cfu * V2[ijk1u] + cfuu * V2[ijk2u];
			}
			else {
				//				printf("   Computing on y1 max  jp = %i   \n", jp);
				cf0 = 1.0 + tem1_3 + drift;
				cfd = -drift - tem2_3;
				cfdd = tem1_3;
				ijk1d = i * (nr_grid + 1)*(nr_grid + 1) + (jp - 1) * (nr_grid + 1) + kp;
				ijk2d = i * (nr_grid + 1)*(nr_grid + 1) + (jp - 2) * (nr_grid + 1) + kp;
				V1[ijk] = cf0 * V2[ijk] + cfd * V2[ijk1d] + cfdd * V2[ijk2d];
			}
		}

	}

}	//	end of ExplicitSolution2_GPU


////////////////////////////////////////////////////////////////////////////////
//Process the implicit solutions on GPU
////////////////////////////////////////////////////////////////////////////////

__global__ void ImplicitSolution_GPU(
		int nr_grid, int jmin, int jmax, int kmin, int kmax, float dt, float dr, 
		float kappa0, float lamsig0, float *rgrid, float *y1grid, float *y2grid,
		float dtdr, float *wkVa, float *V1, float *V2)
{
//	wk workspace arrays arrays need to be much larger
	//	This function calculates implicit solution in the r dimension
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;

	if (j < (1 + jmax - jmin) && k < (1 + kmax - kmin)) {

		int i, ijk, ijk2, jp, kp, jk, ii;
		float a, b, c, drift_r, tem, tem1_3, tem2_3;

		jp = jmin + j;
		kp = kmin + k;
		jk = jp * (nr_grid + 1) + kp;
		tem1_3 = 1.0 / 3.0;
		tem2_3 = 2.0 / 3.0;

//	You can replace wkVb with V2 and eliminate memory requirement for wkVb
		i = 0;
		ijk = i * (nr_grid + 1)*(nr_grid + 1) + jk;
		drift_r = (kappa0 * (y1grid[jp] + y2grid[kp] - rgrid[i]) - lamsig0)*dtdr;
		b = 1.0 + rgrid[i] * dt + tem2_3 + (0.5*drift_r - tem1_3)*(1.0 + rgrid[i] * dt) / (1.0 + (rgrid[0] - dr)*dt);
		c = -0.5*drift_r - tem1_3;
		wkVa[ijk] = c / b;
		V2[ijk] = V2[ijk] / b;
		for (i = 1; i < nr_grid; i++) {
			ijk = i * (nr_grid + 1)*(nr_grid + 1) + jk;
			ijk2 = (i - 1) * (nr_grid + 1)*(nr_grid + 1) + jk;
			drift_r = (kappa0 * (y1grid[jp] + y2grid[kp] - rgrid[i]) - lamsig0)*dtdr;
			b = 1.0 + rgrid[i] * dt + tem2_3;
			a = 0.5*drift_r - tem1_3;
			c = -0.5*drift_r - tem1_3;
			tem = (b - a * wkVa[ijk2]);
			wkVa[ijk] = c / tem;
			V2[ijk] = (V2[ijk] - a * V2[ijk2]) / tem;
		}
		i = nr_grid;
		ijk = i * (nr_grid + 1)*(nr_grid + 1) + jp * (nr_grid + 1) + kp;
		ijk2 = (i - 1) * (nr_grid + 1)*(nr_grid + 1) + jk;
		drift_r = (kappa0 * (y1grid[jp] + y2grid[kp] - rgrid[i]) - lamsig0)*dtdr;
		b = 1.0 + rgrid[i] * dt + tem2_3 + (-0.5*drift_r - tem1_3)*(1.0 + rgrid[i] * dt) / (1.0 + (rgrid[nr_grid] + dr)*dt);
		a = 0.5*drift_r - tem1_3;
		tem = (b - a * wkVa[ijk2]);
		V2[ijk] = (V2[ijk] - a * V2[ijk2]) / tem;

		V1[ijk] = V2[ijk];
		for (ii = 1; ii <= nr_grid; ii++) {
			i = i - 1;
			ijk = i * (nr_grid + 1)*(nr_grid + 1) + jp * (nr_grid + 1) + kp;
			ijk2 = (i + 1) * (nr_grid + 1)*(nr_grid + 1) + jk;
			V1[ijk] = V2[ijk] - wkVa[ijk] * V1[ijk2];
		}

//	Move V1 values back to V2
		for (i = 0; i <= nr_grid; i++) {
			ijk = i * (nr_grid + 1)*(nr_grid + 1) + jp * (nr_grid + 1) + kp;
			V2[ijk] = V1[ijk];
		}

	}	//	end of GPU if on j and k

}	//	end of ImplicitFunction_GPU


__global__ void Check_EarlyExercise_GPU(
	int ny_grid, int jmin, int jmax, int kmin, int kmax, int nMat2,
	int OptType, float Strike, float AFut, float B0Fut, float B1Fut, float B2Fut, 
	float FwdSpread, float *r_grid, float *y1_grid, 
	float *y2_grid, float *V2)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;

	if (j < (1 + jmax - jmin) && k < (1 + kmax - kmin)) {
		int i, ijk, jp, kp, icount;
		float tem, temopt;
		kp = kmin + k;
		jp = jmin + j;
		if (OptType == 2) {
			//	put options on 3m LIBOR futures rate
			icount = 0;
			for (i = 0; i <= ny_grid; i++) {
				if (icount < 50) {
					ijk = i * (ny_grid + 1)*(ny_grid + 1) + jp * (ny_grid + 1) + kp;
					tem = (exp(AFut + B0Fut * r_grid[i] + B1Fut * y1_grid[jp] + B2Fut * y2_grid[kp]) - 1.0)*360.0 / nMat2;
					tem = tem + FwdSpread;
					temopt = max(0.0, Strike - tem);
					if (V2[ijk] < temopt) V2[ijk] = temopt;
					else icount += 1;
				}
			}
		}
		else {
			// call options on 3m LIBOR futures rate
			icount = 0;
			for (i = ny_grid; i >= 0; i--) {
				if (icount < 50) {
					ijk = i * (ny_grid + 1)*(ny_grid + 1) + jp * (ny_grid + 1) + kp;
					tem = (exp(AFut + B0Fut * r_grid[i] + B1Fut * y1_grid[jp] + B2Fut * y2_grid[kp]) - 1.0)*360.0 / nMat2;
					tem = tem + FwdSpread;
					temopt = max(0.0, tem - Strike);
					if (V2[ijk] < temopt) V2[ijk] = temopt;
					else icount += 1;
				}
			}
		}
	}

}

__global__ void Check_EarlyExercise2_GPU(
	int ny_grid, int jmin, int jmax, int kmin, int kmax, int nMat2,
	int OptType, float Strike, float AFut, float B0Fut, float B1Fut, float B2Fut,
	float FwdSpread, float *r_grid, float *y1_grid,
	float *y2_grid, float *V2)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	if (i < ny_grid && j < (1 + jmax - jmin) && k < (1 + kmax - kmin)) {

//	int j = blockIdx.x * blockDim.x + threadIdx.x;
//	int k = blockIdx.y * blockDim.y + threadIdx.y;

//	if (j < (1 + jmax - jmin) && k < (1 + kmax - kmin)) {
		int ijk, jp, kp;
		float tem, temopt;
		kp = kmin + k;
		jp = jmin + j;
		if (OptType == 2) {
	//	put options on 3m LIBOR futures rate
			ijk = i * (ny_grid + 1)*(ny_grid + 1) + jp * (ny_grid + 1) + kp;
			tem = (exp(AFut + B0Fut * r_grid[i] + B1Fut * y1_grid[jp] + B2Fut * y2_grid[kp]) - 1.0)*360.0 / nMat2;
			tem = tem + FwdSpread;
			temopt = max(0.0, Strike - tem);
			if (V2[ijk] < temopt) V2[ijk] = temopt;
		}
		else {
	// call options on 3m LIBOR futures rate
			ijk = i * (ny_grid + 1)*(ny_grid + 1) + jp * (ny_grid + 1) + kp;
			tem = (exp(AFut + B0Fut * r_grid[i] + B1Fut * y1_grid[jp] + B2Fut * y2_grid[kp]) - 1.0)*360.0 / nMat2;
			tem = tem + FwdSpread;
			temopt = max(0.0, tem - Strike);
			if (V2[ijk] < temopt) V2[ijk] = temopt;
		}
	}

}
