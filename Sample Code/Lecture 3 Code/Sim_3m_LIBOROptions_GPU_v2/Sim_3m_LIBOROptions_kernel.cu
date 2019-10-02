//	GPU device code
//	Using Monte Carlo MRG32k3a
//

#include <stdio.h>
#include <windows.h>

__device__ inline float sninvdev(double dseed[])
{
	const double norm = 2.328306549295728e-10;
	const double m1 = 4294967087.0;
	const double m2 = 4294944443.0;
	const double a12 = 1403580.0;
	const double a13n = 810728.0;
	const double a21 = 527612.0;
	const double a23n = 1370589.0;
	int k;
	double p1, p2;
	float ans, p;

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

	if (p1 <= p2) p = ((p1 - p2 + m1)*norm);
	else p = ((p1 - p2)*norm);

	if (p <= 0.0) {
		ans = -100.0f;
	}
	else {
		if (p >= 1.0) ans = 100.0f;
		else ans = normcdfinvf(p);
	}

	return ans;

}

////////////////////////////////////////////////////////////////////////////////
//Process an array of nSim simulations of the 3 Factor Hull-White model on GPU
////////////////////////////////////////////////////////////////////////////////
__global__ void SimulatePathGPU(
	int nSim, int nMat, int nSteps, double dt, float y1const, float r0, 
	float y10, float y20, int *nMat1, int *nMat2,
	unsigned int *seeds, float *AConst, float *B0, float *B1, 
	float *B2, float *FwdSpread, float *temexp, 
	float *lamsig, float *sigz, double *SimDiscount, double *SimLIBOR)
{
	int iSim;
	//Thread index
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	iSim = tid;
	if (iSim < nSim) {

		double dseed[6];
		float r, y1, y2, rnew, y1new, y2new;
		float y1rho, y2rho, cfexp0, y1avg, y2avg;
		double RInt;
		int i, j, k, jj, kk, jk, jk_skip, jstart, jend, ind;

		//	extract seeds for this path from seeds
		for (i = 0; i < 6; i++) dseed[i] = seeds[i + iSim * 6];

		//	Move all of the references from global memory to local memory
		cfexp0 = temexp[0];
		y1rho = temexp[1];
		//	Pass y1const
		//	y1const = theta[1] * (1.0 - y1rho);
		y2rho = temexp[2];

		//	for (i = 1; i <= nSim; i++) {
		y1 = y10;
		y2 = y20;
		r = r0;
		jstart = 1;
		RInt = 0.0;
		for (j = 0; j < nMat; j++) {
			jend = nMat1[j];
			for (jj = jstart; jj <= jend; jj++) {
				for (kk = 1; kk <= nSteps; kk++) {
					//	Simulate y2
					y2new = y2rho * y2 - lamsig[2] + sigz[2] * sninvdev(dseed);
					y2avg = 0.5*(y2new + y2);
					y2 = y2new;
					//	Simulate y1
					y1new = y1const + y1rho * y1 - lamsig[1] + sigz[1] * sninvdev(dseed);
					y1avg = 0.5*(y1 + y1new);
					y1 = y1new;
					//	Simulate r
					rnew = cfexp0 * r + (1.0 - cfexp0)*(y1avg + y2avg) - lamsig[0] + sigz[0] * sninvdev(dseed);
					RInt = RInt + 0.5*(r + rnew);
					r = rnew;
				}		//	end of loop on kk for time steps within a day

			}		//	End of loop on jj

//			tem_dp = exp(-RInt * dt);
			//	compute 3m LIBOR rate and option payoffs
			//	First compute 3m OIS on simulation path
//			tem_LIBOR = (exp(AConst[j] + B0[j] * r + B1[j] * y1 + B2[j] * y2) - 1.0)*360.0 / nMat2[j];
//			tem_LIBOR = tem_LIBOR + FwdSpread[j];

			SimDiscount[iSim*nMat + j] = exp(-RInt * dt);
			SimLIBOR[iSim*nMat + j] = FwdSpread[j] + (exp(AConst[j] + B0[j] * r + B1[j] * y1 + B2[j] * y2) - 1.0)*360.0 / nMat2[j];

			jstart = jend + 1;

		}		//	End of loop on j for nMat maturities

	}
}
