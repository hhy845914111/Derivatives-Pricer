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
//Process an array of nSim simulations of the SPX model on GPU
////////////////////////////////////////////////////////////////////////////////
__global__ void SimulatePathGPU(
	int nSim, int nMat, int nSteps, float dt, float sqdt, float FX0, float v0,
	float BCoef, float ACoef, float sigCoef, float rho, unsigned int *seeds, 
	int *Maturity, float *rd, float *rf, float *d_SimFX)
{
	int iSim;
	//Thread index
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	iSim = tid;
	if (iSim < nSim) {
		double dseed[6];
		int i, j, k, jj, kk, jstart, jend;
		float x, y, v, vnew, vavg, rhosq, z1, z2;

		//	extract seeds for this path from seeds
		for (i = 0; i < 6; i++) dseed[i] = seeds[i + iSim * 6];

		x = log(FX0);
		v = v0;
		y = log(v0*v0);
		rhosq = sqrt(1.0 - rho * rho);
		jstart = 1;
		for (j = 1; j <= nMat; j++) {
			jend = Maturity[j];
			for (jj = jstart; jj <= jend; jj++) {
				for (kk = 1; kk <= nSteps; kk++) {
					z2 = sninvdev(dseed);
						z1 = rho * z2 + rhosq * sninvdev(dseed);
					//	Simulate y
					y = BCoef * y + ACoef + sigCoef * z2;
					//yavg = 0.5*(ynew + y);
					//y = ynew;
					vnew = exp(0.5*y);
					//vavg = 0.5*(v + vnew);
					x += (rd[jj] - rf[jj] - 0.5*v*v)*dt + v * sqdt*z1;
					v = vnew;
				}		//	end of loop on kk for time steps within a day
			}		//	End of loop on jj
			d_SimFX[iSim*(nMat+1)+j] = exp(x);
			jstart = jend + 1;
		}		//	End of loop on j for nMat maturities
	}	
}
