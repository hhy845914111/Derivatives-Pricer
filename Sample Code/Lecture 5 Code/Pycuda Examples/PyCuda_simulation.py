import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

###  Need to add the following lines to environment
###  This will vary on different computers, particularly,
###      the 14.16.27023 folder name
###  These lines are not necessary on Linux, or on the Prince cluster
import os
if (os.system("cl.exe")):
    os.environ['PATH'] += ';'+r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\HostX64\x64"
if (os.system("cl.exe")):
    raise RuntimeError("cl.exe still not found, path probably incorrect")

#a = np.random.randn(4,4)
#a = a.astype(np.float32)
#a_gpu = cuda.mem_alloc(a.nbytes)
#cuda.memcpy_htod(a_gpu, a)

mod = SourceModule("""
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

 __global__ void SimulatePathGPU(int nSim, int nTimePeriods, float dt, float sigma, 
        unsigned int *seeds, float *result)
 {
	int iSim = threadIdx.x + threadIdx.y;

	if (iSim < nSim) {
		int i;
		float sqrt_dt, z;
		double dseed[6];
		//	extract seeds for this path from seeds
		for (i = 0; i < 6; i++) dseed[i] = seeds[i + iSim * 6];

		sqrt_dt = sqrtf(dt);
		z = 0.0;
		for (i=0;i<nTimePeriods;i++) z += sigma * sqrt_dt * sninvdev(dseed);

		result[iSim] = z;
	}
 }                   
""")

nSim = 1028
nTimePeriods = 1000
dt = 1.0 / nTimePeriods
sigma = 0.20
nBlock = (1 + nSim / 128)
#dt = dt.astype(np.float32)
#sigma = sigma.astype(np.float32)
result = np.zeros(nSim).astype(np.float32)
seeds = np.zeros(nSim*6).astype(np.uint32)
seeds[0] = 39499
seeds[1] = 828388
seeds[2] = 129399
seeds[3] = 39499
seeds[4] = 828388
seeds[5] = 1295959
##  Need to apply skip ahead for seeds
for i in range(0,nSim-1):
    for j in range(0,6): seeds[(i+1)*6+j] = seeds[i*6+j]+13

d_seeds = cuda.mem_alloc(seeds.nbytes)
d_result  = cuda.mem_alloc(result.nbytes)
cuda.memcpy_htod(d_seeds, seeds)

func = mod.get_function("SimulatePathGPU")
#func(a_gpu, block=(nBlock,128))
func(np.int32(nSim), np.int32(nTimePeriods), np.float32(dt), np.float32(sigma), d_seeds, d_result, block=(1024,1,1))

cuda.memcpy_dtoh(result, d_result)
#print(a_doubled)
#print(a)

print(' length ', len(result), ' mean ', np.mean(result), np.std(result), np.var(result))

