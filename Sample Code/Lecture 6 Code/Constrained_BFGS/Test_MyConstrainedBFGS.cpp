// Test_MyConstrainedBFGS.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "stdafx.h"

#define NR_END 1
#define FREE_ARG char*

int MyConstrainedBFGS(int n, int maxit, int &niterations, double &f, double *x, double *g,
	double **Hessian, double **HessInv, double bl[], double bu[]);
double funct1(int n, double x[]);
double funct2(int n, double x[], double *g);
double *dvector(long nl, long nh, short ifail);
double **dmatrix(long nrl, long nrh, long ncl, long nch, short ifail);
void free_dvector(double *v, long nl, long nh);
void free_dmatrix(double **m, long nrl, long nrh, long ncl, long nch);
int *ivector(long nl, long nh, short ifail);
void free_ivector(int *iv, long nl, long nh);
void sminv(int n, double **a, double **ainv, double wk[], short ifail);
void choldc(double **a, int n, double p[], short ifail);

int _tmain(int argc, _TCHAR* argv[])
{
	int i, j, ifail, seed, iset = 0, n, maxit, niterations;
	double *x, *g, **Hess, **HessInv, *temx;
	double tem1, tem2, f, tem[30], bl[10], bu[10];
	short icheck = 1;
	long nDim;

	n = 3;
	nDim = n;
	x = dvector(1, nDim, icheck);
	g = dvector(1, nDim, icheck);
	Hess = dmatrix(1, nDim, 1, nDim, icheck);
	HessInv = dmatrix(1, nDim, 1, nDim, icheck);
	temx = dvector(1, nDim, icheck);

	maxit = 100;

	x[1] = 0.0;
	x[2] = 0.0;
	x[3] = 0.0;

	bl[1] = -20.0;
	bl[2] = -20.0;
	bl[3] = -20.0;
	bu[1] = 20.0;
	bu[2] = 20.0;
	bu[3] = 1.0;

	//	Initialize HessInv
	f = funct2(n, x, g);
	tem1 = 0.000001;
	for (i = 1; i <= n; i++) {
		tem[i] = g[i];
		temx[i] = x[i];
	}
	for (i = 1; i <= n; i++) {
		for (j = 1; j <= n; j++) {
			Hess[i][j] = 0.0;
			HessInv[i][j] = 0.0;
		}
	}
	for (i = 1; i <= n; i++) {
		temx[i] = x[i] + tem1;
		f = funct2(n, temx, g);
		Hess[i][i] = (g[i] - tem[i]) / tem1;
		HessInv[i][i] = 1.0 / Hess[i][i];
		temx[i] = x[i];
	}
	printf("  Initial x[1] = %10.6f   x[2] = %10.6f   x[3] = %10.6f \n", x[1], x[2], x[3]);
	//	printf("  Initial HessInv[1][1] = %10.6f   [2] = %10.6f   [3] = %10.6f \n", HessInv[1][1], HessInv[2][2], HessInv[3][3]) ;

	ifail = MyConstrainedBFGS(n, maxit, niterations, f, x, g, Hess, HessInv, bl, bu);

	printf("  ifail = %3i   Number of iterations %5i   Minimum of Function %12.6e \n", ifail, niterations, f);
	printf("  X's at minimum: %12.6e %12.6e %12.6e \n", x[1], x[2], x[3]);
	printf("  Gradient:  %12.6e  %12.6e  %12.6e \n", g[1], g[2], g[3]);
	printf("  Diagonals of Inverse of Information Matrix:   %12.6e  %12.6e  %12.6e \n", HessInv[1][1], HessInv[2][2], HessInv[3][3]);

	free_dvector(x, 1, nDim);
	free_dvector(g, 1, nDim);
	free_dmatrix(Hess, 1, nDim, 1, nDim);
	free_dmatrix(HessInv, 1, nDim, 1, nDim);
	free_dvector(temx, 1, nDim);

	return 0;

}

/*
int MyConstrainedBFGS(int n, int maxit, int *niterations, double *f, double *x, double *g,
double **Hessian, double **HessInv, double bl[], double bu[])
{
//
//	int n = number of parameters to estimate
//  int maxit = maximum number of iterations
//	*f = double pointer for function value to be minimized
//	*x = double pointer for parameter values, 1 dimensional array
//	*g = double pointer for gradient, 1 dimensional array
//	**Hessian = double pointer for Hessian matrix, 2 dimensional array
//	**HessInv = double pointer for inverse of Hessian matrix, 2 dimensional array
//  lower bounds on x's are in bl[]
//	upper bounds on x's are in bu[]
//	funct1 is used to compute likelihood function
//	funct2 is used to compute likelihood function and gradient
//	The BFGS approximates inverse of Hessian, starting with initial Hessian in **Hessian
//
int i, j, k, iter = 0, ifailtest, *gInd ;
double lambda, fx =0.0, ff = 0.0, a1, a2, tem1, tem2 ;
double *wk, *wk2, *ConvCrit ;
long nDim ;
short ifail ;

ifail = 1 ;
nDim = n ;
wk=dvector(1,nDim,ifail) ;
if (ifail < 0) return -7 ;
wk2=dvector(1,nDim,ifail) ;
if (ifail < 0) {
free_dvector(wk,1,nDim) ;
free_dvector(wk2,1,nDim) ;
return -7 ;
}
ConvCrit=dvector(1,nDim,ifail) ;
if (ifail < 0) {
free_dvector(wk,1,nDim) ;
free_dvector(wk2,1,nDim) ;
free_dvector(ConvCrit,1,nDim) ;
return -7 ;
}
gInd=ivector(1,nDim,ifail) ;
if (ifail < 0) {
free_dvector(wk,1,nDim) ;
free_dvector(wk2,1,nDim) ;
free_dvector(ConvCrit,1,nDim) ;
free_ivector(gInd,1,nDim) ;
return -7 ;
}

for (i=1;i<=n;i++) {
ConvCrit[i] = 1.0e-08 ;
gInd[i] = 1 ;
}

iter = 0 ;

fx = funct2(n, x, g) ;

for (i=1;i<=n;i++) {
wk[i] = 0.0 ;
for (k=1;k<=n;k++) wk[i] = wk[i] + HessInv[i][k]*g[k] ;
}
for (i=1;i<=n;i++) {
gInd[i] = 1 ;
if ((wk[i] <= 0.0) && (x[i] == bu[i])) gInd[i] = 0 ;
if ((wk[i] >= 0.0) && (x[i] == bl[i])) gInd[i] = 0 ;
}

for (i=1;i<=n;i++) {
wk[i] = 0.0 ;
for (k=1;k<=n;k++) wk[i] = wk[i] + HessInv[i][k]*g[k]*gInd[k] ;
if (gInd[i] == 0) wk[i] = 0.0 ;
}

ifailtest = 1 ;
while (iter < maxit && ifailtest == 1) {
//    check and do lambda search
lambda = 2.0 ;
j = 0;
ifail = 2 ;
while (j < 20 && ifail == 2) {
lambda = 0.5*lambda ;
for (i=1;i<=n;i++) {
wk2[i] = x[i] - lambda*wk[i] ;
//	Parameter restrictions here
if (wk2[i] > bu[i]) wk2[i] = bu[i] ;
if (wk2[i] < bl[i]) wk2[i] = bl[i] ;
}

ff = funct1(n, wk2) ;
if (ff < fx) ifail = 1 ;
j = j+1 ;
}

if (ifail == 1) {
for (i=1;i<=n;i++) x[i] = wk2[i] ;
//		load s_k into wk2
for (i=1;i<=n;i++) wk2[i] = -lambda*wk[i] ;
//		load old gradient into wk
for (i=1;i<=n;i++) wk[i] = g[i] ;
fx = funct2(n, x, g) ;
//		compute change in gradient and load into wk
for (i=1;i<=n;i++) wk[i] = g[i] - wk[i] ;
//		compute approximation for inverse of Hessian
tem1 = 0.0 ;
for (i=1;i<=n;i++) tem1 += wk[i]*wk2[i] ;
tem2 = 0.0 ;
for (i=1;i<=n;i++) {
for (j=1;j<=n;j++) tem2 += HessInv[i][j]*wk[i]*wk[j] ;
}
a1 = (tem1+tem2)/(tem1*tem1) ;
a2 = 1.0/tem1 ;
for (i=1;i<=n;i++) {
for (j=1;j<=n;j++) {
Hessian[i][j] = 0.0 ;
for (k=1;k<=n;k++) Hessian[i][j] += HessInv[i][k]*wk[k]*wk2[j] + wk2[i]*wk[k]*HessInv[k][j] ;
}
}
for (i=1;i<=n;i++) {
for (j=1;j<=n;j++) HessInv[i][j] = HessInv[i][j] + a1*wk2[i]*wk2[j]
- a2*Hessian[i][j] ;
}

for (i=1;i<=n;i++) {
wk[i] = 0.0 ;
for (k=1;k<=n;k++) wk[i] = wk[i] + HessInv[i][k]*g[k] ;
}
for (i=1;i<=n;i++) {
gInd[i] = 1 ;
if ((wk[i] <= 0.0) && (x[i] == bu[i])) gInd[i] = 0 ;
if ((wk[i] >= 0.0) && (x[i] == bl[i])) gInd[i] = 0 ;
}
for (i=1;i<=n;i++) {
wk[i] = 0.0 ;
for (k=1;k<=n;k++) wk[i] = wk[i] + HessInv[i][k]*g[k]*gInd[k] ;
if (gInd[i] == 0) wk[i] = 0.0 ;
}

ifail = 0 ;
for (i=1;i<=n;i++) {
if (fabs(wk[i]) > ConvCrit[i]*gInd[i]) ifail = 1 ;
}
}

*f = fx ;
iter = iter + 1 ;
ifailtest = ifail ;
}                   //    End of while to perform iterations on algorithm

free_dvector(wk,1,nDim) ;
free_dvector(wk2,1,nDim) ;
free_dvector(ConvCrit,1,nDim) ;
free_ivector(gInd,1,nDim) ;

*f = fx ;
*niterations = iter ;
return ifailtest ;

}              //     End of MyConstrainedBFGS
*/


int MyConstrainedBFGS(int n, int maxit, int &niterations, double &f, double *x, double *g,
	double **Hessian, double **HessInv, double bl[], double bu[])
{
	//
	//	int n = number of parameters to estimate
	//  int maxit = maximum number of iterations
	//	*f = double pointer for function value to be minimized
	//	*x = double pointer for parameter values, 1 dimensional array
	//	*g = double pointer for gradient, 1 dimensional array
	//	**Hessian = double pointer for Hessian matrix, 2 dimensional array
	//	**HessInv = double pointer for inverse of Hessian matrix, 2 dimensional array
	//  lower bounds on x's are in bl[]
	//	upper bounds on x's are in bu[]
	//	funct1 is used to compute likelihood function
	//	funct2 is used to compute likelihood function and gradient
	//	The BFGS approximates inverse of Hessian, starting with initial Hessian in **Hessian
	//
	int i, j, k, iter = 0, ifailtest, *gInd;
	double lambda, fx = 0.0, ff = 0.0, a1, a2, tem1, tem2;
	double *wk, *wk2, *ConvCrit;
	//	long nDim;
	short ifail;

	ifail = 1;
	//	nDim = n;

	wk = (double *)malloc((n + 1) * sizeof(double));
	wk2 = (double *)malloc((n + 1) * sizeof(double));
	ConvCrit = (double *)malloc((n + 1) * sizeof(double));
	gInd = (int *)malloc((n + 1) * sizeof(int));

	for (i = 1; i <= n; i++) {
		ConvCrit[i] = 1.0e-08;
		gInd[i] = 1;
	}

	iter = 0;

	fx = funct2(n, x, g);

	for (i = 1; i <= n; i++) {
		wk[i] = 0.0;
		for (k = 1; k <= n; k++) wk[i] = wk[i] + HessInv[i][k] * g[k];
	}
	for (i = 1; i <= n; i++) {
		gInd[i] = 1;
		if ((wk[i] <= 0.0) && (x[i] == bu[i])) gInd[i] = 0;
		if ((wk[i] >= 0.0) && (x[i] == bl[i])) gInd[i] = 0;
	}

	for (i = 1; i <= n; i++) {
		wk[i] = 0.0;
		for (k = 1; k <= n; k++) wk[i] = wk[i] + HessInv[i][k] * g[k] * gInd[k];
		if (gInd[i] == 0) wk[i] = 0.0;
	}

	ifailtest = 1;
	while (iter < maxit && ifailtest == 1) {
		//    check and do lambda search
		lambda = 2.0;
		j = 0;
		ifail = 2;
		while (j < 20 && ifail == 2) {
			lambda = 0.5*lambda;
			for (i = 1; i <= n; i++) {
				wk2[i] = x[i] - lambda*wk[i];
				//	Parameter restrictions here
				if (wk2[i] > bu[i]) wk2[i] = bu[i];
				if (wk2[i] < bl[i]) wk2[i] = bl[i];
			}

			ff = funct1(n, wk2);
			if (ff < fx) ifail = 1;
			j = j + 1;
		}

		if (ifail == 1) {
			for (i = 1; i <= n; i++) x[i] = wk2[i];
			//		load s_k into wk2
			for (i = 1; i <= n; i++) wk2[i] = -lambda*wk[i];
			//		load old gradient into wk
			for (i = 1; i <= n; i++) wk[i] = g[i];
			fx = funct2(n, x, g);
			//		compute change in gradient and load into wk
			for (i = 1; i <= n; i++) wk[i] = g[i] - wk[i];
			//		compute approximation for inverse of Hessian
			tem1 = 0.0;
			for (i = 1; i <= n; i++) tem1 += wk[i] * wk2[i];
			tem2 = 0.0;
			for (i = 1; i <= n; i++) {
				for (j = 1; j <= n; j++) tem2 += HessInv[i][j] * wk[i] * wk[j];
			}
			a1 = (tem1 + tem2) / (tem1*tem1);
			a2 = 1.0 / tem1;
			for (i = 1; i <= n; i++) {
				for (j = 1; j <= n; j++) {
					Hessian[i][j] = 0.0;
					for (k = 1; k <= n; k++) Hessian[i][j] += HessInv[i][k] * wk[k] * wk2[j] + wk2[i] * wk[k] * HessInv[k][j];
				}
			}
			for (i = 1; i <= n; i++) {
				for (j = 1; j <= n; j++) HessInv[i][j] = HessInv[i][j] + a1*wk2[i] * wk2[j]
					- a2*Hessian[i][j];
			}

			for (i = 1; i <= n; i++) {
				wk[i] = 0.0;
				for (k = 1; k <= n; k++) wk[i] = wk[i] + HessInv[i][k] * g[k];
			}
			for (i = 1; i <= n; i++) {
				gInd[i] = 1;
				if ((wk[i] <= 0.0) && (x[i] == bu[i])) gInd[i] = 0;
				if ((wk[i] >= 0.0) && (x[i] == bl[i])) gInd[i] = 0;
			}
			for (i = 1; i <= n; i++) {
				wk[i] = 0.0;
				for (k = 1; k <= n; k++) wk[i] = wk[i] + HessInv[i][k] * g[k] * gInd[k];
				if (gInd[i] == 0) wk[i] = 0.0;
			}

			ifail = 0;
			for (i = 1; i <= n; i++) {
				if (fabs(wk[i]) > ConvCrit[i] * gInd[i]) ifail = 1;
			}
		}

		f = fx;
		iter = iter + 1;
		ifailtest = ifail;
	}                   //    End of while to perform iterations on algorithm

	free(wk);
	free(wk2);
	free(ConvCrit);
	free(gInd);

	f = fx;
	niterations = iter;
	return ifailtest;

}              //     End of ConstrainedBFGS

//    Function to compute f only
double funct1(int n, double x[])
{
	double f = 0.0;

	//	f = 1.0 - 2.0*x[1] - 3.0*x[2] - 4.0*x[3] + 1.2*x[1]*x[1] + 1.4*x[2]*x[2] + 1.6*x[3]*x[3] ;
	f = 1.0 - 2.0*x[1] - 3.0*x[2] - 4.0*x[3] + 1.2*x[1] * x[1] + 1.4*x[2] * x[2] + 1.6*x[3] * x[3]
		+ 0.5*x[1] * x[1] * x[1] * x[1] + 0.5*x[2] * x[2] * x[3] * x[3];
	/*
	f = 1.0 + 2.0*x[1] + 3.0*x[2] + 4.0*x[3] - 0.5*x[1]*x[1] -0.5*x[2]*x[2]  -4.0*x[1]*x[2]*x[3] ;
	f = 0.5*f*f ;
	*/

	return f;
}              //      End of funct1

//    Function to compute f, first derivatives, and information matrix
double funct2(int n, double x[], double *g)
{
	double f = 0.0;
	//	int i ;

	//	  Compute f and g[]
	//	f = 1.0 - 2.0*x[1] - 3.0*x[2] - 4.0*x[3] + 1.2*x[1]*x[1] + 1.4*x[2]*x[2] + 1.6*x[3]*x[3] ;
	f = 1.0 - 2.0*x[1] - 3.0*x[2] - 4.0*x[3] + 1.2*x[1] * x[1] + 1.4*x[2] * x[2] + 1.6*x[3] * x[3]
		+ 0.5*x[1] * x[1] * x[1] * x[1] + 0.5*x[2] * x[2] * x[3] * x[3];

	//	g[1] = -2.0 + 1.2*2.0*x[1] ;
	//	g[2] = -3.0 + 1.4*2.0*x[2] ;
	//	g[3] = -4.0 + 1.6*2.0*x[3] ;
	g[1] = -2.0 + 1.2*2.0*x[1] + 0.5*4.0*x[1] * x[1] * x[1];
	g[2] = -3.0 + 1.4*2.0*x[2] + x[2] * x[3] * x[3];
	g[3] = -4.0 + 1.6*2.0*x[3] + x[2] * x[2] * x[3];

	/*
	f = 1.0 + 2.0*x[1] + 3.0*x[2] + 4.0*x[3] - 0.5*x[1]*x[1] -0.5*x[2]*x[2] -4.0*x[1]*x[2]*x[3] ;
	g[1] = 2.0 - 4.0*x[2]*x[3] - x[1] ;
	g[2] = 3.0 - 4.0*x[1]*x[3] - x[2] ;
	g[3] = 4.0 - 4.0*x[1]*x[2] ;
	for (i=1;i<=n;i++) g[i] = g[i]*f ;
	f = 0.5*f*f ;
	*/
	printf(" Function = %12.6e \n", f);
	printf(" x's: %12.6e  %12.6e  %12.6e \n", x[1], x[2], x[3]);
	printf(" g's: %12.6e  %12.6e  %12.6e \n", g[1], g[2], g[3]);

	return f;
}              //      End of funct2

void sminv(int n, double **a, double **ainv, double wk[], short ifail)
/*   Function created by LOS 10/18/1999 to invert symmetric positive
definite matrix using functions from Numerical Recipes in C    */
{
	double sum;
	int i, j, k;

	/*    Call to Cholesky decomposition    */
	choldc(a, n, wk, ifail);

	if (ifail >= 0) {
		/*    Compute inverse of L, lower triangular    */
		for (i = 1; i <= n; i++) {
			a[i][i] = 1.0 / wk[i];
			for (j = i + 1; j <= n; j++) {
				sum = 0.0;
				for (k = i; k<j; k++) sum -= a[j][k] * a[k][i];
				a[j][i] = sum / wk[j];
			}
		}

		/*    Compute inverse of a and store in ainv    */
		for (i = 1; i <= n; i++) {
			for (j = i; j <= n; j++) {
				ainv[i][j] = 0.0;
				for (k = j; k <= n; k++) ainv[i][j] = ainv[i][j] + a[k][i] * a[k][j];
			}
		}
		for (i = 1; i <= n; i++) {
			for (j = i + 1; j <= n; j++) {
				ainv[j][i] = ainv[i][j];
			}
		}
	}         //    End of if on ifail
}                 /*   End of sminv    */

/*  The following code is from Numerical Recipes in C     */
/* (C) Copr. 1986-92 Numerical Recipes Software G2v#X):K. */
void choldc(double **a, int n, double p[], short ifail)
{
	void nrerror(char error_text[]);
	int i, j, k;
	double sum;

	for (i = 1; i <= n; i++) {
		for (j = i; j <= n; j++) {
			for (sum = a[i][j], k = i - 1; k >= 1; k--) sum -= a[i][k] * a[j][k];
			if (i == j) {
				if (sum <= 0.0) {
					ifail = -9;
					p[i] = 99999, 9;
				}
				else	p[i] = sqrt(sum);
				/*   The original code with call to nrerror which halts execution
				if (sum <= 0.0)
				nrerror("choldc failed");
				p[i]=sqrt(sum);
				*/
			}
			else a[j][i] = sum / p[i];
		}
	}
}
/* (C) Copr. 1986-92 Numerical Recipes Software G2v#X):K. */

double *dvector(long nl, long nh, short ifail)
/* allocate a double vector with subscript range v[nl..nh] */
{
	double *v;

	v = (double *)malloc((size_t)((nh - nl + 1 + NR_END)*sizeof(double)));
	if (!v) {
		ifail = -5;
		return 0;
	}
	//	if (!v) nrerror("allocation failure in dvector()");
	return v - nl + NR_END;
}

double **dmatrix(long nrl, long nrh, long ncl, long nch, short ifail)
/* allocate a double matrix with subscript range m[nrl..nrh][ncl..nch] */
{
	long i, nrow = nrh - nrl + 1, ncol = nch - ncl + 1;
	double **m;

	/* allocate pointers to rows */
	m = (double **)malloc((size_t)((nrow + NR_END)*sizeof(double*)));
	if (!m) {
		ifail = -5;
		return 0;
	}
	//	if (!m) nrerror("allocation failure 1 in matrix()");
	m += NR_END;
	m -= nrl;

	/* allocate rows and set pointers to them */
	m[nrl] = (double *)malloc((size_t)((nrow*ncol + NR_END)*sizeof(double)));
	if (!m[nrl]) {
		ifail = -5;
		return 0;
	}
	//	if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
	m[nrl] += NR_END;
	m[nrl] -= ncl;

	for (i = nrl + 1; i <= nrh; i++) m[i] = m[i - 1] + ncol;

	/* return pointer to array of pointers to rows */
	return m;
}

void free_dvector(double *v, long nl, long nh)
/* free a double vector allocated with dvector() */
{
	free((FREE_ARG)(v + nl - NR_END));
}

void free_dmatrix(double **m, long nrl, long nrh, long ncl, long nch)
/* free a double matrix allocated by dmatrix() */
{
	free((FREE_ARG)(m[nrl] + ncl - NR_END));
	free((FREE_ARG)(m + nrl - NR_END));
}

int *ivector(long nl, long nh, short ifail)
/* allocate an int vector with subscript range iv[nl..nh] */
{
	int *iv;

	iv = (int *)malloc((size_t)((nh - nl + 1 + NR_END)*sizeof(int)));
	if (!iv) {
		ifail = -8;
		return 0;
	}
	//  if (!iv) nrerror ( "allocation failure in i vector ()"); 
	return iv - nl + NR_END;
}

void free_ivector(int *iv, long nl, long nh)
/* free an int vector allocated with ivector ( ) */
{
	free((FREE_ARG)(iv + nl - NR_END));
}


/* (C) Copr. 1986-92 Numerical Recipes Software G2v#X):K. */
#undef NR_END
#undef FREE_ARG
