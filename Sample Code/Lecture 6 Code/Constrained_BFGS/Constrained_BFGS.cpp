int ConstrainedBFGS(int n, int maxit, int &niterations, double &f, double *x, double *g,
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
