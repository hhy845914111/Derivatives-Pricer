//	Double precision simulation functions for uniform and normal only

double rand_u01(double dseed[]);
void roll_seed(double dseed[]);
void SkipAhead_MRG32k3a(int n, unsigned int **An1, unsigned int **An2);
void SkipAhead2_MRG32k3a(int n, unsigned int **An1, unsigned int **An2, unsigned int **Bn1, unsigned int **Bn2);
double sninvdev(double dseed[]);