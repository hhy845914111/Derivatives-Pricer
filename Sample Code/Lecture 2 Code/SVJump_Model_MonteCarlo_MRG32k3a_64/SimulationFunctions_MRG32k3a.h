//	Double precision simulation functions for uniform and normal only
#define I2_28 268435456 
#define I2_29 536870912 
#define I2_30 1073741824
#define I2_31 2147483648
#define I13_6 4826809
#define I13_7 62748517
#define pi 3.14159265358979
#define dsqr2 0.707106781186547
#define dsqr2pi 0.398942280401433
#define sqrpi 1.77245385090552
#define dsqrpi 0.564189583547756
#define half_ln_2pi 0.918938533204673
#define fI2_28 3.72529029846191E-09
#define fI2_59 1.73472347597681E-18

double rand_u01(double dseed[]);
void roll_seed(double dseed[]);
void SkipAhead_MRG32k3a(int n, unsigned int **An1, unsigned int **An2);
void SkipAhead2_MRG32k3a(int n, unsigned int **An1, unsigned int **An2, unsigned int **Bn1, unsigned int **Bn2);
double sninvdev(double dseed[]);