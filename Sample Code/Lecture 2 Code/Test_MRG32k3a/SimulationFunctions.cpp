//	Double precision simulation functions for uniform and normal only
#include "stdafx.h"
#include "SimulationFunctions.h"

void Gen_seed(unsigned int &ib1, unsigned int &ib2, unsigned int ib3, unsigned int ib4) 
{
	unsigned int item, ic1, ic2 ;

	item = Int64ShrlMod32(UInt32x32To64(ib3, ib1), 31) ;
	ic1 = (UInt32x32To64(ib3, ib1) % I2_31) ;
	ic2 = (UInt32x32To64(ib3, ib2) + UInt32x32To64(ib4, ib1) + item) ;
	ic2 = ic2 % I2_28 ;
	ib1 = ic1 ;
	ib2 = ic2 ;
//	Result is returned in ib1, ib2
}

double sninvdev(unsigned int &ib1, unsigned int &ib2)
{
	double q, r, ans, p ;
    const unsigned int I28_max = 268435455 ;
    const unsigned int I31_max = 2147483647 ;
    
//		This works on GPU
	ib2 = UInt32x32To64(I13_7, ib2) + (UInt32x32To64(I13_7, ib1) >> 31 ) ;	
	ib1 = UInt32x32To64(I13_7, ib1) & I31_max ;
	ib2 = (UInt32x32To64(I13_6, ib2) + (UInt32x32To64(I13_6, ib1) >> 31)) & I28_max ;
	ib1 = UInt32x32To64(I13_6, ib1) & I31_max ;
		
	p = ib2*fI2_28 + ib1*fI2_59 ;   

	/*
		if (p <= 0.0) {
		ans = -100.0f;
	}
	else {
		if (p >= 1.0) ans = 100.0f;
		else ans = erfinv(p);
	}
	*/


	if (p < 0.0 ) { 
		ans = -100.0 ;
	} 
	else { 
		if (p > 1.0) ans = 100.00 ; 
	
		else {
			if (p < 0.02425) { 
				q = sqrt(-2.0*log(p)) ; 
				ans = (((((-0.007784894002430293*q-0.3223964580411365)*q-2.400758277161838)*q-2.549732539343734)*q+4.374664141464968)*q+2.938163982698783) /
						((((0.007784695709041462*q+0.3224671290700398)*q+2.445134137142996)*q+3.754408661907416)*q+1.0) ; 
				}
			else {
				if (p < 0.97575) { 
					q = p - 0.5 ; 
					r = q*q ; 
					ans = (((((-39.69683028665376*r+220.9460984245205)*r-275.9285104469687)*r+138.3577518672690)*r-30.66479806614716)*r+2.506628277459239)*q /
						(((((-54.47609879822406*r+161.5858368580409)*r-155.6989798598866)*r+66.80131188771972)*r-13.28068155288572)*r+1.0) ;
				} 
				else { 
					if (p < 0.99) {
						q = sqrt(-2.0*log(1.0-p)) ;
						ans = -(((((-0.007784894002430293*q-0.3223964580411365)*q-2.400758277161838)*q-2.549732539343734)*q+4.374664141464968)*q+2.938163982698783) /
							((((0.007784695709041462*q+0.3224671290700398)*q+2.445134137142996)*q+3.754408661907416)*q+1.0) ;
					} 
					else { 
						r = (I2_28-ib2)*fI2_28 - ib1*fI2_59 ;
						q = sqrt(-2.0*log(r)) ;

						ans = -(((((-0.007784894002430293*q-0.3223964580411365)*q-2.400758277161838)*q-2.549732539343734)*q+4.374664141464968)*q+2.938163982698783) /
							((((0.007784695709041462*q+0.3224671290700398)*q+2.445134137142996)*q+3.754408661907416)*q+1.0) ;
					}
				}
			}
		}
	}

	return ans ;

}

double rand_u01(unsigned int &ib1, unsigned int &ib2) 
{
	unsigned int item ;
	double answer ;

	item = Int64ShrlMod32(UInt32x32To64(I13_7, ib1), 31) ;
	ib1 = (UInt32x32To64(I13_7, ib1) % I2_31) ;
	ib2 = (UInt32x32To64(I13_7, ib2) + item) ;
	item = Int64ShrlMod32(UInt32x32To64(I13_6, ib1), 31) ;
	ib1 = (UInt32x32To64(I13_6, ib1) % I2_31) ;
	ib2 = (UInt32x32To64(I13_6, ib2) + item) ;
	ib2 = ib2 % I2_28 ;
	answer = ib2*fI2_28 + ib1*fI2_59 ; 

	return answer ;
}

void roll_seed(unsigned int &ib1, unsigned int &ib2)
{
	unsigned int item;
//	double answer;

	item = Int64ShrlMod32(UInt32x32To64(I13_7, ib1), 31);
	ib1 = (UInt32x32To64(I13_7, ib1) % I2_31);
	ib2 = (UInt32x32To64(I13_7, ib2) + item);
	item = Int64ShrlMod32(UInt32x32To64(I13_6, ib1), 31);
	ib1 = (UInt32x32To64(I13_6, ib1) % I2_31);
	ib2 = (UInt32x32To64(I13_6, ib2) + item);
	ib2 = ib2 % I2_28;
//	answer = ib2*fI2_28 + ib1*fI2_59;

//	return answer;
}
