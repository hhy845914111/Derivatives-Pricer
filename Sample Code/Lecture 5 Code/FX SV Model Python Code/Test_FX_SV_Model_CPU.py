import numpy as np
import ctypes
from ctypes import * 
import sys
import os
import pandas as pd 
import math 
from scipy.stats import norm

def get_FXSV_ModelPricer():
	dll = ctypes.windll.LoadLibrary("C:/Users/Louis/Documents/Xlcpp/FX_SV_Model_CPU/x64/Release/FX_SV_Model_CPU.dll")    
	func = dll.FX_SV_Model_CPU
	func.argtypes = [c_int, c_int, c_int, c_int, c_int, c_double, c_double, \
                POINTER(c_int), POINTER(c_double), POINTER(c_double),POINTER(c_double), \
                POINTER(c_int), POINTER(c_double), POINTER(c_double),POINTER(c_double)] 
	return func

__FXSV_ModelPricer = get_FXSV_ModelPricer()

def FXSV_ModelPricer(nThreads, nSimPerThread, nTimeSteps, nExpirations, \
	nStrikesPerExp, spotFX, spotV, daysExp, r_dom, r_for, params,\
    OptTypeIn, StrikesIn, OptPrice, StdErr):

	daysExp_p = daysExp.ctypes.data_as(POINTER(c_int))
	r_dom_p = r_dom.ctypes.data_as(POINTER(c_double))
	r_for_p = r_for.ctypes.data_as(POINTER(c_double))
	params_p = params.ctypes.data_as(POINTER(c_double))
	OptTypeIn_p = OptTypeIn.ctypes.data_as(POINTER(c_int))
	StrikesIn_p = StrikesIn.ctypes.data_as(POINTER(c_double))
	OptPrice_p = OptPrice.ctypes.data_as(POINTER(c_double))
	StdErr_p = StdErr.ctypes.data_as(POINTER(c_double))

	__FXSV_ModelPricer(nThreads, nSimPerThread, nTimeSteps, nExpirations, \
	nStrikesPerExp, spotFX, spotV, daysExp_p, r_dom_p, r_for_p, params_p,\
    OptTypeIn_p, StrikesIn_p, OptPrice_p, StdErr_p)

if __name__ == '__main__':
    os.chdir('C:/Users/Louis/Documents/Python')
    fxdata = pd.read_csv('FX_Spot_Fwd_Rates_20190501.csv')
    
    nThreads = int(10)
    nSimPerThread = int(25000)
    nTimeSteps = int(1)
    nExpirations = int(5)
    nStrikesPerExp = int(5)
    params_p = np.zeros(6, dtype='float64')
    
    spotV  = float(0.042)
    longV  = float(0.038)
#   params:  kappa, theta, sigma, lambda, rho
    params_p[1] = 3.0
    params_p[2] = math.log(longV*longV)
    params_p[3] = 4.0
    params_p[4] = 0.0
    params_p[5] = -0.45   
    
    #input or assign values for remaining variables/pointers
	#daysExp = np.zeros(nExpirations).astype('int')
    daysExp_p = np.zeros((nExpirations+1), dtype='int')
    OptType_p = np.zeros((nExpirations+1)*(nStrikesPerExp+1), dtype='int')
    Strikes_p = np.zeros((nExpirations+1)*(nStrikesPerExp+1), dtype='float64')
    BS_Vol_bid = np.zeros((nExpirations+1, nStrikesPerExp+1), dtype='float64')
    BS_Vol_ask = np.zeros((nExpirations+1, nStrikesPerExp+1), dtype='float64')
    MktPrice_bid = np.zeros((nExpirations+1, nStrikesPerExp+1), dtype='float64')
    MktPrice_ask = np.zeros((nExpirations+1, nStrikesPerExp+1), dtype='float64')
    OptPrice_p = np.zeros((nExpirations+1)*(nStrikesPerExp+1), dtype='float64')
    StdErr_p = np.zeros((nExpirations+1)*(nStrikesPerExp+1), dtype='float64')
    fwdFX = np.zeros((nExpirations+1), dtype='float64')
    dom_rp = np.zeros((nExpirations+1), dtype='float64')
    for_rp = np.zeros((nExpirations+1), dtype='float64')

    spotFX = fxdata.FX_Spot[0]
    for i in range(0,nExpirations):
        daysExp_p[i+1] = fxdata.DaysExp[i]
        fwdFX[i+1] = fxdata.FX_Fwd[i]
        dom_rp[i+1] = fxdata.dom_r[i]
        OptType_p[(i+1)*(nStrikesPerExp+1)+1] = 2
        OptType_p[(i+1)*(nStrikesPerExp+1)+2] = 2
        OptType_p[(i+1)*(nStrikesPerExp+1)+3] = 1
        OptType_p[(i+1)*(nStrikesPerExp+1)+4] = 1
        OptType_p[(i+1)*(nStrikesPerExp+1)+5] = 1
        for_rp[i+1] = (math.log(spotFX/fwdFX[i+1]) + dom_rp[i+1]*daysExp_p[i+1]/365.0)*365.0/daysExp_p[i+1]

    r_dom_p = np.zeros((daysExp_p[5]+1), dtype='float64')
    r_for_p = np.zeros((daysExp_p[5]+1), dtype='float64')
    
    jstart = 0
    for i in range(0,nExpirations):
        if i==0:
            f_dom = dom_rp[i+1]
            f_for = for_rp[i+1]
        else:
            f_dom = (dom_rp[i+1]*daysExp_p[i+1]-dom_rp[i]*daysExp_p[i])/(daysExp_p[i+1]-daysExp_p[i])
            f_for = (for_rp[i+1]*daysExp_p[i+1]-for_rp[i]*daysExp_p[i])/(daysExp_p[i+1]-daysExp_p[i])
        for j in range(jstart,daysExp_p[i+1]+1):
            r_dom_p[j] = f_dom
            r_for_p[j] = f_for
        jstart = daysExp_p[i+1]+1

    v_deltas = [ 0.90, 0.75, 0.50, 0.25, 0.10]

    for i in range(0,nExpirations):
        BS_Vol_bid[i+1][1] = fxdata.Bid_10D_P[i]/100.0
        BS_Vol_ask[i+1][1] = fxdata.Ask_10D_P[i]/100.0
        BS_Vol_bid[i+1][2] = fxdata.Bid_25D_P[i]/100.0
        BS_Vol_ask[i+1][2] = fxdata.Ask_25D_P[i]/100.0
        BS_Vol_bid[i+1][3] = fxdata.Bid_ATM[i]/100.0
        BS_Vol_ask[i+1][3] = fxdata.Ask_ATM[i]/100.0
        BS_Vol_bid[i+1][4] = fxdata.Bid_25D_C[i]/100.0
        BS_Vol_ask[i+1][4] = fxdata.Ask_25D_C[i]/100.0
        BS_Vol_bid[i+1][5] = fxdata.Bid_10D_C[i]/100.0
        BS_Vol_ask[i+1][5] = fxdata.Ask_10D_C[i]/100.0
        for j in range(0, nStrikesPerExp):
            k = (i+1)*(nStrikesPerExp+1)+j+1
            T = daysExp_p[i+1]/365.0
            tem = 0.5*(BS_Vol_bid[i+1][j+1]+BS_Vol_ask[i+1][j+1])
            tem1 = 0.5*tem*tem*T
            tem2 = tem*math.sqrt(T)            
            Strikes_p[k] = fwdFX[i+1]*math.exp(tem1 - norm.ppf(v_deltas[j])*tem2)
            d1 = (math.log(spotFX/Strikes_p[k])+(dom_rp[i+1]-for_rp[i+1]+0.5*BS_Vol_bid[i+1][j+1]*BS_Vol_bid[i+1][j+1])*T)
            d1 = d1 / (BS_Vol_bid[i+1][j+1]*math.sqrt(T))
            d2 = d1 - BS_Vol_bid[i+1][j+1]*math.sqrt(T)
            tem1 = norm.cdf(d1)
            tem2 = norm.cdf(d2)
            MktPrice_bid[i+1, j+1] = spotFX*tem1 - math.exp(-dom_rp[i+1]*T)*Strikes_p[k]*tem2
            if OptType_p[k] == 2: MktPrice_bid[i+1, j+1] += math.exp(-dom_rp[i+1]*T)*Strikes_p[k] - spotFX
            d1 = (math.log(spotFX/Strikes_p[k])+(dom_rp[i+1]-for_rp[i+1]+0.5*BS_Vol_ask[i+1][j+1]*BS_Vol_ask[i+1][j+1])*T)
            d1 = d1 / (BS_Vol_ask[i+1][j+1]*math.sqrt(T))
            d2 = d1 - BS_Vol_ask[i+1][j+1]*math.sqrt(T)
            tem_bid = tem1
            tem1 = norm.cdf(d1)
            tem2 = norm.cdf(d2)
            MktPrice_ask[i+1, j+1] = spotFX*tem1 - math.exp(-dom_rp[i+1]*T)*Strikes_p[k]*tem2
            if OptType_p[k] == 2: MktPrice_ask[i+1, j+1] += math.exp(-dom_rp[i+1]*T)*Strikes_p[k] - spotFX
            print(f'  Exp {daysExp_p[i+1]: 5d}  Strike {Strikes_p[k]: .4f} Mkt Bid-Ask {MktPrice_bid[i+1,j+1]: .4f} {MktPrice_ask[i+1,j+1]: .4f}  N(d1) {tem_bid: .4f} {tem1: .4f}')
   
    
    # set input values
    # read in interest rates and FX forwards
    
    FXSV_ModelPricer(nThreads, nSimPerThread, nTimeSteps, nExpirations, \
	nStrikesPerExp, spotFX, spotV, daysExp_p, r_dom_p, r_for_p, params_p,\
    OptType_p, Strikes_p, OptPrice_p, StdErr_p)

    print(" The Monte Carlo Reesults for the FX SV Model ")
    for i in range(0,nExpirations):
        for j in range(0,nStrikesPerExp):
            k = (i+1)*(nStrikesPerExp+1)+j+1
            print(f' Exp {daysExp_p[i+1]: 5d} Strike {Strikes_p[k]: .4f}  Market {MktPrice_bid[i+1, j+1]: .6f} {MktPrice_ask[i+1, j+1]: .6f}  Model {OptPrice_p[k]: .6f} {StdErr_p[k]: .6f}')
            
