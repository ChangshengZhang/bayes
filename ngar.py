#!/usr/bin/python
# File Name: ngar.py
# Author: Changsheng Zhang
# mail: zhangcsxx@gmail.com
# Created Time: Mon Dec  7 00:26:38 2015

#########################################################################

import numpy as np
import copy
from scipy import stats
import scipy.special as spec
import math
import random
import copy
import datetime
import matplotlib.pyplot as plt
from rpy2.robjects.packages import importr 
import rpy2.robjects as robjects

rgig = importr("GeneralizedHyperbolic")

class NGAR():
    def __init__(self,x_list,y_list,mu_mean,mu_lambda_star,burnin,numofits,every):
        # x_list size: T*p; y_list 
        self.x_list = x_list
        self.y_list = y_list
        self.mu_mean = mumean 
        self.mu_lambda_star = mu_lambda_star
        self.numofits = numofits
        self.every = every 
        
        #initial value
        self.T = len(self.x_list)
        self.p = len(self.x_list[0])
        self.rho = 0.97*np.ones(self.p)
        self.rho_beta = 0.97*np.ones(self.p)
        self.lambda_ = self.mu_lambda_star*np.ones(self.p)
        self.mu = self.mu_mean*np.ones(self.p)
        self.delta = self.rho/(1-self.rho)*self.lambda_/self.mu

        self.start_samples = 500
        self.start_adap = 1000

        self.new_beta = np.zeros((self.T,self.p))
        self.psi = np.zeros((self.T,self.p))
        self.kappa = np.zeros((self.T,self.p))
        for ii in range(self.p):
            for jj in range(self.T):
                self.kappa[jj][ii] = stats.poisson.rvs(self.delta[ii]*self.psi[jj][ii],size=1)[0]

        self.kappa_sigma_sq = np.ones(self.T)
        self.lambda_sigma = 3
        self.mu_sigma = 0.03
        self.rho_sigma = 0.95
        self.sigma_sq = self.mu_sigma*np.ones(T)
        self.lambda_star =1
        self.mu_star = 1

        self.psi_sd = 0.01*np.ones((self.T,self.p))
        self.log_lambda_sd = np.log(0.1)*np.ones(self.p)
        self.log_mena_sd = np.log(0.1)*np.ones(self.p)
        self.log_scale = 0.5*np.log(2.4^2/4)*np.ones(self.p)
        self.log_rho_beta_sd = np.log(0.1)*np.ones(self.p)
        self.log_rho_sd = np.log(0.1)*np.ones(self.p)
        self.log_rho_sigma_sd = np.log(0.01)
        self.log_lambda_sigma_sd = np.log(0.001)
        self.log_gamma_sigma_sq_sd = np.log(0.001)
        self.log_scale_sigma_sq = 0.5*np.log(2.4^2/3)
        self.log_sigma_sq_sd = np.log(0.001)*np.ones(self.T)
        self.log_kappa_q = 4/3*np.ones((self.T-1,self.p))
        self.log_kappa_sigma_sqq = 4/3*np.ones(self.T)
        self.mu_gamma_sd = 0.003
        self.v_gamma_sd = 0.003
    
        self.number_of_iteration = self.burnin + self.every*self.numbofits
    
        # beta 初始值
        
        
        self.beta = 

        self.check_star = 1
        
        for ii in range(self.number_of_iteration):
            # iteration
            # print 

            if check_star ==1:
                # run iteration

            
            
             


    def UpdatePsi(self):
        for ii in range(self.T):
            for jj in range(self.p):
                new_psi = self.psi[ii][jj]*np.exp(self.psi_sd[ii][jj]*np.random.random())

                if ii ==0:
                    loglike = (self.lambda_[jj]-1)*np.log(self.psi[0][jj])-self.lambda_[jj]*self.psi[0][jj]/self.mu[jj]
                    pnmean = self.psi[ii][jj]*self.delta[jj]
                    loglike = loglike - pnmean + self.kappa[ii][jj]*np.log(pnmean)-0.5*np.log(self.psi[0][jj])-0.5*self.beta[0][jj]**2/self.psi[0][jj]
                    var1 = self.psi[1][jj]*(1-self.rho_beta[jj]**2)
                    mean1 = self.rho_beta[jj]*np.sqrt(self.phi[1][jj]/self.phi[0][jj])*self.beta[0][jj]
                    loglike = loglike -0.5*(self.beta[1][jj]-mean1)**2/var1

                    new_loglike = (self.lambda_[jj]-1)*np.log(new_psi)-self.lambda_[jj]*new_psi/self.mu[jj]
                    pnmean = new_psi*self.delta[jj]
                    new_loglike = new_loglike - pnmean+self.kappa[ii][jj]*np.log(pnmean)-0.5*np.log(new_psi)-0.5*self.beta[0][jj]**2/new_psi
                    var1 = self.psi[1][jj]*(1-self.rho_beta[jj]**2)
                    mean1 = self.rho_beta[jj]*np.sqrt(self.psi[1][jj]/new_psi)*self.beta[0][jj]
                    new_loglike = new_loglike - 0.5*(self.beta[1][jj]-mean1)**2/var1
                
                elif i<T-1:
                    lam1 = self.lambda_[jj]+self.kappa[ii-1][jj]
                    gam1 = self.lambda_[jj]/self.mu[jj] +self.delta[jj]
                    loglike = (lam1-1)*np.log(self.psi[ii][jj])-gam1*self.psi[ii][jj]-self.psi[ii][jj]*self.delta[jj]+self.kappa[ii][jj]*np.log(self.psi[ii][jj]*self.delta[jj])
                    var1 = self.psi[ii][jj]*(1-self.rho_beta[jj]**2)
                    mean1 = self.rho_beta*np.sqrt(self.psi[ii][jj]/self.psi[ii-1][jj])*self.beta[ii-1][jj]
                    loglike = loglike - 0.5*np.log(var1)-0.5*(self.beta[ii][jj]-mean1)**2/var1
                    var1 = self.psi[ii+1][jj]*(1-self.rho_beta[jj]**2)
                    mean1 = self.rho_beta[jj]*np.sqrt(self.psi[ii+1][jj]/self.psi[ii][jj])*self.beta[ii][jj]
                    loglike = loglike -0.5*(self.beta[ii+1][jj]-mean1)**2/var1
                    
                    lam1 = self.lambda_[jj]+ self.kappa[ii-1][jj]
                    gam1 = self.lambda_[jj]/self.mu[jj] +self.delta[jj]
                    new_loglike = (lam1-1)*np.log(new_psi)- gam1*new_psi-new_psi*self.delta[jj]+self.kappa[ii][jj]*np.log(new_psi*self.delta[jj])
                    var1 = new_psi*(1-self.rho_beta[jj]**2)
                    mean1 = self.rho_beta[jj]*np.sqrt(new_psi/self.psi[ii-1][jj])*self.beta[ii-1][jj]
                    new_loglike = new_loglike-0.5*(self.beta[ii+1][jj]-mean1)**2/var1

                else:
                    




