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
        self.log_kappa_q = 4.0/3*np.ones((self.T-1,self.p))
        self.log_kappa_sigma_sqq = 4/3*np.ones(self.T)
        self.mu_gamma_sd = 0.003
        self.v_gamma_sd = 0.003
    
        self.kappa_accept = 0
        self.kappa_count =0
        self.kappa_lambda_sigma_accept = 0
        self.kappa_sigma_sq_count =0
        self.sigma_sq_param_accept = np.zeros(self.p)
        self.sigma_sq_param_count = np.zeros(self.p)
        self.psi_param_accept = np.zeros(self.p)
        self.psi_param_count = np.zeros(self.p)
        self.mu_gamma_accept = 0
        self.mu_gamma_count = 0
        self.v_gamma_accept = 0
        self.v_gamma_account = 0
        self.sigma_sq1_accept = np.zeros(self.T)
        self.sigma_sq1_count = np.zeros(T)
        self.number_of_iteration = self.burnin + self.every*self.numbofits
    
        # beta 初始值
        
        
        self.beta = 

        self.check_star = 1
        
        for it in range(self.number_of_iteration):
            # iteration
            # print 

            if check_star ==1:
                # run iteration

            
            
             

    # it 是当前迭代次数
    def UpdatePsi(self,it):
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
                    lam1 = self.lambda_[jj]+self.kappa[ii-1][jj]
                    gam1 = self.lambda_[jj]/self.mu[jj] + self.delta[jj]
                    loglike =(lam1-1)*np.log(self.psi[ii][jj])-gam1*self.psi[ii][jj]
                    var1 = self.psi[ii][jj]*(1-self.rho_beta**2)
                    mean1 = self.rho_beta[jj]*np.sqrt(self.psi[ii][jj]/self.psi[ii-1][jj])*self.beta[ii-1][jj]
                    loglike = loglike -0.5*np.log(var1)-0.5*(self.beta[ii][jj]-mean1)**2/var1

                    lam1 = self.lambda_[jj]+self.kappa[ii-1][jj]
                    gam1 = self.lambda_[jj]/self.mu[jj]+self.delta[jj]
                    new_loglike = (lam1-1)*np.log(new_psi)-gam1*new_psi
                    var1 = new_psi*(1-self.rho_beta[jj]**2)
                    mean1 = self.rho_beta[jj]*np.sqrt(new_psi/self.psi[ii-1][jj])*self.beta[ii-1][jj]
                    new_loglike = new_loglike -0.5*np.log(var1)-0.5*(self.beta[ii][jj]-mean1)**2/var1
                log_accept = new_loglike-loglike +np.log(new_psi)-np.log(self.psi[ii][jj])
                accept =1 
                if np.isnan(log_accept) or np.isinf(log_accept):
                    accept =0
                elif log_accept <0:
                    accept = np.exp(log_accept)
                
                self.psi_sd[ii][jj] = self.psi_sd[ii][jj] + (accept-0.3)/((it+1)**0.6)
                    
                if np.random.random() <accept:
                    self.psi[ii][jj] = new_psi


    def UpdateKappa(self,it):
        for ii in range(self.T-1):
            for jj in range(self.p):

                new_kappa = self.kappa[ii][jj]+(2*np.ceil(np.random.random())-1)*stats.geom.rvs(1.0/(1+np.exp(self.log_kappa_q[ii][jj])))

                if new_kappa <0:
                    accept = 0
                else:
                    lam1 = self.lambda_[jj] + self.kappa[ii][jj]
                    gam1 = self.lambda_[jj]/self.mu[jj] + self.delta[jj]
                    loglike = lam1*np.log(gam1) - np.log(spec.gamma(lam1))+(lam1-1)*np.log(self.psi[ii+1],jj)
                    pnmean = self.psi[ii][jj] + new_kappa
                    loglike = loglike + self.kappa[ii][jj]*np.log(pnmean)- np.log(spec.gamma(self.kappa[ii][jj]+1))

                    lam1 = self.lambda_[jj]+ new_kappa 
                    gam1 = self.lambda_[jj]/self.mu[jj] + self.delta[jj]
                    new_loglike = lam1*np.log(gam1) - np.log(spec.gamma(lam1))+(lam-1)*np.log(self.psi[ii+1][jj])
                    pnmean = self.psi[ii][jj]*self.delta[jj]
                    new_loglike = new_loglike + new_kappa*np.log(pnmean)-np.log(spec.gamma(new_kappa+1))
                    log_accept = new_loglike - loglike 
                    accept =1
                    if np.isnan(log_accept) or np.isinf(log_accept):
                        accept =0
                    elif: log_accept <0:
                        accept = np.exp(log_accept)
                    
                self.kappa_accept = self.kappa_accept + accept 
                self.kappa_count = self.kappa_count +1
                
                if np.random.random() < accept:
                    self.kappa[ii][jj] = new_kappa
                self.log_kappa_q[ii][jj] = self.log_kappa_q[ii][jj] + 1.0/it**0.55*(accept-0.3)
                if np.isnan(self.kappa[ii][jj]) or np.isreal ==False:
                    stop

    def UpdateSigmaSq(self,it):
        for ii in range(self.T):
            chi1 = (self.y_list[ii] - sum(self.x_list[ii]*self.beta[ii]))**2
            if ii ==1:
                lam1 = self.kappa_sigma_sq[ii] + self.lambda_star -0.5
                psi1 = 2*(self.lambda_star/self.mu_sigma+self.rho_sigma/(1-self.rho_sigma)*self.lambda_sigma/self.mu_sigma)
            elif ii ==self.T-1:
                lam1 = self.kappa_sigma_sq[ii-1]+self.lambda_sigma -0.5
                psi1 = 2*(self.lambda_sigma/self.mu_sigma+self.rho_sigma/(1-self.rho_sigma)*self.lambda_sigma/self.mu_sigma)
            else:
                lam1= self.kappa_sigma_sq[ii]+self.kappa_sigma_sq[ii-1]+self.lambda_sigma-0.5
                psi1 = 2*(self.lambda_sigma/self.mu_sigma+2*self.rho_sigma/(1-self.rho_sigma)*self.lambda_sigma/self.mu_sigma)
            new_sigma_sq = self.sigma_sq[ii]*np.exp(np.exp(self.log_sigma_sq_sd[ii])*np.random.random())
            loglike = (lam1-1)*np.log(self.sigma_sq[ii])-0.5*chi1/self.sigma_sq[ii]-0.5*psi1*self.sigma_sq[ii]
            new_loglike = (lam1-1)*np.log(new_sigma_sq)-0.5*chi1/new_sigma_sq-0.5*psi1*new_sigma_sq
            log_accept = new_loglike - loglike +np.log(new_sigma_sq)-np.log(sigma_sq[ii])

            accept =1
            if np.isnan(log_accept) or np.isinf(log_accept):
                accept =0
            elif log_accept <0:
                accept = np.exp(log_accept)
            
            self.sigma_sq1_accept = self.sigma_sq1_accept +accept
            self.sigma_sq1_count = self.sigma_sq1_count +1

            if np.random.random() < accept :
                self.sigma_sq[ii] = new_sigma_sq
            self.log_sigma_sq_sd = self.log_sigma_sq_sd +1.0/it**0.55*(accept-0.3)

    def UpdateKappaSigmaSq(self,it):
        for ii in range(self.T-1):
            new_kappa_sigma_sq = self.kappa_sigma_sq[ii]+(2*np.ceil(np.random.random)-1)*stats.geom.rvs(1.0/(1+np.exp(self.log_kappa_sigma_sqq[ii])))

            if new_kappa_sigma_sq <0:
                accept = 0
            else:
                lam1 = self.lambda_sigma + self.kappa_sigma_sq[ii]
                gam1 = self.lambda_sigma/self.mu_sigma + self.rho_sigma/(1-self.rho_sigma)*self.lambda_sigma/self.mu_sigma
                loglike = lam1*np.log(gam1)-np.log(spec.gamma(lam1))+(lam1-1)*np.log(self.sigma_sq[ii+1])
                pnmean = self.sigma_sq[ii]*self.rho_sigma/(1-self.rho_sigma)*self.lambda_sigma/self.mu_sigma
                loglike = loglike + self.kappa_sigma_sq[ii]*np.log(pnmean)- np.log(spec.gamma(self.kappa_sigma_sq[ii])+1)

                lam1 = self.lambda_sigma + new_kappa_sigma_sq
                gam1 = self.lambda_sigma/self.mu_sigma + self.rho_sigma/(1-self.rho_sigma)*self.lambda_sigma/self.mu_sigma
                new_loglike = lam1*np.log(gam1)-np.log(spec.gamma(lam1))+(lam1-1)*np.log(self.sigma_sq[ii+1])
                pnmean = self.sigma_sq[ii]*self.rho_sigma/(1-self.rho_sigma)*self.lambda_sigma/self.mu_sigma
                new_loglike = new_loglike + new_kappa_sigma_sq*np.log(pnmean)-np.log(spec.gamma(new_kappa_sigma_sq+1))
                log_accept = new_loglike - loglike
                accept =1
                if np.isnan(log_accept) or np.isinf(log_accept):
                    accept = 0
                elif log_accept <0:
                    accept = np.exp(log_accept)

            self.kappa_lambda_sigma_accept = self.kappa_lambda_sigma_accept + accept 
            self.kappa_sigma_sq_count = self.kappa_sigma_sq_count +1
            if np.random.random()<accept :
                self.kappa_sigma_sq[ii] = new_kappa_sigma_sq 
            self.log_kappa_sigma_sqq[ii] = log_kappa_sigma_sqq[ii]+1.0/it**0.55*(accept-0.3)

            if np.isnan(self.kappa_sigma_sq[ii]) or np.isreal(self.kappa_sigma_sq[ii]==0):
                stop

    def UpdateTheta(self,it):
        zstar = np.random.random(self.p) < 5/self.p
        y_star = self.y_list - (sum(self.x_list[zstar==0]))

            

