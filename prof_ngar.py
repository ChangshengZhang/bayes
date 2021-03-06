#!/usr/bin/python
# -*- coding: utf-8 -*-
# File Name: ngar.py
# Author: Changsheng Zhang
# mail: zhangcsxx@gmail.com
# Created Time: Mon Dec  7 00:26:38 2015

#########################################################################

import numpy as np
import copy
import scipy.io as sio
import scipy.special as spec
import math
import random
import datetime
import matplotlib.pyplot as plt
import copy
import time

class NGAR():
    def __init__(self,x_path,y_path,mu_mean,mu_lambda_star,burnin,numofits,every):
        # x_list size: T*p; y_list
        self.x_list,self.y_list = self.LoadData(x_path, y_path)
        self.mu_mean = mu_mean
        self.mu_lambda_star = mu_lambda_star
        self.numofits = numofits
        self.every = every
        self.burnin = burnin

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
        self.psi = np.ones((self.T,self.p))
        self.kappa = np.zeros((self.T,self.p))
        for ii in range(self.p):
            for jj in range(self.T):
                self.kappa[jj][ii] = np.random.poisson(self.delta[ii]*self.psi[jj][ii])

        self.kappa_sigma_sq = np.ones(self.T)
        self.lambda_sigma = 3
        self.mu_sigma = 0.03
        self.rho_sigma = 0.95
        self.sigma_sq = self.mu_sigma*np.ones(self.T)
        self.lambda_star =1
        self.mu_star = 1

        self.psi_sd = 0.01*np.ones((self.T,self.p))
        self.log_lambda_sd = np.log(0.1)*np.ones(self.p)
        self.log_mean_sd = np.log(0.1)*np.ones(self.p)
        self.log_scale = 0.5*np.log(2.4**2/4)*np.ones(self.p)
        self.log_rho_beta_sd = np.log(0.1)*np.ones(self.p)
        self.log_rho_sd = np.log(0.1)*np.ones(self.p)
        self.log_rho_sigma_sd = np.log(0.01)
        self.log_lambda_sigma_sd = np.log(0.001)
        self.log_gamma_sigma_sq_sd = np.log(0.001)
        self.log_scale_sigma_sq = 0.5*np.log(2.4**2/3)
        self.log_sigma_sq_sd = np.log(0.001)*np.ones(self.T)
        self.log_kappa_q = 4.0/3*np.ones((self.T-1,self.p))
        self.log_kappa_sigma_sqq = 4.0/3*np.ones(self.T)
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
        self.v_gamma_count = 0
        self.sigma_sq1_accept = np.zeros(self.T)
        self.sigma_sq1_count = np.zeros(self.T)
        self.number_of_iteration = self.burnin + self.every*self.numofits

        self.sum_1 = np.zeros((4,self.p))
        self.sum_2 = np.zeros((10,self.p))
        self.sum_1_sigma_sq = np.zeros(3)
        self.sum_2_sigma_sq = np.zeros(6)
        self.limit = 0.9999

        self.hold_psi = []
        self.hold_beta = []
        self.hold_sigma_sq = []
        self.hold_lambda = []
        self.hold_mu = []
        self.hold_rho_beta =[]
        self.hold_rho = []
        self.hold_lambda_sigma = []
        self.hold_mu_sigma =[]
        self.hold_rho_sigma = []
        self.hold_lambda_star = []
        self.hold_mu_star = []

        # beta 初始值
        self.mean_kf,self.var_kf,loglike = self.KalmanFilter(self.x_list,self.y_list,self.psi,self.sigma_sq,self.rho_beta)

        chol_star = np.linalg.cholesky(self.var_kf[:,:,self.T-1])

        self.new_beta[self.T-1]=self.mean_kf[:,self.T-1] + np.dot(chol_star,np.random.normal(size=len(chol_star[0])))
        # 测试时间
        start_time = datetime.datetime.now()
        for jj in range(self.T-2,-1,-1):
            Gkal = np.diag(self.rho_beta*np.sqrt(1.0*self.psi[jj+1]/self.psi[jj]))
            invQ = np.diag(1.0/(1-self.rho_beta**2))*1.0/self.psi[jj+1]
            var_fb = np.linalg.inv(np.linalg.inv(self.var_kf[:,:,jj])+ np.asarray(np.asmatrix(Gkal).T*np.asmatrix(invQ)*np.asmatrix(Gkal)))
            mean_fb = np.dot(var_fb,np.dot(np.linalg.inv(self.var_kf[:,:,jj]),self.mean_kf[:,jj])+ np.dot(np.asarray(np.asmatrix(Gkal).T*np.asmatrix(invQ)),self.new_beta[jj+1]))

            chol_star = np.linalg.cholesky(var_fb)
            self.new_beta[jj] = mean_fb + np.dot(np.asarray(chol_star),np.array(np.random.normal(size=len(chol_star[0]))))

        self.beta = copy.deepcopy(self.new_beta[:,0:self.p])

        print self.beta

        self.check_star = 1

        #注意，迭代次数的下标是从1开始
        for it in range(1,self.number_of_iteration+1):
            if it ==10:
                break
            # iteration
            print "start %s th iteration:\n" %it
            if self.check_star ==1:
                # run iteration
                self.UpdatePsi(it)
                self.UpdateKappa(it)
                self.UpdateSigmaSq(it)
                self.UpdateKappaSigmaSq(it)
            self.UpdateTheta(it)
            self.UpdateMuStar(it)
            self.UpdateLambdaStar(it)
            self.UpdateOutput(it,self.burnin,self.every)
            if it% 1000==0:
                end_time = datetime.datetime.now()
                print (end_time-start_time).seconds
                time.sleep(5)

        print "done."

    def KalmanFilter(self,x_list,y_list,psi,sigma_sq,rho_beta):
        p = len(x_list[0])
        T = len(y_list)
        mean_kf = np.zeros((p,T))
        var_kf = np.zeros((p,p,T))

        aminus = np.zeros(p)
        pminus = np.diag(psi[0])
        x_star_list = x_list[0]
        e = y_list[0] - np.dot(x_star_list,aminus)
        invF = 1.0/(sigma_sq[0]+ np.asarray(np.asmatrix(x_star_list)*np.asmatrix(pminus)*np.asmatrix(x_star_list).T)[0][0])
        mean_kf[:,0] = aminus + np.dot(np.asarray(pminus),np.asarray(x_star_list))*invF*e

        var_kf[:,:,0] = pminus - np.dot(np.asarray(np.asmatrix(np.dot(pminus,x_star_list)*invF).T*np.asmatrix(x_star_list)),pminus)
        loglike = -0.5*e**2*invF+ 0.5*np.log(invF)

        for ii in range(1,T):
            x_star_list = x_list[ii]
            Q = np.diag((1-rho_beta**2)*psi[ii])
            Gkal = np.diag(rho_beta*np.sqrt(1.0*psi[ii]/psi[ii-1]))
            aminus = np.dot(Gkal,mean_kf[:,ii-1])
            pminus = np.asarray(np.asmatrix(np.dot(Gkal,var_kf[:,:,ii-1]))*np.asmatrix(Gkal).T)+Q
            e = y_list[ii] - np.dot(x_star_list,aminus)
            invF = 1.0/(sigma_sq[ii]+ np.dot(np.dot(x_star_list,pminus),x_star_list))
            mean_kf[:,ii] = aminus +np.dot(pminus,x_star_list)*invF*e
            var_kf[:,:,ii] = pminus -np.dot(np.asarray(np.asmatrix(np.dot(pminus,x_star_list)*invF).T*np.asmatrix(x_star_list)),pminus)
            loglike = loglike -0.5*e**2*invF+0.5*np.log(invF)

        return mean_kf,var_kf,loglike


    # it 是当前迭代次数
    def UpdatePsi(self,it):
        for ii in xrange(self.T):
            for jj in xrange(self.p):
                new_psi = self.psi[ii][jj]*np.exp(self.psi_sd[ii][jj]*np.random.normal())
                new_loglike = 0
                loglike = 0

                if ii ==0:
                    loglike = (self.lambda_[jj]-1)*np.log(self.psi[0][jj])-1.0*self.lambda_[jj]*self.psi[0][jj]/self.mu[jj]
                    pnmean = self.psi[ii][jj]*self.delta[jj]
                    loglike = loglike - pnmean + self.kappa[ii][jj]*np.log(pnmean)-0.5*np.log(self.psi[0][jj])-0.5*self.beta[0][jj]**2/self.psi[0][jj]
                    var1 = self.psi[1][jj]*(1-self.rho_beta[jj]**2)
                    mean1 = self.rho_beta[jj]*np.sqrt(self.psi[1][jj]/self.psi[0][jj])*self.beta[0][jj]
                    loglike = loglike -0.5*(self.beta[1][jj]-mean1)**2/var1

                    new_loglike = (self.lambda_[jj]-1)*np.log(new_psi)-self.lambda_[jj]*new_psi/self.mu[jj]
                    pnmean = new_psi*self.delta[jj]
                    new_loglike = new_loglike - pnmean+self.kappa[ii][jj]*np.log(pnmean)-0.5*np.log(new_psi)-0.5*self.beta[0][jj]**2/new_psi
                    var1 = self.psi[1][jj]*(1-self.rho_beta[jj]**2)
                    mean1 = self.rho_beta[jj]*np.sqrt(self.psi[1][jj]/new_psi)*self.beta[0][jj]
                    new_loglike = new_loglike - 0.5*(self.beta[1][jj]-mean1)**2/var1

                elif ii<self.T-1:
                    lam1 = self.lambda_[jj]+self.kappa[ii-1][jj]
                    gam1 = self.lambda_[jj]/self.mu[jj] +self.delta[jj]
                    loglike = (lam1-1)*np.log(self.psi[ii][jj])-gam1*self.psi[ii][jj]-self.psi[ii][jj]*self.delta[jj]+self.kappa[ii][jj]*np.log(self.psi[ii][jj]*self.delta[jj])
                    var1 = self.psi[ii][jj]*(1-self.rho_beta[jj]**2)
                    mean1 = self.rho_beta[jj]*np.sqrt(self.psi[ii][jj]/self.psi[ii-1][jj])*self.beta[ii-1][jj]
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
                    var1 = self.psi[ii][jj]*(1-self.rho_beta[jj]**2)
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

                self.psi_sd[ii][jj] = self.psi_sd[ii][jj] + (accept-0.3)/(it**0.6)

                if np.random.random() <accept:
                    self.psi[ii][jj] = new_psi


    def UpdateKappa(self,it):
        for ii in xrange(self.T-1):
            for jj in xrange(self.p):

                new_kappa = self.kappa[ii][jj]+(2*np.ceil(2*np.random.random())-3)*np.random.geometric(1.0/(1+np.exp(self.log_kappa_q[ii][jj])))

                if new_kappa <0:
                    accept = 0
                else:
                    lam1 = self.lambda_[jj] + self.kappa[ii][jj]
                    gam1 = self.lambda_[jj]/self.mu[jj] + self.delta[jj]
                    loglike = lam1*np.log(gam1) - np.log(spec.gamma(lam1))+(lam1-1)*np.log(self.psi[ii+1][jj])
                    pnmean = self.psi[ii][jj] + new_kappa
                    loglike = loglike + self.kappa[ii][jj]*np.log(pnmean)- np.log(spec.gamma(self.kappa[ii][jj]+1))

                    lam1 = self.lambda_[jj]+ new_kappa
                    gam1 = self.lambda_[jj]/self.mu[jj] + self.delta[jj]
                    new_loglike = lam1*np.log(gam1) - np.log(spec.gamma(lam1))+(lam1-1)*np.log(self.psi[ii+1][jj])
                    pnmean = self.psi[ii][jj]*self.delta[jj]
                    new_loglike = new_loglike + new_kappa*np.log(pnmean)-np.log(spec.gamma(new_kappa+1))
                    log_accept = new_loglike - loglike
                    accept =1
                    if np.isnan(log_accept) or np.isinf(log_accept):
                        accept =0
                    elif log_accept <0:
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
            new_sigma_sq = self.sigma_sq[ii]*np.exp(np.exp(self.log_sigma_sq_sd[ii])*np.random.normal())
            loglike = (lam1-1)*np.log(self.sigma_sq[ii])-0.5*chi1/self.sigma_sq[ii]-0.5*psi1*self.sigma_sq[ii]
            new_loglike = (lam1-1)*np.log(new_sigma_sq)-0.5*chi1/new_sigma_sq-0.5*psi1*new_sigma_sq
            log_accept = new_loglike - loglike +np.log(new_sigma_sq)-np.log(self.sigma_sq[ii])

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
            new_kappa_sigma_sq = self.kappa_sigma_sq[ii]+(2*np.ceil(2*np.random.random())-3)*np.random.geometric(1.0/(1+np.exp(self.log_kappa_sigma_sqq[ii])))

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
            self.log_kappa_sigma_sqq[ii] = self.log_kappa_sigma_sqq[ii]+1.0/it**0.55*(accept-0.3)

            if np.isnan(self.kappa_sigma_sq[ii]) or np.isreal(self.kappa_sigma_sq[ii])==0:
                stop


    def UpdateTheta(self,it):
        z_star = np.random.random(self.p) < 5.0/self.p
        y_star = self.y_list - (self.x_list[:,z_star==0]*self.beta[:,z_star==0]).sum(axis=1)
        x_star = self.x_list[:,z_star==1]
        psi_star = copy.deepcopy(self.psi[:,z_star==1])
        kappa_star = copy.deepcopy(self.kappa[:,z_star==1])

        self.mean_kf,self.var_kf,loglike = self.KalmanFilter(x_star,y_star,psi_star,self.sigma_sq,self.rho_beta[z_star==1])
        xi_star = np.array([np.log(self.lambda_sigma),np.log(self.mu_sigma),np.log(self.rho_sigma)-np.log(1-self.rho_sigma)])

        if it <100:
            new_xi_star = xi_star +np.array([np.exp(self.log_lambda_sigma_sd),np.exp(self.log_gamma_sigma_sq_sd),np.exp(self.log_rho_sigma_sd)])*np.random.normal(size=3)
        else:
            var_star_1 =(np.array([[self.sum_2_sigma_sq[0],self.sum_2_sigma_sq[1],self.sum_2_sigma_sq[3]],[self.sum_2_sigma_sq[1],self.sum_2_sigma_sq[2],self.sum_2_sigma_sq[4]],[self.sum_2_sigma_sq[3],self.sum_2_sigma_sq[4],self.sum_2_sigma_sq[5]]])- np.asarray(np.asmatrix(self.sum_1_sigma_sq).T*np.asmatrix(self.sum_1_sigma_sq)/it))/(it-1)
            new_xi_star = xi_star + np.dot(np.linalg.cholesky(np.exp(self.log_scale_sigma_sq)*var_star_1),np.random.normal( size= 3))

        new_lambda_sigma = np.exp(new_xi_star[0])
        new_mu_sigma = np.exp(new_xi_star[1])
        new_rho_sigma = np.exp(new_xi_star[2])/(1+np.exp(new_xi_star[2]))

        if new_rho_sigma > self.limit:
            accept =0
        else:
            new_sigma_sq = copy.deepcopy(self.sigma_sq)
            new_kappa_sigma_sq = copy.deepcopy(self.kappa_sigma_sq)

            new_sigma_sq[0] = self.sigma_sq[0]*new_mu_sigma/self.mu_sigma
            if new_lambda_sigma >self.lambda_sigma:
                new_sigma_sq[0] = new_sigma_sq[0]+ np.random.gamma(new_lambda_sigma-self.lambda_sigma,scale= new_mu_sigma/new_lambda_sigma)
            else:
                new_sigma_sq[0] = new_sigma_sq[0]*np.random.beta(new_lambda_sigma,self.lambda_sigma-new_lambda_sigma)


            for ii in range(1,self.T):
                old_mean = self.rho_sigma/(1-self.rho_sigma)*self.lambda_sigma/self.mu_sigma*self.sigma_sq[ii-1]
                new_mean = new_rho_sigma/(1-new_rho_sigma)*new_lambda_sigma/new_mu_sigma*new_sigma_sq[ii-1]
                if new_mean > old_mean:
                    new_kappa_sigma_sq[ii-1] = self.kappa_sigma_sq[ii-1]+ np.random.poisson(new_mean-old_mean)
                else:
                    new_kappa_sigma_sq[ii-1] = np.random.binomial(self.kappa_sigma_sq[ii-1],1.0*new_mean/old_mean)
                old_lam = self.kappa_sigma_sq[ii-1]+ self.lambda_sigma
                old_gam = self.rho_sigma/(1-self.rho_sigma)*self.lambda_sigma/self.mu_sigma+self.lambda_sigma/self.mu_sigma
                new_lam = new_kappa_sigma_sq[ii-1]+new_lambda_sigma
                new_gam = new_rho_sigma/(1-new_rho_sigma)*new_lambda_sigma/new_mu_sigma+new_lambda_sigma/new_mu_sigma
                new_sigma_sq[ii] = self.sigma_sq[ii]*old_gam/new_gam


                if new_lam > old_lam:
                    new_sigma_sq[ii] = new_sigma_sq[ii] + np.random.gamma(new_lam-old_lam,scale = 1.0/new_gam)
                else:
                    new_sigma_sq[ii] = new_sigma_sq[ii]*np.random.beta(new_lam,old_lam-new_lam)

            new_mean_kf,new_var_kf,new_loglike = self.KalmanFilter(x_star,y_star,psi_star,new_sigma_sq,self.rho_beta[z_star==1])

            log_accept = new_loglike -loglike+3*(np.log(new_lambda_sigma)-np.log(self.lambda_sigma))-1.0*(new_lambda_sigma-self.lambda_sigma)+np.log(new_mu_sigma)-np.log(self.mu_sigma)
            log_accept = log_accept-(1+0.5)*np.log(1+new_mu_sigma)+(1+0.5)*np.log(1+self.mu_sigma)
            log_accept = log_accept+np.log(1.0/self.rho_sigma+1.0/(1-self.rho_sigma))-np.log(1.0/new_rho_sigma+1.0/(1-new_rho_sigma))
            log_accept = log_accept +40*0.95*(np.log(new_rho_sigma)-np.log(self.rho_sigma))+40*0.05*(np.log(1-new_rho_sigma)-np.log(1-self.rho_sigma))

            accept =1
            if np.isnan(log_accept) or np.isinf(log_accept):
                accept =0
            elif log_accept<0:
                accept = np.exp(log_accept)

        self.sigma_sq_param_accept = self.sigma_sq_param_accept +accept
        self.sigma_sq_param_count = self.sigma_sq_param_count +1

        if np.random.random() < accept:
            self.lambda_sigma = copy.deepcopy(new_lambda_sigma)
            self.mu_sigma = copy.deepcopy(new_mu_sigma)
            self.rho_sigma = copy.deepcopy(new_rho_sigma)
            self.kappa_sigma_sq = copy.deepcopy(new_kappa_sigma_sq)
            self.sigma_sq = copy.deepcopy(new_sigma_sq)
            loglike = copy.deepcopy(new_loglike)
            self.mean_kf = copy.deepcopy(new_mean_kf)
            self.var_kf = copy.deepcopy(new_var_kf)

        if it <100:
            self.log_lambda_sigma_sd = self.log_lambda_sigma_sd+1.0/(it**0.55)*(accept-0.3)
            self.log_gamma_sigma_sq_sd = self.log_gamma_sigma_sq_sd +1.0/it**0.55*(accept-0.3)
            self.log_rho_sigma_sd = self.log_rho_sigma_sd+1.0/it**0.55*(accept-0.3)
        else:
            self.log_scale_sigma_sq = self.log_scale_sigma_sq +1.0/(it-99)**0.55*(accept-0.3)
        x0 = np.log(self.lambda_sigma)
        x1 = np.log(self.mu_sigma)
        x2 = np.log(self.rho_sigma) -np.log(1-self.rho_sigma)
        self.sum_1_sigma_sq[0] = self.sum_1_sigma_sq[0] +x0
        self.sum_1_sigma_sq[1] = self.sum_1_sigma_sq[1] +x1
        self.sum_1_sigma_sq[2] = self.sum_1_sigma_sq[2] +x2
        self.sum_2_sigma_sq[0] = self.sum_2_sigma_sq[0] + x0**2
        self.sum_2_sigma_sq[1] = self.sum_2_sigma_sq[1] + x0*x1
        self.sum_2_sigma_sq[2] = self.sum_2_sigma_sq[2] + x1**2
        self.sum_2_sigma_sq[3] = self.sum_2_sigma_sq[3] + x0*x2
        self.sum_2_sigma_sq[4] = self.sum_2_sigma_sq[4] + x1*x2
        self.sum_2_sigma_sq[5] = self.sum_2_sigma_sq[5] + x2**2


        counter = -1

        for jj in range(self.p):
            if z_star[jj]== True:
                counter = counter +1
                xi_star = [np.log(self.lambda_[jj]),np.log(self.mu[jj]),np.log(self.rho[jj])-np.log(1-self.rho[jj]),np.log(self.rho_beta[jj])-np.log(1-self.rho_beta[jj])]

                if it < self.start_adap:
                    new_xi_star = xi_star +[np.exp(self.log_lambda_sd[jj]),np.exp(self.log_mean_sd[jj]),np.exp(self.log_rho_sd[jj]),np.exp(self.log_rho_beta_sd[jj])]*np.random.normal(size=4)
                else:
                    sxx = np.array([[self.sum_2[0][jj],self.sum_2[1][jj],self.sum_2[3][jj],self.sum_2[6][jj]],[self.sum_2[1][jj],self.sum_2[2][jj],self.sum_2[4][jj],self.sum_2[7][jj]],[self.sum_2[3][jj],self.sum_2[4][jj],self.sum_2[5][jj],self.sum_2[8][jj]],[self.sum_2[6][jj],self.sum_2[7][jj],self.sum_2[8][jj],self.sum_2[9][jj]]])

                    var_star_1 = (sxx-np.asarray(np.asmatrix(self.sum_1[:,jj]).T*np.asmatrix(self.sum_1[:,jj]))/(it-self.start_samples))/(it-self.start_samples-1)
                    new_xi_star = xi_star + np.dot(np.linalg.cholesky(np.exp(self.log_scale[jj])*var_star_1),np.random.normal(size=4))

                new_lambda = np.exp(new_xi_star[0])
                new_mu = np.exp(new_xi_star[1])
                new_rho = np.exp(new_xi_star[2])/(1+np.exp(new_xi_star[2]))
                new_rho_beta = copy.deepcopy(self.rho_beta)
                new_rho_beta[jj] = np.exp(new_xi_star[3])/(1+np.exp(new_xi_star[3]))
                new_delta = new_rho/(1-new_rho)*new_lambda/new_mu


                if new_rho_beta[jj] >self.limit or new_rho >self.limit:
                    accept =0
                else:
                    new_psi_star = copy.deepcopy(psi_star)
                    new_kappa_star = copy.deepcopy(kappa_star)

                    new_psi_star[0][counter] = psi_star[0][counter]*new_mu/self.mu[jj]
                    if new_lambda > self.lambda_[jj]:
                        new_psi_star[0][counter] = new_psi_star[0][counter]+ np.random.gamma(new_lambda-self.lambda_[jj],scale= new_mu/new_lambda)
                    else:
                        new_psi_star[0][counter] = new_psi_star[0][counter]*np.random.beta(new_lambda,self.lambda_[jj]-new_lambda)
                    for ii in range(1,self.T):
                        old_mean = self.delta[jj]*psi_star[ii-1][counter]
                        new_mean = new_delta*new_psi_star[ii-1][counter]

                        if new_mean > old_mean:
                            new_kappa_star[ii-1][counter] = kappa_star[ii-1][counter]+np.random.poisson(new_mean-old_mean)
                        else:
                            new_kappa_star[ii-1][counter] = np.random.binomial(kappa_star[ii-1][counter],1.0*new_mean/old_mean)

                        old_lam = kappa_star[ii-1][counter]+ self.lambda_[jj]
                        old_gam = self.delta[jj] + self.lambda_[jj]/self.mu[jj]
                        new_lam = new_kappa_star[ii-1][counter] + new_lambda
                        new_gam = new_delta + new_lambda/new_mu

                        new_psi_star[ii][counter] = psi_star[ii][counter]*old_gam/new_gam

                        if new_lam > old_lam:
                            new_psi_star[ii][counter] = new_psi_star[ii][counter] + np.random.gamma(new_lam-old_lam,scale= 1.0/new_gam)
                        else:
                            new_psi_star[ii][counter] = new_psi_star[ii][counter]*np.random.beta(new_lam,old_lam-new_lam)

                    new_mean_kf,new_var_kf,new_loglike = self.KalmanFilter(x_star,y_star,new_psi_star,self.sigma_sq,new_rho_beta[z_star==1])
                    log_accept = new_loglike - loglike

                    if jj ==0:
                        log_accept = log_accept +2*np.log(new_lambda)-2*np.log(self.lambda_[jj])-4*np.log(0.5+new_lambda)+4*np.log(0.5+self.lambda_[jj])+np.log(new_mu)-np.log(self.mu[jj])
                        log_accept = log_accept -1.5*np.log(1+new_mu)+1.5*np.log(1+self.mu[jj])
                    else:
                        log_accept = log_accept +2*np.log(new_lambda)- 2*np.log(self.lambda_[jj])- 4*np.log(0.5+new_lambda) + 4*np.log(0.5+self.lambda_[jj])
                        log_accept = log_accept + self.lambda_star*(np.log(new_mu)-np.log(self.mu[jj])) - self.lambda_star/self.mu_star*(new_mu-self.mu[jj])

                    log_accept = log_accept +np.log(1.0/self.rho[jj]+1.0/(1-self.rho[jj]))- np.log(1.0/new_rho+1.0/(1-new_rho))
                    log_accept = log_accept+80*0.97*(np.log(new_rho)-np.log(self.rho[jj]))+80*0.03*(np.log(1-new_rho)-np.log(1-self.rho[jj]))
                    log_accept = log_accept + np.log(1.0/self.rho_beta[jj]+1.0/(1-self.rho_beta[jj]))-np.log(1.0/new_rho_beta[jj]+1.0/(1-new_rho_beta[jj]))
                    log_accept = log_accept + 80*0.97*(np.log(new_rho_beta[jj])-np.log(self.rho_beta[jj]))+80*0.03*(np.log(1 -new_rho_beta[jj])-np.log(1 -self.rho_beta[jj]))

                    accept =1
                    if np.isnan(log_accept) or np.isinf(log_accept):
                        accept = 0
                    elif log_accept<0:
                        accept = np.exp(log_accept)

                self.psi_param_accept[jj] = self.psi_param_accept[jj] + accept
                self.psi_param_count[jj] = self.psi_param_count[jj] +1

                if np.random.random() < accept:
                    self.lambda_[jj] = new_lambda
                    self.mu[jj] = new_mu
                    self.rho[jj] = new_rho
                    self.rho_beta[jj] = new_rho_beta[jj]
                    self.delta[jj] = new_delta
                    psi_star[:,counter] = copy.deepcopy(new_psi_star[:,counter])
                    kappa_star[:,counter] = copy.deepcopy(new_kappa_star[:,counter])
                    loglike = new_loglike
                    self.mean_kf = copy.deepcopy(new_mean_kf)
                    self.var_kf = copy.deepcopy(new_var_kf)

                if it < self.start_adap:
                    self.log_lambda_sd[jj] = self.log_lambda_sd[jj] +1.0/it**0.55*(accept-0.3)
                    self.log_mean_sd[jj] = self.log_mean_sd[jj] +1.0/it**0.55*(accept-0.3)
                    self.log_rho_sd[jj] = self.log_rho_sd[jj]+ 1.0/it**0.55*(accept-0.3)
                    self.log_rho_beta_sd[jj] = self.log_rho_beta_sd[jj] +1.0/it**0.55*(accept-0.3)

                else:
                    self.log_scale[jj] = self.log_scale[jj] + 1.0/(it-99)**0.55*(accept-0.3)


        if it >= self.start_samples:
            x0 = np.log(self.lambda_)
            x1 = np.log(self.mu)
            x2 = np.log(self.rho)- np.log(1-self.rho)
            x3 = np.log(self.rho_beta) - np.log(1-self.rho_beta)

            self.sum_1[0] = self.sum_1[0] + x0
            self.sum_1[1] = self.sum_1[1] + x1
            self.sum_1[2] = self.sum_1[2] + x2
            self.sum_1[3] = self.sum_1[3] + x3

            self.sum_2[0] = self.sum_2[0] + x0**2
            self.sum_2[1] = self.sum_2[1] + x0*x1
            self.sum_2[2] = self.sum_2[2] + x1**2
            self.sum_2[3] = self.sum_2[3] + x0*x2
            self.sum_2[4] = self.sum_2[4] + x1*x2
            self.sum_2[5] = self.sum_2[5] + x2**2
            self.sum_2[6] = self.sum_2[6] + x0*x3
            self.sum_2[7] = self.sum_2[7] + x1*x3
            self.sum_2[8] = self.sum_2[8] + x2*x3
            self.sum_2[9] = self.sum_2[9] + x3**2


        self.psi[:,z_star==1] = copy.deepcopy(psi_star)
        self.kappa[:,z_star==1] = copy.deepcopy(kappa_star)

        new_beta = np.zeros((self.T,sum(z_star)))

        self.check_star =1
        chol_star,check = self.Chol(self.var_kf[:,:,self.T-1])
        if check ==0:
            new_beta[self.T-1] = self.mean_kf[:,self.T-1]+ np.dot(chol_star,np.random.normal(size=len(chol_star[0])))
        else:
            self.check_star = 0

        for ii in range(self.T-2,-1,-1):
            Gkal = np.diag(self.rho_beta[z_star ==1]*np.sqrt(1.0*psi_star[ii+1]/psi_star[ii]))
            invQ = np.diag(1.0/(1-self.rho_beta[z_star]**2)*1.0/psi_star[ii+1])
            inv_var_kf = np.linalg.inv(self.var_kf[:,:,ii])
            var_fb = np.linalg.inv(inv_var_kf+np.asarray(np.asmatrix(Gkal).T*np.asmatrix(invQ)*np.asmatrix(Gkal)))
            mean_fb = np.dot(var_fb,np.dot(inv_var_kf,self.mean_kf[:,ii])+np.dot(np.asarray(np.asmatrix(Gkal).T*np.asmatrix(invQ)),new_beta[ii+1]))
            chol_star, check = self.Chol(var_fb)
            if check ==0:
                new_beta[ii] = mean_fb + np.dot(chol_star,np.random.normal(size=len(chol_star[0])))
            else:
                self.check_star = 0

        if self.check_star ==1 and sum(sum(np.isnan(new_beta)))==0:
            self.beta[:,z_star==1] = copy.deepcopy(new_beta)

    def Chol(self,matrix):
        check = 0
        chol_star = []
        if matrix.shape[0] ==0:
            chol_star = []
            check =1
        else:
            try:
                chol_star = np.linalg.cholesky(matrix)
                check = 0
            except:
                check =1
        return chol_star,check

    def UpdateMuStar(self,it):
        new_mu_star = self.mu_star*np.exp(self.mu_gamma_sd*np.random.normal())
        log_accept = (self.p-1)*self.lambda_star*(np.log(self.mu_star)-np.log(new_mu_star))
        log_accept = log_accept-self.lambda_star*(1.0/new_mu_star-1/self.mu_star)*sum(self.mu[1:])
        log_accept = log_accept +np.log(new_mu_star)-np.log(self.mu_star)-3*np.log(new_mu_star+self.mu_mean)+3*np.log(self.mu_star+self.mu_mean)

        accept = 1
        if np.isnan(log_accept) or np.isinf(log_accept):
            accept =0
        elif log_accept <0:
            accept = np.exp(log_accept)
        self.mu_gamma_accept = self.mu_gamma_accept + accept
        self.mu_gamma_count = self.mu_gamma_count +1

        if np.random.random() < accept:
            self.mu_star = new_mu_star
        new_mu_gammma_sd = self.mu_gamma_sd +1.0/it**0.5*(accept-0.3)
        if new_mu_gammma_sd > 10**(-3) and  new_mu_gammma_sd <10**3:
            self.mu_gamma_sd = new_mu_gammma_sd

    def UpdateLambdaStar(self,it):
        new_lambda_star = self.lambda_star*np.exp(self.v_gamma_sd*np.random.normal())
        log_accept = (self.p-1)*(new_lambda_star*np.log(new_lambda_star/self.mu_star)-self.lambda_star*np.log(self.lambda_star/self.mu_star))
        log_accept = log_accept -(self.p-1)*(np.log(spec.gamma(new_lambda_star))-np.log(spec.gamma(self.lambda_star)))
        log_accept = log_accept +(new_lambda_star-self.lambda_star)*sum(np.log(self.mu[1:]))-(new_lambda_star-self.lambda_star)/self.mu_star*sum(self.mu[1:])
        log_accept = log_accept +np.log(new_lambda_star)-np.log(self.lambda_star)-1.0/self.mu_lambda_star*(new_lambda_star-self.lambda_star)
        accept =1
        if np.isnan(log_accept) or np.isinf(log_accept):
            accept =0
        elif log_accept <0:
            accept = np.exp(log_accept)

        self.v_gamma_accept = self.v_gamma_accept +accept
        self.v_gamma_count = self.v_gamma_count +1
        if np.random.random() < accept:
            self.lambda_star = new_lambda_star
        new_gamma_sd = self.v_gamma_sd +1.0/it**0.5*(accept-0.3)

        if new_gamma_sd >10**(-0.3) and new_gamma_sd <10**3:
            self.v_gamma_sd = new_gamma_sd

    def UpdateOutput(self,it,burnin,every):
        if it > burnin and (it-burnin)%every ==0:
            self.hold_beta.append(self.beta)
            self.hold_psi.append(self.psi)
            self.hold_sigma_sq.append(self.sigma_sq)
            self.hold_lambda.append(self.lambda_)
            self.hold_mu.append(self.mu)
            self.hold_rho.append(self.rho)
            self.hold_rho_beta.append(self.rho_beta)
            self.hold_lambda_sigma.appen(self.lambda_sigma)
            self.hold_mu_sigma.append(self.mu_sigma)
            self.hold_rho_sigma.append(self.rho_sigma)
            self.hold_mu_star.append(self.mu_star)

            print self.beta

    def LoadData(self,x_path,y_path):
        x_origin_list = sio.loadmat(x_path)['data']
        y_origin_list = sio.loadmat(y_path)['target']
        y_list = np.zeros(len(y_origin_list))

        for ii in range(len(y_origin_list)):
            y_list[ii] = y_origin_list[ii][0]

        y_list= (y_list-np.mean(y_list))/np.std(y_list,ddof=1)

        for ii in range(len(x_origin_list[0])):
            x_origin_list[:,ii] = copy.deepcopy((x_origin_list[:,ii]-np.mean(x_origin_list[:,ii]))/np.std(x_origin_list[:,ii],ddof =1))
        x_list = np.concatenate((np.array([np.ones(len(x_origin_list))]).T,x_origin_list),axis=1)

        return x_list,y_list


if __name__ == "__main__":

    x_path = "data/GDP_data.mat"
    y_path = "data/GDP_target.mat"
    
    import profile

    profile.run("case = NGAR(x_path,y_path,0.1,0.1,2000,5000,5)", "prof.txt")
    import pstats
    p = pstats.Stats("prof.txt")
    p.sort_stats("time").print_stats()

    #gdp_case = NGAR(x_path,y_path,0.1,0.1,2000,5000,5)
