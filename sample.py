#!/usr/bin/python
# -*- coding: utf-8 -*-
# File Name: data.py
# Author: Changsheng Zhang
# mail: zhangcsxx@gmail.com

import numpy as np
import copy
from scipy import stats
import scipy.special as spec
import math
import random
import copy

# phi、tao、k，beta 是二维数组
# lambda、u、rou、varphi是一维数组
# iterNum 是迭代次数，eta,T,alpha_hat 为常数
def UpdatePhi(phi_old_list,tao_phi_old_list,lambda_old_list,k_old_list,beta_old_list,u_list,rou_list,varphi_list,
      iterNum,eta,alpha_phi_hat,T):
    print "begin a new iteration:"

    #二维数组
    phi_new_list = []
    tao_phi_new_list = []
    for ii in range(len(phi_old_list)):
        #因为之前把phi 当做一维数组来算，为了补写方便，故用了下面的赋值方式
        #注意要用深复制，否则可能造成原变量改变，这是python数据结构的特性
        phi_old =copy.deepcopy(phi_old_list[ii])
        tao_phi_old = copy.deepcopy(tao_phi_old_list[ii])
        lambda_old = copy.deepcopy(lambda_old_list[ii])
        k_old = copy.deepcopy(k_old_list[ii])
        beta_old = copy.deepcopy(beta_old_list[ii])
        u = copy.deepcopy(u_list[ii])
        rou = copy.deepcopy(rou_list[ii])
        varphi = copy.deepcopy(varphi_list[ii])

        #当前下标为 ii时 的临时数组数据
        phi_new = []
        tao_phi_new =[]
        # step 1
        xPhi = stats.norm(loc=np.log(phi_old[0]) ,scale=tao_phi_old[0] )

        #注意这里rvs函数返回的是一个数组！！！
        logPhi_i1_new =xPhi.rvs(size =1)
        phi_i1_new_temp = np.exp(logPhi_i1_new[0])

        # step 2
        temp_a = phi_old[0]**(lambda_old+k_old[0]-1.5)
        temp_b = np.exp(-1.0*phi_old[0]*lambda_old/(u*(1-rou)))
        temp_c =(beta_old[1]-varphi*((phi_old[1]/phi_old[0])**0.5)*beta_old[0])**2/(phi_old[1]*(1-varphi**2))
        p_Phi_i1 =temp_a*temp_b*np.exp(-0.5*((beta_old[0]**2)/phi_old[0]+temp_c))

        temp_a = phi_i1_new_temp**(lambda_old+k_old[0]-1.5)
        temp_b = np.exp(-1.0*phi_i1_new_temp*lambda_old/(u*(1-rou)))
        temp_c =(beta_old[1]-varphi*((phi_old[1]/phi_i1_new_temp)**0.5)*beta_old[0])**2/(phi_old[1]*(1-varphi**2))
        p_Phi_i1_new =temp_a*temp_b*np.exp(-0.5*((beta_old[0]**2)/phi_i1_new_temp+temp_c))

        ap_phi = min(1,1.0*p_Phi_i1_new/p_Phi_i1)

        # step 3

        yPhi = stats.binom(1,ap_phi)

        temp = yPhi.rvs(1)

        if temp[0]==1:
            phi_i1_new = phi_i1_new_temp
        else:
            phi_i1_new = phi_old[0]

        phi_new.append(phi_i1_new)

        # step 4
        log_tao_i1_new = np.log(tao_phi_old[0])+(iterNum)**(-1.0*eta)*(ap_phi-alpha_phi_hat)
        tao_phi_new.append(np.exp(log_tao_i1_new))

        t =1
        while t<T-1:
            # step 1
            xPhi = stats.norm(loc=np.log(phi_old[t]) ,scale=tao_phi_old[t] )

            logPhi_it_new =xPhi.rvs(size =1)
            phi_it_new_temp = np.exp(logPhi_it_new[0])

            # step 2
            temp_a = phi_old[t]**(lambda_old+k_old[t-1]+k_old[t]-1.5)

            temp_b = np.exp(-1.0*phi_old[t]*lambda_old/u*(1+(2*rou*lambda_old)/((1-rou)*u)))
            temp_c =(beta_old[t+1]-varphi*((phi_old[t+1]/phi_old[t])**0.5)*beta_old[t])**2/(phi_old[t+1]*(1-varphi**2))
            temp_d =(beta_old[t]-varphi*((phi_old[t]/phi_old[t-1])**0.5)*beta_old[t-1])**2/(phi_old[t]*(1-varphi**2))
            p_Phi_it =temp_a*temp_b*np.exp(-0.5*(temp_c+temp_d))

            temp_a = phi_it_new_temp**(lambda_old+k_old[t-1]+k_old[t]-1.5)
            temp_b = np.exp(-1.0*phi_it_new_temp*lambda_old/u*(1+(2*rou*lambda_old)/((1-rou)*u)))
            temp_c =(beta_old[t+1]-varphi*((phi_old[t+1]/phi_it_new_temp)**0.5)*beta_old[t])**2/(phi_old[t+1]*(1-varphi**2))
            temp_d =(beta_old[t]-varphi*((phi_it_new_temp/phi_old[t-1])**0.5)*beta_old[t-1])**2/(phi_it_new_temp*(1-varphi**2))
            p_Phi_it_new =temp_a*temp_b*np.exp(-0.5*(temp_c+temp_d))


            ap_phi = min(1,float(1.0*p_Phi_it_new/p_Phi_it))

            # step 3

            yPhi = stats.binom(1,ap_phi)
            temp  =yPhi.rvs(1)
            if temp[0]==0:
                phi_it_new = phi_it_new_temp
            else:
                phi_it_new = phi_old[t]

            phi_new.append(phi_it_new)

            # step 4
            log_tao_it_new = np.log(tao_phi_old[t])+(iterNum)**(-1.0*eta)*(ap_phi-alpha_phi_hat)
            tao_phi_new.append(np.exp(log_tao_it_new))

            t = t+1

        # t = T时,下标要-1
        # step 1
        t=T-1

        xPhi = stats.norm(loc=np.log(phi_old[t]) ,scale=tao_phi_old[t] )

        logPhi_it_new =xPhi.rvs(size =1)
        phi_it_new_temp = np.exp(logPhi_it_new[0])

        # step 2
        temp_a = phi_old[t]**(lambda_old+k_old[t-1]-1.5)
        temp_b = np.exp(-1.0*phi_old[t]*lambda_old/(u*(1-rou)))
        temp_c =(beta_old[t]-varphi*((phi_old[t]/phi_old[t-1])**0.5)*beta_old[t-1])**2/(phi_old[t]*(1-varphi**2))
        p_Phi_it =temp_a*temp_b*np.exp(-0.5*temp_c)

        temp_a = phi_it_new_temp**(lambda_old+k_old[t-1]-1.5)
        temp_b = np.exp(-1.0*phi_it_new_temp*lambda_old/(u*(1-rou)))
        temp_c =(beta_old[t]-varphi*((phi_it_new_temp/phi_old[t-1])**0.5)*beta_old[t-1])**2/(phi_it_new_temp*(1-varphi**2))
        p_Phi_it_new =temp_a*temp_b*np.exp(-0.5*temp_c)

        ap_phi = min(1,1.0*p_Phi_it_new/p_Phi_it)

        # step 3

        yPhi = stats.binom(1,ap_phi)
        temp  = yPhi.rvs(1)
        if temp[0]==0:
            phi_it_new = phi_it_new_temp
        else:
            phi_it_new = phi_old[t]

        phi_new.append(phi_it_new)

        # step 4
        log_tao_it_new = np.log(tao_phi_old[t])+(iterNum)**(-1.0*eta)*(ap_phi-alpha_phi_hat)
        tao_phi_new.append(np.exp(log_tao_it_new))

        phi_new_list.append(phi_new)
        tao_phi_new_list.append(tao_phi_new)

    return  phi_new_list, tao_phi_new_list



# k,z,phi,是二维数组
# lambda, rou, u, 是一维数组
# T,iterNum,eta ,alpha是常数
def UpdateK(k_old_list,z_old_list,T,lambda_old_list,rou_list,u_list,phi_list,alpha_hat,iterNum,eta):

    k_new_list = []
    z_new_list = []

    for ii in range(len(k_old_list)):
        #同phi，降维
        k_old = copy.deepcopy(k_old_list[ii])
        z_old = copy.deepcopy(z_old_list[ii])
        lambda_old = copy.deepcopy(lambda_old_list[ii])
        rou = copy.deepcopy(rou_list[ii])
        u = copy.deepcopy(u_list[ii])
        phi = copy.deepcopy(phi_list[ii])

        k_new =  []
        z_new =  []
        for t in range(T-1):
            dK = stats.binom(1,0.5)
            temp = dK.rvs(1)
            d_k = 0
            if temp[0] ==0:
                d_k = 1
            else:
                d_k = -1

            epsilon_K = stats.geom(1.0/(1+z_old[t]))
            epsilon = epsilon_K.rvs(1)

            k_new_temp = k_old[t]+d_k*epsilon[0]


            # step 3 and step 4
            if k_new_temp < 0:
                k_new.append(k_old[t])

                z_new.append(z_old[t])

            else:
                p_k = (lambda_old/((1-rou)*u))**k_old[t]*(phi[t]*lambda_old*rou/((1-rou)*u))**k_old[t]*\
                  phi[t+1]/(math.factorial(k_old[t])*spec.gamma(lambda_old+k_old[t]))

                p_k_new = (lambda_old/((1-rou)*u))**k_new_temp*(phi[t]*lambda_old*rou/((1-rou)*u))**k_new_temp*\
                  phi[t+1]/(math.factorial(k_new_temp)*spec.gamma(lambda_old+k_new_temp))

                ap = min(1,p_k_new/p_k)

                y_AP = stats.binom(1,ap)
                temp = y_AP.rvs(1)

                if temp[0] ==1:
                    k_new.append(k_new_temp)
                else:
                    k_new.append(k_old[t])

                temp_z = z_old[t]+ iterNum**(-1.0*eta)*(ap-alpha_hat)
                z_new.append(temp_z)

            k_new_list.append(k_new)
            z_new_list.append(z_new)

    return  k_new_list, z_new_list


# x,beta 是二维数组，
# y,k_sigma 是一维数组
# T，rou_sigma,u_sigma,lambda_sigma 是常数
def UpdateSigma(x,y,T,beta,rou_sigma,u_sigma,lambda_sigma,k_sigma):

    # 返回一个list，是sigma
    sigma_new =[]
    sigma_new_2 =[]

    for t in range(T):
        tempSum = 0
        for i in range(len(x)):
            tempSum =tempSum+ beta[i][t]*x[i][t]
        d =(y[t]-tempSum)**2

        c=2*(lambda_sigma+rou_sigma*lambda_sigma)/(u_sigma*(1-rou_sigma))
        if t ==0 or t== T-1 :
            c= 2.0*lambda_sigma/(u_sigma*(1-rou_sigma))

        h = 0
        if t==0:
            h = k_sigma[t]+lambda_sigma-0.5
        elif t==T-1:
            h = k_sigma[t-1]+lambda_sigma-0.5
        else:
            h = k_sigma[t] +k_sigma[t-1]+lambda_sigma-0.5

        #cont 的上确界在几百左右，故取1000
        cont = 1000
        flag = 1
        while flag:
            u_random = random.uniform(0,1)
            v_random_temp = stats.gamma.rvs(1,size =1)


            v_random =v_random_temp[0]

            fY = (1.0*c/d)**(1.0*h/2)/(2*spec.kv(h,(c*d)**0.5))*v_random**(h-1)*np.exp(-0.5*(c*v_random+1.0*d/v_random))
            gY = np.exp(-1.0*v_random)

            if u_random <= fY/(cont*gY):
                #要开方处理
                sigma_new.append(v_random**0.5)
                sigma_new_2.append(v_random)
                flag =0
            else:
                flag =1
    #返回sigma 和sigma^2
    return sigma_new,sigma_new_2

# sigma_2 表示方程 sigma^2
# k_sigma,z_sigma,sigma_2 是一维数组
# lambda_sigma,rou_sigma,u_sigma,iterNum,eta,alpha,T 是常数

def UpdateK_sigma(k_sigma_old,z_sigma_old,lambda_sigma,rou_sigma,u_sigma,sigma_2,iterNum,eta,alpha,T):

    k_sigma_new = []
    z_sigma_new = []

    for t in range(T-1):

        dK = stats.binom(1,0.5)
        temp = dK.rvs(1)
        d_k = 0
        if temp[0] ==1:
            d_k = 1
        else:
            d_k =-1

        epsilon_K = stats.geom(1.0/(1+z_sigma_old[t]))
        epsilon = epsilon_K.rvs(1)

        k_sigma_new_temp = k_sigma_old[t]+d_k*epsilon[0]

        # step 3
        if k_sigma_new_temp < 0:
            k_sigma_new.append(k_sigma_old[t])
            z_sigma_new.append(z_sigma_old[t])
        else:
            # step 2
            p_k_sigma = (lambda_sigma/((1-rou_sigma)*u_sigma))**k_sigma_old[t]*\
            (sigma_2[t]*lambda_sigma*rou_sigma/((1-rou_sigma)*u_sigma))**k_sigma_old[t]\
            *(sigma_2[t+1])**k_sigma_old[t]/(math.factorial(k_sigma_old[t])*spec.gamma(lambda_sigma+k_sigma_old[t]))

            p_k_sigma_new = (lambda_sigma/((1-rou_sigma)*u_sigma))**k_sigma_new_temp*\
            (sigma_2[t]*lambda_sigma*rou_sigma/((1-rou_sigma)*u_sigma))**k_sigma_new_temp\
            *(sigma_2[t+1])**k_sigma_new_temp/(math.factorial(k_sigma_new_temp)*spec.gamma(lambda_sigma+k_sigma_new_temp))


            ap = min(1,p_k_sigma_new/p_k_sigma)

            y_AP = stats.binom(1,ap)
            temp = y_AP.rvs(1)
            if temp[0] ==1:
                k_sigma_new.append(k_sigma_new_temp)
            else:
                k_sigma_new.append(k_sigma_old[t])

            # step 4
            temp_z = z_sigma_old[t]+ iterNum**(-1.0*eta)*(ap-alpha)
            z_sigma_new.append(temp_z)

    return  k_sigma_new,z_sigma_new


# xi_old 是三维数组
# beta，xi_sigma_old。phi_old，k_old 是二维数组
# sigma_old，k_sigma_old，lambda_old，u_old，rou_old,varphi_old,s_xi_old是一维数组
# alpha_hat,alpha_hat_sigma，iterNum,eta,eta_sigma，lambda_sigma_old,u_sigma_old, rou_sigma_old,s_xi_sigma_old 是常数
def UpdateTheta(alpha_hat,alpha_hat_sigma,iterNum,eta,eta_sigma,sigma_old,k_sigma_old,
    beta,lambda_old,u_old,rou_old,varphi_old,xi_old,s_xi_old_list,lambda_sigma_old,u_sigma_old,
    rou_sigma_old,xi_sigma_old,s_xi_sigma_old,phi_old_list,k_old_list):
    # step 1
    s_i=[]
    beta_M =[]
    beta_C =[]


    phi_new_list = []
    k_new_list =[]
    sigma_new = []
    k_sigma_new =[]

    #xi_new_list is 2-dim
    xi_new_list = []
    xi_sigma_new = []
    s_xi_new= []
    s_xi_sigma_new = 0

    for i in range(len(beta)):
        # binom() 第二个参数是第一个参数出现的概率
        s_bi = stats.binom(1,5.0/len(beta))
        temp = s_bi.rvs(1)
        s_i.append(temp[0])

        if temp[0] ==1:
            beta_M.append(beta[i])
        else:
            beta_C.append(beta[i])

    # step 2
    for ii in range(len(beta)):

        # ！！！这里多元正态分布有问题
        lambda_i_old = copy.deepcopy(lambda_old[ii])
        u_i_old = copy.deepcopy(u_old[ii])
        rou_i_old = copy.deepcopy(rou_old[ii])
        varphi_i_old = copy.deepcopy(varphi_old[ii])
        phi_old = copy.deepcopy(phi_old_list[ii])
        k_i_old = copy.deepcopy(k_old_list[ii])
        s_xi_old = copy.deepcopy(s_xi_old_list[ii])

        # xi_i 和 xi_sigma 两者要分开讨论，一个是二维数组，一个是一维数组
        xi_i =[np.log(lambda_i_old),np.log(u_i_old),np.log(rou_i_old)-np.log(1-rou_i_old),np.log(varphi_i_old)-np.log(1-varphi_i_old)]

        # xi_old 的列数为4，而np.cov()的接口是，行数为指标的个数，即位4，故需要转置
        #S_xi的type 是array，4*4
        #xi_old 是一个三维数组，第一个是 迭代次数，第二个是 ii，第三个是4，要取出第二个参数
        S_xi = np.cov(np.array(zip(*xi_old)[ii]).T)

        #xi_X = stats.norm(loc = xi_i,scale = np.dot(s_xi_old,S_xi))
        xi_X = stats.multivariate_normal(mean=xi_i,cov=np.dot(s_xi_old,S_xi))
        xi_i_1 = xi_X.rvs(1)


        # step 3

        lambda_i_new  = np.exp(xi_i_1[0])
        u_i_new = np.exp(xi_i_1[1])

        rou_divide_old = rou_i_old/(1-rou_i_old)
        rou_divide_new = np.exp(float(xi_i_1[2]))

        phi_new = []
        k_i_new = []

        if lambda_i_new > lambda_i_old:
            temp = stats.gamma.rvs(lambda_i_new-lambda_i_old,lambda_i_new/u_i_new,size=1)
            phi_new.append(lambda_i_old*u_i_new/(lambda_i_new*u_i_old)*phi_old[0]+temp[0])
        else:
            temp = stats.bernoulli.rvs(lambda_i_new ,loc =lambda_i_old-lambda_i_new,size=1)
            phi_new.append(lambda_i_old*u_i_new/(lambda_i_new*u_i_old)*phi_old[0]*temp[0])



        for t in range(min(len(phi_old),len(k_i_old))):

            if rou_divide_new*phi_new[t]*lambda_i_new/u_i_new > rou_divide_old*phi_old[t]*lambda_i_old/u_i_old:
                k_X= stats.poisson(rou_divide_new*phi_new[t]*lambda_i_new/u_i_new -rou_divide_old*phi_old[t]*lambda_i_old/u_i_old)
                temp_sample = k_X.rvs(1)
                k_i_new.append(k_i_old[t]+temp_sample[0])
            else:
                k_X = stats.binom.rvs(k_i_old[t],rou_divide_new*phi_new[t]*lambda_i_new/u_i_new/(rou_divide_old*phi_old[t]*lambda_i_old/u_i_old),size =1)
                k_i_new.append(k_X[0])

            if t<len(phi_old)-1:
                if lambda_i_new+k_i_new[t-1] > lambda_i_old + k_i_old[t-1]:
                    temp_sample = stats.gamma.rvs(lambda_i_new+k_i_new[t-1]-lambda_i_old-k_i_old[t-1],lambda_i_new/u_i_new*(1+rou_divide_new),size=1)

                    phi_new.append(lambda_i_new*u_i_new*(1+rou_divide_old)/(lambda_i_old*u_i_old*(1+rou_divide_new))+temp_sample[0])
                else:

                    print "probability :",lambda_i_new+k_i_new[t-1]

                    temp = stats.bernoulli.rvs(lambda_i_new+k_i_new[t-1],
                           loc =lambda_i_old+k_i_old[t-1]-lambda_i_new-k_i_new[t-1],size=1)
                    phi_new.append(lambda_i_new*u_i_new*(1+rou_divide_old)/(lambda_i_old*u_i_old*(1+rou_divide_new))*temp[0])

        phi_new_list.append(phi_new)
        k_new_list.append(k_i_new)

        # step 4.1
        # step 5.1
        ap = 0.5

        #step 6.1
        xi_new =[]
        xi_ap = stats.binom(1,ap)
        temp = xi_ap.rvs(1)
        # update
        if temp[0]==1:
            for jj in range(len(xi_i_1)):
                xi_new.append(xi_i_1[jj])

            phi_old_list[ii]=copy.deepcopy(phi_new)


            k_old_list[ii] = copy.deepcopy(k_i_new)

        else:
            for jj in range(len(xi_i)):
                xi_new.append(xi_i[jj])


        xi_new_list.append(xi_new)

        # step 2.7.1
        temp = np.exp(np.log(s_xi_old)+iterNum**(-1.0*eta)*(ap-alpha_hat))
        s_xi_new.append(temp)


        # step2.2

        if ii == len(beta)-1:

            xi_sigma =[np.log(lambda_sigma_old),np.log(u_sigma_old),np.log(rou_sigma_old)-np.log(1-rou_sigma_old)]
            S_xi_sigma = np.cov(np.array(xi_sigma_old).T)
            #xi_sigma_1 = stats.norm(loc = xi_sigma,scale =np.dot(s_xi_sigma_old,S_xi_sigma)).rvs(1)
            xi_sigma_1 = stats.multivariate_normal(mean=xi_sigma,cov=np.dot(s_xi_sigma_old,S_xi_sigma)).rvs(1)


            lambda_sigma_new = np.exp(xi_sigma_1[0])
            u_sigma_new = np.exp(xi_sigma_1[1])
            rou_sigma_divide_new = np.exp(xi_sigma_1[2])
            rou_sigma_divide_old = rou_sigma_old/(1-rou_sigma_old)

            if lambda_sigma_new >lambda_sigma_old:
                temp = stats.gamma.rvs(lambda_sigma_new-lambda_sigma_old,lambda_sigma_new/u_sigma_new,size=1)
                sigma_new.append(lambda_sigma_old*u_sigma_new/(lambda_sigma_new*u_sigma_old)*sigma_old[0]+temp[0])

            else:
                temp = stats.bernoulli.rvs(lambda_sigma_new ,loc =lambda_sigma_old-lambda_sigma_new ,size =1)
                sigma_new.append(lambda_sigma_old*u_sigma_new/(lambda_sigma_new*u_sigma_old)*sigma_old[0]*temp[0])

            for t in range(min(len(sigma_old),len(k_sigma_old))):
                if rou_sigma_divide_new*phi_new[t]*lambda_sigma_new/u_sigma_new >rou_sigma_divide_old*phi_old[t]*lambda_sigma_old/u_sigma_old:
                    temp = stats.poisson.rvs(rou_sigma_divide_new*phi_new[t]*lambda_sigma_new/u_sigma_new-rou_sigma_divide_old*phi_old[t]*lambda_sigma_old/u_sigma_old,size=1)
                    k_sigma_new.append(k_sigma_old[t]+temp[0])
                else:
                    temp = stats.binom.rvs(k_sigma_old[t],rou_sigma_divide_new*phi_new[t]*lambda_sigma_new/u_sigma_new/(rou_sigma_divide_old*phi_old[t]*lambda_sigma_old/u_sigma_old),size=1)
                    k_sigma_new.append(temp[0])

                if t<len(sigma_old)-1:
                    if lambda_sigma_new+k_sigma_new[t-1]>lambda_sigma_old+k_sigma_old[t-1]:
                        temp = stats.gamma.rvs(lambda_sigma_new-lambda_sigma_old+k_sigma_new[t-1]-k_sigma_old[t-1],lambda_sigma_new/u_sigma_new*(1+rou_sigma_divide_new),size=1)
                        sigma_new.append(temp[0]+lambda_sigma_new*u_sigma_new*(1+rou_sigma_divide_old)/(lambda_sigma_old*u_sigma_old*(1+rou_sigma_divide_new)))
                    else:
                        temp = stats.bernoulli.rvs(lambda_sigma_new+k_sigma_new[t-1],
                               loc=lambda_sigma_old+k_sigma_old[t-1]-lambda_sigma_new-k_sigma_new[t-1],
                               size=1)
                        sigma_new.append(temp[0]*lambda_sigma_new*u_sigma_new*(1+rou_sigma_divide_old)/(lambda_sigma_old*u_sigma_old*(1+rou_sigma_divide_new)))

            # step 4

            # step 5

            ap_sigma = 0.5


            # step 6

            sigma_ap = stats.binom(1,ap_sigma)
            temp =sigma_ap.rvs(1)
            if temp[0] ==1:
                for jj in range(len(xi_sigma_1)):
                    xi_sigma_new.append(xi_sigma_1[jj])

                for jj in range(len(sigma_new)):
                    sigma_old[jj]=sigma_new[jj]

                for jj in range(len(k_sigma_new)):
                    k_sigma_old[jj]=k_sigma_new[jj]

            else:
                for jj in range(len(xi_sigma)):
                    xi_sigma_new.append(xi_sigma[jj])


            xi_sigma_old.append(np.array(xi_sigma_new))

            # step 7

            s_xi_sigma_new = np.exp(np.log(s_xi_sigma_old)+iterNum**(-1.0*eta_sigma)*(ap_sigma-alpha_hat_sigma))


    xi_old.append(np.array(xi_new_list))


    return xi_new_list,xi_sigma_new,s_xi_new,s_xi_sigma_new


# phi,beta, 是二维数组
# varphi 是一维数组
# T 是常数
def UpdateBeta(phi_list,varphi_list,T):


    beta_list = []
    for ii in range(len(phi_list)):

        phi = copy.deepcopy(phi_list[ii])
        varphi = copy.deepcopy(varphi_list[ii])
        beta =[]

        beta_X = stats.norm(loc=0,scale =phi[0])
        beta_1 = beta_X.rvs(1)
        beta.append(beta_1[0])
        for t in range(1,T):
            eta_X = stats.norm(loc =0,scale =(1-varphi**2)*phi[t])
            eta_t = eta_X.rvs(1)
            beta_t =(phi[t]/phi[t-1])**0.5*varphi*beta[t-1] + eta_t[0]
            beta.append(beta_t)
        beta_list.append(beta)
    return  beta_list

#u,是一维数组
# u_star,tao,b_star,lambda_star,m,iterNum,eta,alpha 是常数
def UpdateU_star(u_star_old,tao_u_star,b_star,lambda_star,m,iterNum,eta,alpha):
    log_u_star_X = stats.norm(loc =np.log(u_star_old),scale =tao_u_star)
    log_u_star = log_u_star_X.rvs(1)
    u_star = np.exp(log_u_star[0])

    p_u_star = (u_star_old+2*b_star)**(-3)*(1.0/u_star_old)**(m*lambda_star)*np.exp(-1.0*lambda_star/u_star_old*sum(u))
    p_u_star_new = (u_star+2*b_star)**(-3)*(1.0/u_star)**(m*lambda_star)*np.exp(-1.0*lambda_star/u_star*sum(u))

    ap = min(1,p_u_star_new/p_u_star)

    # step 3

    y_AP = stats.binom(1,ap)
    u_star_new = u_star_old
    temp = y_AP.rvs(1)
    if temp[0] ==1:
        u_star_new = u_star

    # step 4

    log_tao_u_star = np.log(tao_u_star)+iterNum**(-1.0*eta)*(ap-alpha)
    tao_u_star_new = np.exp(log_tao_u_star)

    return u_star_new,tao_u_star_new

# u 是一维数组
# lambda_star,tao,s_star,u_star,m,eta,alpha,iterNum 是常数

def UpdateLambda(lambda_star_old,tao_lambda_star_old,s_star,u_star,m,u,eta,alpha,iterNum):
    log_lambda_star_X = stats.norm(loc =np.log(lambda_star_old),scale =tao_lambda_star_old )
    log_lambda_star = log_lambda_star_X.rvs(1)
    lambda_star = np.exp(log_lambda_star[0])

    temp =1.0
    for item in u:
        temp = temp*item**lambda_star_old


    p_lambda_star = np.exp(-1.0*lambda_star_old/s_star)*\
        (((lambda_star_old**lambda_star_old)/(u_star**lambda_star_old*spec.gamma(lambda_star_old)))**m)*\
        np.exp(-1.0*lambda_star_old/u_star*sum(u))*temp

    temp =1.0
    for item in u:
        temp = temp*item**lambda_star


    p_lambda_star_new = np.exp(-1.0*lambda_star/s_star)*\
        (((lambda_star**lambda_star)/(u_star**lambda_star*spec.gamma(lambda_star)))**m)*\
        np.exp(-1.0*lambda_star/u_star*sum(u))*temp

    ap = min(1,p_lambda_star_new/p_lambda_star)

    # step 3

    y_AP = stats.binom(1,ap)
    lambda_star_new = lambda_star_old
    temp = y_AP.rvs(1)
    if temp[0] ==1:
        lambda_star_new = lambda_star

    # step 4

    log_tao_lambda_star = np.log(tao_lambda_star_old)+iterNum**(-1.0*eta)*(ap-alpha)
    tao_lambda_star_new = np.exp(log_tao_lambda_star)

    print "finished a iteration."
    return lambda_star_new,tao_lambda_star_new


if __name__ =='__main__':
    #初始值
    T = 200
    #phi_old tao_phi_old 是二维数组,维度和x一样
    m = 5
    #x 是6＊T的矩阵，第一行全为1
    temp_x_list= []
    for ii in range(T):
        mul_var = stats.multivariate_normal(mean=[0,0,0,0,0],cov=np.identity(5))
        temp_x = [1]
        mul_var_rvs = mul_var.rvs(1)
        for jj in range(len(mul_var_rvs)):
            temp_x.append(mul_var_rvs[jj])
        temp_x_list.append(temp_x)
    x = map(list,zip(*temp_x_list))
    #beta
    beta_0 = [0]
    beta_1 = [0]
    beta_2 = [stats.norm.rvs(loc=2,scale=0.5,size=1)[0]]
    beta_3 = [0]
    beta_4 = [0]
    beta_5 = [0]
    for ii in range(T-1):
        beta_0.append(0)
        beta_1.append(0.97*beta_1[-1]+stats.norm.rvs(loc=2,scale=0.5,size=1)[0])
        if ii <99:
            beta_2.append(0.97*beta_2[-1]+stats.norm.rvs(loc=0,scale=0.5,size=1)[0])
        else:
            beta_2.append(0)
        if (t>=20 and t<=49)or (t>=120 and t<=149):
            beta_3.append(-2)
        else:
            beta_3.append(0)
        beta_4.append(0)
        beta_5.append(0)
    beta= [beta_0,beta_1,beta_2,beta_3,beta_4,beta_5]

    lambda_old = 2
    alpha_hat = 0.3
    sigma = 1
    #y
    y= []
    for ii in range(T):
        y_ii = stats.norm.rvs(scale =sigma ,size=1)[0]
        for jj in range(m+1):
            y_ii = y_ii + x[jj][ii]*beta[jj][ii]
        y.append(y_ii)
    # u,rou,lambda,varphi, lambda_star,u_star
    u = []
    s_star = 0.1
    b_star = 0.1
    u_star = []
    lambda_star = []
    lambda_old = []
    varphi_old = []
    rou_old = []
    for ii in range(m+1):
        temp_lambda = stats.expon.rvs(scale = s_star,size =1 )[0]
        lambda_star.append(temp_lambda)
        flag = 1
        cont = 1000
        while flag:
            u_ranom =random.uniform(0,1)
            v_random = stats.gamma.rvs(1,size=1)[0]
            fy = (v_random+2*b_star)**(-3)
            gy = np.exp(-1.0*v_random)
            if u_ranom <=fy/(cont*gy):
                u_star.append(v_random)
                flag = 0
                break
            else:
                flag =1 
    for ii in range(m+1):
        temp_u = stats.gamma.rvs(lambda_star[ii],lambda_star[ii]/u_star[ii])[0]
        u.append(temp_u)
    for ii in range(m+1):
        cont =1000
        while flag:
            u_ranom = random.uniform(0,1)
            v_random = stats.gamma.rvs(1,size=1)[0]
            fy = v_random*(0.5+v_random)**(-4)
            gy = np.exp(-1.0*v_random)
            if u_ranom <=fy/(cont*gy):
                lambda_old.append(v_random)
                flag = 0
                break
            else:
                flag =1
    #这个分布有问题，目前还没写
    for ii in range(m+1):
        varphi_old.append(stats.)
        rou_old.append(stats.)
    lambda_sigma = stats.gamma.rvs(3,1)[0]
    k_sigma = random.uniform(0,1)
    u_sigma = 0
    cont =1000
    flag = 1
    while flag:
        u_random = random.uniform(0,1)
        v_random = stats.gamma.rvs(1,size=1)[0]
        fy = (1+v_random)**(-1.5)
        gy = np.exp(-1.0*v_random)
        if u_random <= fy/(cont*gy):
            u_sigma = v_random
            break
        else:
            flag =1
    rou_sigma = stats.


    #生成phi,tao,k,z
    tao_phi_old = 0.3
    phi_old_temp = []
    k_old_temp = []
    k_per_row = []
    z_old = 0.3 
    
    for ii in range(m+1):
        phi_per_row.append(stats.gamma.rvs(lambda_old[ii],lambda_old[ii]/u[ii]))
    phi_old_temp.append(phi_per_row)

    for jj in range(m+1):
        k_per_row.append(stats.gamma.rvs(rou_old[jj]*lambda_old*phi_old_temp[0][jj]/(u_old[jj]*(1-rou_old[jj])))[0])
    k_old_temp.append(k_per_row)
    
    for ii in range(1,T):
        phi_per_row = []
        k_per_row = []
        for jj in range(1+m):
            phi_per_row.append(stats.gamma.rvs(lambda_old+k_old_temp[ii-1][jj],lambda_old/(u_old[jj]*(1-rou_old[jj])))[0])
            k_per_row.append(stats.gamma.rvs(rou_old[jj]*lambda_old*phi_per_row[jj]/(u_old[jj]*(1-rou_old[jj])))[0])
        phi_old_temp.append(phi_per_row)
        k_old_temp.append(k_per_row)
    k_old = map(list,zip(*k_old_temp))
    phi_old = map(list,zip(*phi_old_temp))

    eta = random.uniform(0.5,1)
    eta_sigma = random.uniform(0.5,1)
    z_sigma_old = random.uniform(0,1)
 
    for ii in range(1000):
        iterNum = ii +1 
        phi_new,tao_phi_new = UpdatePhi(phi_old,tao_phi_old,lambda_old,k_old,beta,u,rou_old,varphi_old,iterNum,eta,alpha_hat,T)
        
        k_new,z_new = UpdateK(k_old,z_old,T,lambda_old,rou_old,u,phi_new,alpha_hat,iterNum,eta)

        temp_sigma,sigma_square = UpdateSigma(x,y,T,beta,rou_sigma.u_sigma,lambda_sigma,k_sigma)




    phi_old = [[1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1],
           [1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1]]
    lambda_old = [1,1,1,1,1,1,1,1,1,1]
    k_old = [[2,1,4,1,2,1,4,1,1,1],[2,1,4,1,2,1,4,1,1,1],[2,1,4,1,2,1,4,1,1,1],[2,1,4,1,2,1,4,1,1,1],
         [2,1,4,1,2,1,4,1,1,1],[2,1,4,1,2,1,4,1,1,1]]
    z_old = [[1,0.5,1,1,1,2,1,1.3,1,1],[1,0.5,1,1,1,2,1,1.3,1,1],[1,0.5,1,1,1,2,1,1.3,1,1],[1,0.5,1,1,1,2,1,1.3,1,1],
         [1,0.5,1,1,1,2,1,1.3,1,1],[1,0.5,1,1,1,2,1,1.3,1,1]]

    alpha_phi_old = [[0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97],[0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97],
         [0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97],[0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97],
         [0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97],[0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97]]

    u = [0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98]
    rou = [0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99]
    varphi = [0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99]
    #iterNum =1
    eta = 0.75
    alpha_k_hat = [0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]

    #update sigma^2
    x = [[1,2,3,2,3,2,3,2,3,1],[2,3,4,2,3,2,3,2,3,1],[5,3,1,2,3,2,3,2,3,1],[1,1,3,2,3,2,3,2,3,1],
         [1,2,3,2,3,2,3,2,3,1],[2,3,4,2,3,2,3,2,3,1]]
    m = len(x)-1
    y = [1,2,3,3,2,1,1,2,3,2]
    beta_old = [[1,2,3,2,3,2,3,2,3,1],[2,3,4,2,3,2,3,2,3,1],[5,3,1,2,3,2,3,2,3,1],[1,1,3,2,3,2,3,2,3,1],
        [5,3,1,2,3,2,3,2,3,1],[1,1,3,2,3,2,3,2,3,1]]

    # rou_sigma != 1
    rou_sigma = 0.95
    u_sigma = 2.0
    lambda_sigma = 1.0
    k_sigma = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]



    #update k_sigma
    z_sigma_old = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    alpha = 0.95

    #2.5 update

    m =5

    alpha_hat = 0.97
    alpha_hat_sigma = 0.97
    eta_sigma = 0.75

    xi_old  = list(np.random.random((100,6,4)))
    xi_sigma_old = list(np.random.random((100,3)))
    s_xi_old = [1,1,1,1,1,1]
    s_xi_sigma_old = 1



    # update U_star
    u_star_old = 2
    tao_u_star = 0.8
    b_star= 1
    lambda_star = 2


    # update lambda
    # s_star,u_star 是2.5里的
    s_star =2
    tao_lambda_star = 0.95


    # run sample
    print "begin \n"

    for ii in range(1000):
        iterNum = ii+1
        phi_new, tao_phi_new = UpdatePhi(phi_old,tao_phi_old,lambda_old,k_old,beta_old,u,rou,varphi,iterNum,eta,alpha_phi_old,T)
        k_new, z_new = UpdateK(k_old,z_old,T,lambda_old,rou,u,phi_new,alpha_k_hat,iterNum,eta)

        temp_sigma,sigma_square = UpdateSigma(x,y,T,beta,rou_sigma,u_sigma,lambda_sigma,k_sigma)


        k_sigma_new,z_sigma_new= UpdateK_sigma(k_sigma,z_sigma_old,lambda_sigma,rou_sigma,u_sigma,sigma_square,
               iterNum,eta,alpha,T)


        #2.5
        #xi_new is 2-dim
        # xi_sigma_new is 1-dim s_xi_new is 1-dim, either
        # s_xi_sigma_new is const
        xi_new,xi_sigma_new,s_xi_new,s_xi_sigma_new = UpdateTheta(alpha_hat,alpha_hat_sigma,iterNum,eta,eta_sigma,
              temp_sigma,k_sigma_new,beta_old,
              lambda_old,u,rou,varphi,xi_old,s_xi_old,
              lambda_sigma,u_sigma,rou_sigma,xi_sigma_old,
              s_xi_sigma_old,phi_new,k_new)
        #还原 xi 到变量中
        # xi_new 是二维数组
        lambda_new = np.exp(zip(*xi_new)[0])
        u_new = np.exp(zip(*xi_new)[1])
        rou_new = np.exp(zip(*xi_new)[2])/(np.exp(zip(*xi_new)[2])+1)
        varphi_new = np.exp(zip(*xi_new)[3])/(np.exp(zip(*xi_new)[3])+1)

        lambda_sigma_new = np.exp(xi_sigma_new[0])
        u_sigma_new = np.exp(xi_sigma_new[1])
        rou_sigma_new =np.exp(xi_sigma_new[2])


        beta_new = UpdateBeta(phi_new,varphi_new,T)

        print "beta:"
        print beta_new
        print "\n"

        # u_i 不等于 u，u_i是2.5得到的
        u_star_new, tao_u_star_new = UpdateU_star(u_star_old,tao_u_star,b_star,lambda_star,m,iterNum,eta,alpha)

        lambda_star_new, tao_lambda_star_new = UpdateLambda(lambda_star,tao_lambda_star,s_star,u_star_new,m,
            u_new,eta,alpha,iterNum)

        #在这里更新变量
        #deepcopy

        phi_old = copy.deepcopy(phi_new)
        tao_phi_old = copy.deepcopy(tao_phi_new)
        lambda_old = copy.deepcopy(lambda_new)
        k_old = copy.deepcopy(k_new)
        beta_old = copy.deepcopy(beta_new)
        u = copy.deepcopy(u_new)
        rou = copy.deepcopy(rou_new)
        varphi =copy.deepcopy(varphi_new)
        z_old = copy.deepcopy(z_new)
        rou_sigma = copy.deepcopy(rou_sigma_new)
        u_sigma = copy.deepcopy(u_sigma_new)
        lambda_sigma=copy.deepcopy(lambda_sigma_new)
        k_sigma = copy.deepcopy(k_sigma_new)
        z_sigma_old = copy.deepcopy(z_sigma_new)

        u_star_old= copy.deepcopy(u_star_new)
        tao_u_star = copy.deepcopy(tao_u_star_new)
        lambda_star = copy.deepcopy(lambda_star_new)
        tao_lambda_star = copy.deepcopy(tao_lambda_star_new)

    #print lambda_star_new
    #print tao_lambda_star_new

    print "\nfinished!\n"


    #### the author's configuration
    # m = 5
    # T = 1000
    #
    # mul_var = stats.multivariate_normal(mean=[0,0,0,0,0],cov=np.identity(5))
    # temp_x = mul_var.rvs(T)
    # x = map(list,zip(*temp_x))
    # beta

