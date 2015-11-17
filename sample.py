#!/usr/bin/python
# -*- coding: utf-8 -*-
# File Name: sample.py
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
        
        phi_i1_new_temp = np.exp(stats.norm.rvs(loc=np.log(phi_old[0]),scale = np.sqrt(tao_phi_old[0]),size=1)[0])
        # step 2
        
        temp_a = phi_old[0]**(lambda_old+k_old[0]-1.5)
        temp_b = np.exp(-1.0*phi_old[0]*lambda_old/(u*(1-rou)))
        temp_c =(beta_old[1]-varphi*((phi_old[1]/phi_old[0])**0.5)*beta_old[0])**2/(phi_old[1]*(1-varphi**2))
        p_Phi_i1 =temp_a*temp_b*np.exp(-0.5*((beta_old[0]**2)/phi_old[0]+temp_c))
        print p_Phi_i1
        temp_a = phi_i1_new_temp**(lambda_old+k_old[0]-1.5)
        temp_b = np.exp(-1.0*phi_i1_new_temp*lambda_old/(u*(1-rou)))
        temp_c =(beta_old[1]-varphi*((phi_old[1]/phi_i1_new_temp)**0.5)*beta_old[0])**2/(phi_old[1]*(1-varphi**2))
        p_Phi_i1_new =temp_a*temp_b*np.exp(-0.5*((beta_old[0]**2)/phi_i1_new_temp+temp_c))
        print p_Phi_i1_new
        ap_phi = min(1,1.0*p_Phi_i1_new/p_Phi_i1)

        # step 3
    
        temp_binom = stats.binom.rvs(1,ap_phi,size=1)[0]
        if temp_binom ==1:
            phi_new.append(phi_i1_new_temp)
        else:
            phi_new.append(phi_old[0])
        
        # step 4
        log_tao_i1_new = np.log(tao_phi_old[0])+(iterNum)**(-1.0*eta)*(ap_phi-alpha_phi_hat)
        tao_phi_new.append(np.exp(log_tao_i1_new))

        t =1
        while t<T-1:
            # step 1
            
            phi_it_new_temp = np.exp(stats.norm.rvs(loc=np.log(phi_old[t]),scale = np.sqrt(tao_phi_old[t]),size=1)[0])
            # step 2
            temp_a = phi_old[t]**(lambda_old+k_old[t-1]+k_old[t]-1.5)
            #print temp_a
            temp_b = np.exp(-1.0*phi_old[t]*lambda_old/u*(1+(2*rou*lambda_old)/((1-rou)*u)))
            temp_c =(beta_old[t+1]-varphi*((phi_old[t+1]/phi_old[t])**0.5)*beta_old[t])**2/(phi_old[t+1]*(1-varphi**2))
            temp_d =(beta_old[t]-varphi*((phi_old[t]/phi_old[t-1])**0.5)*beta_old[t-1])**2/(phi_old[t]*(1-varphi**2))
            p_Phi_it =temp_a*temp_b*np.exp(-0.5*(temp_c+temp_d))
            #print p_Phi_it 
            temp_a = phi_it_new_temp**(lambda_old+k_old[t-1]+k_old[t]-1.5)
            temp_b = np.exp(-1.0*phi_it_new_temp*lambda_old/u*(1+(2*rou*lambda_old)/((1-rou)*u)))
            temp_c =(beta_old[t+1]-varphi*((phi_old[t+1]/phi_it_new_temp)**0.5)*beta_old[t])**2/(phi_old[t+1]*(1-varphi**2))
            temp_d =(beta_old[t]-varphi*((phi_it_new_temp/phi_old[t-1])**0.5)*beta_old[t-1])**2/(phi_it_new_temp*(1-varphi**2))
            p_Phi_it_new =temp_a*temp_b*np.exp(-0.5*(temp_c+temp_d))
            #print p_Phi_it_new

            ap_phi = min(1,float(1.0*p_Phi_it_new/p_Phi_it))

            # step 3
            
            temp_binom = stats.binom.rvs(1,ap_phi,size=1)[0]

            if temp_binom ==1:
                phi_new.append(phi_it_new_temp)
            else:
                phi_new.append(phi_old[t])

            # step 4
            log_tao_it_new = np.log(tao_phi_old[t])+(iterNum)**(-1.0*eta)*(ap_phi-alpha_phi_hat)
            tao_phi_new.append(np.exp(log_tao_it_new))
            
            t = t+1

        # t = T时,下标要-1
        # step 1
        t=T-1

        xPhi = stats.norm(loc=np.log(phi_old[t]) ,scale=np.sqrt(tao_phi_old[t]))

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

            epsilon = stats.geom.rvs(1.0/(1+z_old[t]),loc=-1,size=1)
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
        
        # 舍选法经过验证，误差较小
        #cont 的上确界在几百左右，故取2000
        cont = 2000
        flag = 1
        while flag:
            u_random = random.uniform(0,1)
            v_random = stats.gamma.rvs(1,size =1)[0]
            fY = (1.0*c/d)**(1.0*h/2)/(2*spec.kv(h,(c*d)**0.5))*v_random**(h-1)*np.exp(-0.5*(c*v_random+1.0*d/v_random))
            gY = np.exp(-1.0*v_random)
            
            #print "u_random:",u_random,fY,gY
            if u_random <= fY/(cont*gY):
                #要开方处理
                #print "sigma_t^2:",v_random
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
        
        print "z_sigma_old[t]:",z_sigma_old[t]
        
        k_sigma_new_temp = k_sigma_old[t]+d_k*stats.geom.rvs(1.0/(1+z_sigma_old[t]),loc=-1,size=1)[0]
        
        print k_sigma_new_temp
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

def ClacKalmanFilter(beta,phi_new_list,varphi_old,sigma_square,x,y):
    ####卡尔曼滤波
    beta_new_list = []
    beta_var_list = []
    beta_final_list = []
    Py_condition_new=0
    SumTemp_new=0
    
    beta_temp_old = np.zeros(len(beta))
    beta_var = np.diag(zip(*phi_new_list)[0])
    K_k = np.dot(np.dot(beta_var,zip(*x)[0]),1.0/(np.dot(np.dot(zip(*x)[0],beta_var),zip(*x)[0])+sigma_square[0]**2))
    beta_temp = beta_temp_old+ np.dot(K_k,y[0]-np.dot(zip(*x)[0],beta_temp_old)) 
    beta_var_new = np.dot(np.identity(len(beta))-np.dot(np.array([K_k]).T,np.array([zip(*x)[0]])),beta_var)
    beta_final_list.append(list(stats.multivariate_normal.rvs(mean=beta_temp,cov=beta_var_new,size=1)))
    beta_new_list.append(beta_temp)
    beta_var_list.append(beta_var_new)
    
    vt_0=(np.dot(np.dot(zip(*x)[0],beta_var),zip(*x)[0])+sigma_square[0]**2)**0.5
    
    for tt in range(1,len(phi_new_list[0])):
        temp_varphi = []
        temp_eta = []
        for kk in range(len(beta)):
            temp_varphi.append(np.sqrt(phi_new_list[kk][tt]/phi_new_list[kk][tt-1])*varphi_old[ii])
            temp_eta.append((1-varphi_old[ii])**2*phi_new_list[kk][tt])
        varphi_t = np.diag(temp_varphi)
        Q_t = np.diag(temp_eta)
        beta_temp_old = np.dot(varphi_t,beta_new_list[tt-1])
        beta_var = np.dot(np.dot(varphi_t,beta_var_list[tt-1]),varphi_t)+Q_t
        K_k = np.dot(np.dot(beta_var,zip(*x)[tt]),1.0/(np.dot(np.dot(zip(*x)[tt],beta_var),zip(*x)[tt])+sigma_square[tt]**2))
        beta_temp = beta_temp_old + np.dot(K_k,y[tt]-np.dot(zip(*x)[tt],beta_temp_old))
        beta_var_new = np.dot(np.identity(len(beta))-np.dot(np.array([K_k]).T,np.array([zip(*x)[tt]])),beta_var)
        beta_new_list.append(beta_temp)
        beta_var_list.append(beta_var_new)
        
        beta_final_list.append(list(stats.multivariate_normal.rvs(mean=beta_temp,cov=beta_var_new,size=1)))
##计算p(y|...)的条件概率

        vt=(np.dot(np.dot(zip(*x)[tt],beta_var),zip(*x)[tt])+sigma_square[tt]**2)**0.5
        et=y[tt]-np.dot(zip(*x)[tt],beta_temp_old)
        SumTemp_new=SumTemp_new+(np.log(vt)+et**2/vt)
    
    beta_final_list = list(np.array(beta_final_list).T)
    
    SumTemp_new=SumTemp_new+(np.log(vt_0)+y[0]**2/vt_0)
    SumTemp_new = -0.5*SumTemp_new 
    #Py_condition_new=np.exp(-0.5*SumTemp_new)
    #print"Py_condition_new:",Py_condition_new
    return beta_final_list,SumTemp_new

# xi_old 是三维数组
# beta，xi_sigma_old。phi_old，k_old 是二维数组
# sigma_old，k_sigma_old，lambda_old，u_old，rou_old,varphi_old,s_xi_old是一维数组
# alpha_hat,alpha_hat_sigma，iterNum,eta,eta_sigma，lambda_sigma_old,u_sigma_old, rou_sigma_old,s_xi_sigma_old 是常数
def UpdateTheta(alpha_hat,alpha_hat_sigma,iterNum,eta,eta_sigma,sigma_old,k_sigma_old,beta,lambda_old,u_old,rou_old,varphi_old,xi_old,s_xi_old_list,lambda_sigma_old,u_sigma_old,rou_sigma_old,xi_sigma_old,s_xi_sigma_old,phi_old_list,k_old_list,x,y):
    # step 1

    phi_new_list = []
    k_new_list =[]
    sigma_new = []
    k_sigma_new =[]

    #xi_new_list is 2-dim
    xi_new_list = []
    xi_sigma_new = []
    s_xi_new= []
    s_xi_sigma_new = 0

    xi_new_temp = []

    lambda_i_new_list=[]
    u_i_new_list=[]
    rou_i_new_list=[]
    varphi_i_new_list=[]

    #step 2
    for ii in range(len(beta)):
        lambda_i_old = copy.deepcopy(lambda_old[ii])
        u_i_old = copy.deepcopy(u_old[ii])
        rou_i_old = copy.deepcopy(rou_old[ii])
        varphi_i_old = copy.deepcopy(varphi_old[ii])
        phi_old = copy.deepcopy(phi_old_list[ii])
        k_i_old = copy.deepcopy(k_old_list[ii])
        s_xi_old = copy.deepcopy(s_xi_old_list[ii])
        # xi_i 和 xi_sigma 两者要分开讨论，一个是二维数组，一个是一维数组
        xi_i =[np.log(lambda_i_old),np.log(u_i_old),np.log(rou_i_old)-np.log(1-rou_i_old),np.log(varphi_i_old)-np.log(1-varphi_i_old)]
        print "xi_i:",xi_i
        # xi_old 的列数为4，而np.cov()的接口是，行数为指标的个数，即位4，故需要转置
        #S_xi的type 是array，4*4
        #xi_old 是一个三维数组，第一个是 迭代次数，第二个是 ii，第三个是4，要取出第二个参数
        S_xi = np.cov(np.array(zip(*xi_old)[ii]).T)
        xi_i_1 = stats.multivariate_normal.rvs(mean=xi_i,cov=np.dot(s_xi_old,S_xi),size=1)
        print "xi_i_1",xi_i_1
        xi_new_temp.append(xi_i_1) 
        # step 3

        lambda_i_new  = np.exp(xi_i_1[0])
        u_i_new = np.exp(xi_i_1[1])
        rou_divide_old  = rou_i_old/(1-rou_i_old)
        rou_divide_new = np.exp(float(xi_i_1[2]))
        varphi_divide_new = np.exp(float(xi_i_1[3]))
        lambda_i_new_list.append(lambda_i_new)
        u_i_new_list.append(u_i_new)
        rou_i_new_list.append(((rou_divide_new)**-1+1)**-1)
        varphi_i_new_list.append(((varphi_divide_new)**-1+1)**-1)


        phi_new = []
        k_i_new = []

        if lambda_i_new > lambda_i_old:
            temp = stats.gamma.rvs(lambda_i_new-lambda_i_old,scale = 1.0/(lambda_i_new/u_i_new),size=1)
            phi_new.append(lambda_i_old*u_i_new/(lambda_i_new*u_i_old)*phi_old[0]+temp[0])
        else:
            temp = stats.beta.rvs(lambda_i_new ,lambda_i_old-lambda_i_new,size=1)
            phi_new.append(lambda_i_old*u_i_new/(lambda_i_new*u_i_old)*phi_old[0]*temp[0])

        for t in range(len(phi_old)-1):

            if rou_divide_new*phi_new[t]*lambda_i_new/u_i_new > rou_divide_old*phi_old[t]*lambda_i_old/u_i_old:
                k_X= stats.poisson(rou_divide_new*phi_new[t]*lambda_i_new/u_i_new -rou_divide_old*phi_old[t]*lambda_i_old/u_i_old)
                temp_sample = k_X.rvs(1)
                k_i_new.append(k_i_old[t]+temp_sample[0])

            else:
                k_X = stats.binom.rvs(k_i_old[t],rou_divide_new*phi_new[t]*lambda_i_new/u_i_new/(rou_divide_old*phi_old[t]*lambda_i_old/u_i_old),size =1)
                k_i_new.append(k_X[0])

            if t<len(phi_old)-1:
                if lambda_i_new+k_i_new[t] > lambda_i_old + k_i_old[t]:
                    temp_sample = stats.gamma.rvs(lambda_i_new+k_i_new[t]-lambda_i_old-k_i_old[t],scale = 1.0/(lambda_i_new/u_i_new*(1+rou_divide_new)),size=1)
                    phi_new.append(lambda_i_new*u_i_new*(1+rou_divide_old)/(lambda_i_old*u_i_old*(1+rou_divide_new))+temp_sample[0])
                else:
                    temp = stats.beta.rvs(lambda_i_new+k_i_new[t],lambda_i_old+k_i_old[t]-lambda_i_new-k_i_new[t],size=1)
                    phi_new.append(lambda_i_new*u_i_new*(1+rou_divide_old)/(lambda_i_old*u_i_old*(1+rou_divide_new))*temp[0])
        phi_new_list.append(phi_new)
    
    xi_sigma = [np.log(lambda_sigma_old),np.log(u_sigma_old),np.log(rou_sigma_old)-np.log(1-rou_sigma_old)]
    S_xi_sigma = np.cov(np.array(xi_sigma_old).T)
    xi_sigma_1 = stats.multivariate_normal.rvs(mean=xi_sigma,cov= np.dot(s_xi_sigma_old,S_xi_sigma),size=1)
    print "xi_sigma_1:",xi_sigma_1
    lambda_sigma_new = np.exp(xi_sigma_1[0])
    u_sigma_new = np.exp(xi_sigma_1[1])
    rou_sigma_divide_new = np.exp(xi_sigma_1[2])
    rou_sigma_new=(rou_sigma_divide_new**-1+1)**-1
    rou_sigma_divide_old = rou_sigma_old/(1-rou_sigma_old)

    if lambda_sigma_new >lambda_sigma_old:
        temp = stats.gamma.rvs(lambda_sigma_new-lambda_sigma_old,scale =lambda_sigma_new/u_sigma_new,size=1)
        sigma_new.append(lambda_sigma_old*u_sigma_new/(lambda_sigma_new*u_sigma_old)*sigma_old[0]+temp[0])
    else:
        temp = stats.beta.rvs(lambda_sigma_new,lambda_sigma_old-lambda_sigma_new ,size =1)
        sigma_new.append(lambda_sigma_old*u_sigma_new/(lambda_sigma_new*u_sigma_old)*sigma_old[0]*temp[0])

    for t in range(len(sigma_old)-1):
        if rou_sigma_divide_new*phi_new[t]*lambda_sigma_new/u_sigma_new >rou_sigma_divide_old*phi_old[t]*lambda_sigma_old/u_sigma_old:
            temp = stats.poisson.rvs(rou_sigma_divide_new*phi_new[t]*lambda_sigma_new/u_sigma_new-rou_sigma_divide_old*phi_old[t]*lambda_sigma_old/u_sigma_old,size=1)
            k_sigma_new.append(k_sigma_old[t]+temp[0])
        else:
            temp = stats.binom.rvs(k_sigma_old[t],rou_sigma_divide_new*phi_new[t]*lambda_sigma_new/u_sigma_new/(rou_sigma_divide_old*phi_old[t]*lambda_sigma_old/u_sigma_old),size=1)
            k_sigma_new.append(temp[0])

        if lambda_sigma_new+k_sigma_new[t-1]>lambda_sigma_old+k_sigma_old[t-1]:
            temp = stats.gamma.rvs(lambda_sigma_new-lambda_sigma_old+k_sigma_new[t-1]-k_sigma_old[t-1],scale=1.0/(lambda_sigma_new/u_sigma_new*(1+rou_sigma_divide_new)),size=1)
            sigma_new.append(temp[0]+lambda_sigma_new*u_sigma_new*(1+rou_sigma_divide_old)/(lambda_sigma_old*u_sigma_old*(1+rou_sigma_divide_new)))
        else:
            temp = stats.beta.rvs(lambda_sigma_new+k_sigma_new[t-1],lambda_sigma_old+k_sigma_old[t-1]-lambda_sigma_new-k_sigma_new[t-1],size=1)
            sigma_new.append(temp[0]*lambda_sigma_new*u_sigma_new*(1+rou_sigma_divide_old)/(lambda_sigma_old*u_sigma_old*(1+rou_sigma_divide_new)))

    print sigma_new,len(sigma_new)
    print k_sigma_new,len(k_sigma_new)
#Step 4 and 5
    #调用卡尔曼滤波函数，注意实际穿入的是 sigma 而不是 sigma_square
    beta_phi_old_simga_old,p_y_phi_old_sigma_old = ClacKalmanFilter(beta,phi_old_list,varphi_old,sigma_old,x,y)
    beta_phi_new_sigma_old,p_y_phi_new_sigma_old = ClacKalmanFilter(beta,phi_new_list,varphi_old,sigma_old,x,y)
    beta_phi_old_simga_new,p_y_phi_old_sigma_new = ClacKalmanFilter(beta,phi_old_list,varphi_old,sigma_new,x,y)
    beta_phi_new_sigma_new,p_y_phi_new_sigma_new = ClacKalmanFilter(beta,phi_new_list,varphi_old,sigma_new,x,y)

##计算两个接受概率，由于伪代码原因，ap与ap_sigma求不出！！！这里跑不通！！
    prior_theta_old=1
    prior_theta_new=1
    prior_theta_sigma=lambda_sigma_old**3*np.exp(-3*lambda_sigma_old)*u_sigma_old*(1+u_sigma_old)**-1.5*rou_sigma_old**38*(1-rou_sigma_old**2)
    prior_theta_sigma_new=lambda_sigma_new**3*np.exp(-3*lambda_sigma_new)*u_sigma_new*(1+u_sigma_new)**-1.5*rou_sigma_new**38*(1-rou_sigma_new**2)
    for ii in range(len(beta)):
        prior_theta_old=prior_theta_old*lambda_old[ii]**2*(0.5+lambda_old[ii])**-4*u_old[ii]**lambda_star*np.exp(-u_old[ii]*lambda_star/u_star)*rou_old[ii]**77.6*(1-rou_old[ii])**2.4*varphi_old[ii]**77.6*(1-varphi_old[ii])**2.4
        prior_theta_new=prior_theta_new*lambda_i_new_list[ii]**2*(0.5+lambda_i_new_list[ii])**-4*u_i_new_list[ii]**lambda_star*np.exp(-u_i_new_list[ii]*lambda_star/u_star)*rou_i_new_list[ii]**77.6*(1-rou_i_new_list[ii])**2.4*varphi_i_new_list[ii]**77.6*(1-varphi_i_new_list[ii])**2.4

    ap=min(1,np.exp(p_y_phi_new_sigma_old-p_y_phi_old_sigma_old )*prior_theta_new/prior_theta_old)
    ap_sigma=min(1,np.exp(p_y_phi_old_sigma_new-p_y_phi_old_sigma_old)*prior_theta_sigma_new/prior_theta_sigma)
    print "ap:",ap 
    print "ap_sigma:",ap_sigma
    s_xi_new=[]
    xi_new =[]
    for jj in range(len(beta)):
        xi_ap = stats.binom(1,ap)
        temp = xi_ap.rvs(1)
        # 依据接受概率update  xi
        if temp[0]==1:
            xi_new.append(xi_new_temp[jj])
        else:
            xi_new.append(xi_old[-1][jj])
##更新s_xi标量
        s_xi_new.append(np.exp(np.log(s_xi_old_list[jj])+iterNum**-eta*(ap-alpha_hat)))
##更新xi_sigma以及s_xi_sigma
    xi_sigma_new=[]
    xi_sigma_ap = stats.binom(1,ap_sigma)
    temp = xi_sigma_ap.rvs(1)
    if temp[0]==1:
        xi_sigma_new=xi_sigma_1
    else:
        xi_sigma_new=xi_sigma_old[-1]
    s_xi_sigma_new=np.exp(np.log(s_xi_sigma_old)+iterNum**-eta_sigma*(ap_sigma-alpha_hat_sigma))

    return xi_new,xi_sigma_new,s_xi_new,s_xi_sigma_new,beta_phi_new_sigma_new



#u,是一维数组
# u_star,tao,b_star,lambda_star,m,iterNum,eta,alpha 是常数
def UpdateU_star(u_star_old,tao_u_star,b_star,lambda_star,m,iterNum,eta,alpha):
    log_u_star_X = stats.norm(loc =np.log(u_star_old),scale =np.sqrt(tao_u_star))
    log_u_star = log_u_star_X.rvs(1)
    u_star = np.exp(log_u_star[0])

    p_u_star = (u_star_old+2*b_star)**(-3)*(1.0/u_star_old)**(m*lambda_star)*np.exp(-1.0*lambda_star/u_star_old*sum(u))
    p_u_star_new = (u_star+2*b_star)**(-3)*(1.0/u_star)**(m*lambda_star)*np.exp(-1.0*lambda_star/u_star*sum(u))

    ap = min(1,p_u_star_new/p_u_star)

    # step 3

    temp = stats.binom.rvs(1,ap,size=1)
    u_star_new = u_star_old
    if temp[0] ==1:
        u_star_new = u_star
    # step 4
    log_tao_u_star = np.log(tao_u_star)+iterNum**(-1.0*eta)*(ap-alpha)
    tao_u_star_new = np.exp(log_tao_u_star)

    return u_star_new,tao_u_star_new

# u 是一维数组
# lambda_star,tao,s_star,u_star,m,eta,alpha,iterNum 是常数

def UpdateLambda(lambda_star_old,tao_lambda_star_old,s_star,u_star,m,u,eta,alpha,iterNum):
    log_lambda_star_X = stats.norm(loc =np.log(lambda_star_old),scale =np.sqrt(tao_lambda_star_old))
    log_lambda_star = log_lambda_star_X.rvs(1)
    lambda_star = np.exp(log_lambda_star[0])

    temp =1.0
    for item in u:
        temp = temp*item**lambda_star_old
    
    p_lambda_star = np.exp(-1.0*lambda_star_old/s_star)*(((lambda_star_old**lambda_star_old)/(u_star**lambda_star_old*spec.gamma(lambda_star_old)))**m)*np.exp(-1.0*lambda_star_old/u_star*sum(u))*temp

    temp =1.0
    for item in u:
        temp = temp*item**lambda_star
   
    print "test:"
    print temp
    print sum(u)
    print -1.0*lambda_star/u_star*sum(u)
    print "\n"
    p_lambda_star_new = np.exp(-1.0*lambda_star/s_star)*(((lambda_star**lambda_star)/(u_star**lambda_star*spec.gamma(lambda_star)))**m)*np.exp(-1.0*lambda_star/u_star*sum(u))*temp
    print "p_lambda_star:", p_lambda_star
    print "p_lambda_star_new:",p_lambda_star_new
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
    #beta 存放的是初值，后续不需要更改
    #最终的生成的beta 添加到beta_list 里面
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
        if (ii>=20 and ii<=49)or (ii>=120 and ii<=149):
            beta_3.append(-2)
        else:
            beta_3.append(0)
        beta_4.append(0)
        beta_5.append(0)
    beta= [beta_0,beta_1,beta_2,beta_3,beta_4,beta_5]
    beta_list = []

    lambda_old = 2
    alpha_hat = 0.3
    alpha_hat_sigma = 0.3
    sigma = 1
    #y
    y= []
    for ii in range(T):
        y_ii = stats.norm.rvs(scale =sigma ,size=1)[0]
        for jj in range(m+1):
            y_ii = y_ii + x[jj][ii]*beta[jj][ii]
        y.append(y_ii)
    # u,rou,lambda,varphi, lambda_star,u_star
    u = list(0.1*np.ones(m+1))
    s_star = 0.1
    b_star = 0.1
    u_star = 1 
    lambda_star = 1 
    lambda_old = list(0.1*np.ones(m+1))
    varphi_old = list(0.97*np.ones(m+1))
    rou_old = list(0.97*np.ones(m+1))
    tao_u_star = 1
    tao_lambda_star = 0.95
    
    lambda_sigma = 3 
    k_sigma = list(np.ones(T))
    u_sigma = 0.03
    rou_sigma = stats.beta.rvs(38,2.0,size=1)[0]
    #生成phi,tao,k,z
    tao_phi_old = 0.3*np.ones((m+1,T),dtype=np.int16)
    phi_old_temp = []
    k_old_temp = []
    k_per_row = []
    z_old = []
    for ii in range(m+1):
        temp_z_old = []
        for jj in range(T):
            temp_z_old.append(random.uniform(0.3,0.7))
        z_old.append(temp_z_old)

    # 初值有待改进  
    xi_old  = list(np.random.random((100,6,4)))
    xi_sigma_old = list(np.random.random((100,3)))
    s_xi_old = [1,1,1,1,1,1]
    s_xi_sigma_old = 1
    

    print "initial value:\n"
    print "rou_old:",rou_old 
    print "lambda_old:", lambda_old
    print "u:", u

    phi_old = []
    k_old = []
    for ii in range(m+1):
        temp_phi_init = [1]
        phi_old.append(temp_phi_init)
        temp_k_init = []
        temp_k_init.append(stats.poisson.rvs(rou_old[ii]*lambda_old[ii]*phi_old[ii][0]/(u[ii]*(1-rou_old[ii])),size=1)[0])
        k_old.append(temp_k_init)
        for jj in range(1,T):
            phi_old[ii].append(1)
            temp_k = stats.poisson.rvs(rou_old[ii]*lambda_old[ii]*phi_old[ii][-1]/(u[ii]*(1-rou_old[ii])),size=1)[0]
            k_old[ii].append(temp_k)
    print "phi_old"
    print phi_old
    print "\n"
    print "k_old"
    print k_old

    eta = 0.55
    eta_sigma = 0.3
    z_sigma_old = []
    for ii in range(T):
        z_sigma_old.append(4*1.0/3)

    for ii in range(1000):
        print "It is the "+str(ii) + " 's iteration." 
        iterNum = ii +1
        print "update phi:"
        phi_new,tao_phi_new = UpdatePhi(phi_old,tao_phi_old,lambda_old,k_old,beta,u,rou_old,varphi_old,iterNum,eta,alpha_hat,T)
        print "update kappa:"
        k_new,z_new = UpdateK(k_old,z_old,T,lambda_old,rou_old,u,phi_new,alpha_hat,iterNum,eta)
        print"k_new:"
        print k_new
        #print "update sgima^2"
        temp_sigma,sigma_square = UpdateSigma(x,y,T,beta,rou_sigma,u_sigma,lambda_sigma,k_sigma)
        print "sigma_new:"
        #print temp_sigma
        print "update kappa^{sgima}"
        k_sigma_new,z_sigma_new = UpdateK_sigma(k_sigma,z_sigma_old,lambda_sigma,rou_sigma,u_sigma,sigma_square,iterNum,eta,alpha_hat,T)

        #2.5
        #xi_new is 2-dim
        # xi_sigma_new is 1-dim s_xi_new is 1-dim, either
        # s_xi_sigma_new is const
        print "update theta,beta"
        xi_new,xi_sigma_new,s_xi_new,s_xi_sigma_new, beta_new = UpdateTheta(alpha_hat,alpha_hat_sigma,iterNum,eta,eta_sigma,temp_sigma,k_sigma_new,beta,lambda_old,u,rou_old,varphi_old,xi_old,s_xi_old,lambda_sigma,u_sigma,rou_sigma,xi_sigma_old,s_xi_sigma_old,phi_new,k_new,x,y)
        #还原 xi 到变量中
        # xi_new 是二维数组
        print "xi_new,",xi_new
        xi_old.append(xi_old)
        s_xi_old.append(s_xi_new)
        lambda_new = np.exp(zip(*xi_new)[0])
        u_new = np.exp(zip(*xi_new)[1])
        rou_new = np.exp(zip(*xi_new)[2])/(np.exp(zip(*xi_new)[2])+1)
        varphi_new = np.exp(zip(*xi_new)[3])/(np.exp(zip(*xi_new)[3])+1)

        lambda_sigma_new = np.exp(xi_sigma_new[0])
        u_sigma_new = np.exp(xi_sigma_new[1])
        rou_sigma_new =np.exp(xi_sigma_new[2])/(np.exp(xi_sigma_new[2])+1)
        
        # u_i 不等于 u，u_i是2.5得到的
        u_star_new, tao_u_star_new = UpdateU_star(u_star,tao_u_star,b_star,lambda_star,m,iterNum,eta,alpha_hat)
        
        lambda_star_new,tao_lambda_star_new = UpdateLambda(lambda_star,tao_lambda_star,s_star,u_star_new,m,u_new,eta,alpha_hat,iterNum)

        #在这里更新变量
        #deepcopy

        phi_old = copy.deepcopy(phi_new)
        tao_phi_old = copy.deepcopy(tao_phi_new)
        lambda_old = copy.deepcopy(lambda_new)
        k_old = copy.deepcopy(k_new)
        u = copy.deepcopy(u_new)
        rou_old = copy.deepcopy(rou_new)
        varphi_old =copy.deepcopy(varphi_new)
        z_old = copy.deepcopy(z_new)
        rou_sigma = copy.deepcopy(rou_sigma_new)
        u_sigma = copy.deepcopy(u_sigma_new)
        lambda_sigma=copy.deepcopy(lambda_sigma_new)
        k_sigma = copy.deepcopy(k_sigma_new)
        z_sigma_old = copy.deepcopy(z_sigma_new)

        u_star= copy.deepcopy(u_star_new)
        tao_u_star = copy.deepcopy(tao_u_star_new)
        lambda_star = copy.deepcopy(lambda_star_new)
        tao_lambda_star = copy.deepcopy(tao_lambda_star_new)

        beta_list.append(beta_new)
    #print lambda_star_new
    #print tao_lambda_star_new
    print "beta new:", beta_list
    print "\nfinished!\n"

