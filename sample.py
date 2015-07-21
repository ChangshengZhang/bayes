#!/usr/bin/python
# -*- coding: utf-8 -*-
# File Name: data.py
# Author: Changsheng Zhang
# mail: zhangcsxx@gmail.com

import numpy as np
import numpy
from scipy import stats
import scipy.special as spec
import math
import random

# iterNum 表示迭代次数
def UpdatePhi(phi_old,tao_phi_old,lambda_old,k_old,beta_old,u,rou,varphi,iterNum,eta,alpha_phi_old,T):
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

	if temp[0]==0:
		phi_i1_new = phi_i1_new_temp
	else:
		phi_i1_new = phi_old[0]

	phi_new.append(phi_i1_new)

	# step 4
	log_tao_i1_new = np.log(tao_phi_old[0])+(iterNum)**(-1.0*eta)*(ap_phi-alpha_phi_old[0])
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

		ap_phi = min(1,1.0*p_Phi_it_new/p_Phi_it)

		# step 3

		yPhi = stats.binom(1,ap_phi)
		temp  =yPhi.rvs(1)
		if temp[0]==0:
			phi_it_new = phi_it_new_temp
		else:
			phi_it_new = phi_old[t]

		phi_new.append(phi_it_new)

		# step 4
		log_tao_it_new = np.log(tao_phi_old[t])+(iterNum)**(-1.0*eta)*(ap_phi-alpha_phi_old[t])
		tao_phi_new.append(log_tao_it_new)

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
	log_tao_it_new = np.log(tao_phi_old[t])+(iterNum)**(-1.0*eta)*(ap_phi-alpha_phi_old[t])
	tao_phi_new.append(log_tao_it_new)

	return  phi_new, tao_phi_new



def UpdateK(k_old,z_old,T,lambda_old,rou,u,phi,alpha,iterNum,eta):
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


		# step 3
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

			if temp[0] ==0:
				k_new.append(k_new_temp)
			else:
				k_new.append(k_old[t])

			temp_z = z_old[t]+ iterNum**(-1.0*eta)*(ap-alpha)
			z_new.append(temp_z)

		# step 4


	return  k_new, z_new

# x 是二维矩阵，
def UpdateSigma(x,y,T,beta,rou_sigma,u_sigma,lambda_sigma,k_sigma):

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
				sigma_new_2.append(v_random)
				flag =0
			else:
				flag =1

	return sigma_new_2

# sigma_2 表示方程 sigma^2
def UpdateK_sigma(k_sigma_old,z_sigma_old,lambda_sigma,rou_sigma,u_sigma,sigma_2,iterNum,eta,alpha,T):

	k_sigma_new = []
	z_sigma_new = []

	for t in range(T-1):

		dK = stats.binom(1,0.5)
		temp = dK.rvs(1)
		d_k = 0
		if temp[0] ==0:
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
			if temp[0] ==0:
				k_sigma_new.append(k_sigma_new_temp)
			else:
				k_sigma_new.append(k_sigma_old[t])

			# step 4
			temp_z = z_sigma_old[t]+ iterNum**(-1.0*eta)*(ap-alpha)
			z_sigma_new.append(temp_z)



	return  k_sigma_new,z_sigma_new


# x_i_old is a list, xi_sigma_old is a list too.
def UpdateTheta(m,alpha_theta,alpha_theta_sigma,alpha_hat,alpha_hat_sigma,iterNum,eta,eta_sigma,sigma_old,k_sigma_old,beta,lambda_i_old,u_i_old,rou_i_old
                ,varphi_i_old,xi_old,s_xi_old,lambda_sigma_old,u_sigma_old,rou_sigma_old,xi_sigma_old,s_xi_sigma_old,phi_old,k_i_old):
	# step 1
	s_i=[]
	beta_M =[]
	beta_C =[]
	for i in range(m):
		s_bi = stats.binom(1,1-5.0/m)
		temp = s_bi.rvs(1)
		s_i.append(temp[0])

		if s_bi ==1:
			beta_M.append(beta[i])
		else:
			beta_C.append(beta[i])

	# step 2
	xi_i =[np.log(lambda_i_old),np.log(u_i_old),np.log(rou_i_old)-np.log(1-rou_i_old),np.log(varphi_i_old)-np.log(1-varphi_i_old)]

	S_xi = np.cov(xi_old,rowvar=False)

	xi_X = stats.norm(loc = xi_i,scale = np.dot(s_xi_old,S_xi))
	xi_i_1 = xi_X.rvs(1)
	xi_sigma =[np.log(lambda_sigma_old),np.log(u_sigma_old),np.log(rou_sigma_old)-np.log(1-rou_sigma_old)]

	S_xi_sigma = np.cov(np.array(xi_sigma_old),rowvar= False)
	xi_sigma_1 = stats.norm(loc = xi_sigma,scale =np.dot(s_xi_sigma_old,S_xi_sigma)).rvs(1)


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
		temp = stats.bernoulli.rvs(lambda_i_old-lambda_i_new,loc =lambda_i_new,size=1)
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
				temp = stats.bernoulli.rvs(lambda_i_old+k_i_old[t-1]-lambda_i_new-k_i_new[t-1],loc =lambda_i_new+k_i_new[t-1],size=1)
				phi_new.append(lambda_i_new*u_i_new*(1+rou_divide_old)/(lambda_i_old*u_i_old*(1+rou_divide_new))*temp[0])

	sigma_new = []
	k_sigma_new =[]
	lambda_sigma_new = np.exp(xi_sigma_1[0])
	u_sigma_new = np.exp(xi_sigma_1[1])
	rou_sigma_divide_new = np.exp(xi_sigma_1[2])
	rou_sigma_divide_old = rou_sigma_old/(1-rou_sigma_old)

	if lambda_sigma_new >lambda_sigma_old:
		temp = stats.gamma.rvs(lambda_sigma_new-lambda_sigma_old,lambda_sigma_new/u_sigma_new,size=1)
		sigma_new.append(lambda_sigma_old*u_sigma_new/(lambda_sigma_new*u_sigma_old)*sigma_old[0]+temp[0])

	else:
		temp = stats.bernoulli.rvs(lambda_sigma_old-lambda_sigma_new,loc = lambda_sigma_new,size =1)
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
				temp = stats.bernoulli.rvs(lambda_sigma_old+k_sigma_old[t-1]-lambda_sigma_new-k_sigma_new[t-1],loc=lambda_sigma_new+k_sigma_new[t-1],size=1)
				sigma_new.append(temp[0]*lambda_sigma_new*u_sigma_new*(1+rou_sigma_divide_old)/(lambda_sigma_old*u_sigma_old*(1+rou_sigma_divide_new)))

	# step 4



	# step 5
	ap = 0.5
	ap_sigma = 0.5



	# step 6
	xi_new =[]
	xi_sigma_new = []
	xi_ap = stats.binom(1,ap)
	temp = xi_ap.rvs(1)
	# update
	if temp==0:
		for ii in range(len(xi_i_1)):

			xi_new.append(xi_i_1[ii])

		for ii in range(len(phi_new)):
			phi_old[ii]=phi_new[ii]

		for ii in range(len(k_i_new)):
			k_i_old[ii] = k_i_new[ii]

	else:
		for ii in range(len(xi_i)):
			xi_new.append(xi_i[ii])

	sigma_ap = stats.binom(1,ap_sigma)
	temp =sigma_ap.rvs(1)
	if temp ==0:
		for ii in range(len(xi_sigma_1)):
			xi_sigma_new.append(xi_sigma_1[ii])

		for ii in range(len(sigma_new)):
			sigma_old[ii]=sigma_new[ii]

		for ii in range(len(k_sigma_new)):
			k_sigma_old[ii]=k_sigma_new[ii]

	else:
		for ii in range(len(xi_sigma)):
			xi_sigma_new.append(xi_sigma[ii])

	xi_old.append(xi_new)
	xi_sigma_old.append(xi_sigma_new)

	# step 7
	s_xi_new= []
	s_xi_sigma_new =[]
	for i in range(len(s_xi_old)):

		temp = np.exp(np.log(s_xi_old[i])+iterNum**(-1.0*eta)*(alpha_theta-alpha_hat))
		s_xi_new.append(temp)
	for i in range(len(s_xi_sigma_old)):

		temp = np.exp(np.log(s_xi_sigma_old[i])+iterNum**(-1.0*eta_sigma)*(alpha_theta_sigma-alpha_hat_sigma))
		s_xi_sigma_new.append(temp)




	return xi_new,xi_sigma_new,s_xi_new,s_xi_sigma_new



def UpdateBeta(phi,varphi,T):

	beta =[]
	beta_X = stats.norm(loc=0,scale  =phi[0])
	beta_1 = beta_X.rvs(1)
	beta.append(beta_1[0])
	for t in range(1,T):
		eta_X = stats.norm(loc =0,scale =(1-varphi**2)*phi[t])
		eta_t = eta_X.rvs(1)
		beta_t =(phi[t]/phi[t-1])**0.5*varphi*beta[t-1] + eta_t[0]
		beta.append(beta_t)

	return  beta

def UpdateU_star(u_star_old,u,tao,b_star,lambda_star,m,iterNum,eta,alpha):
	log_u_star_X = stats.norm(loc =np.log(u_star_old),scale =tao**(u_star_old) )
	log_u_star = log_u_star_X.rvs(1)
	u_star = np.exp(log_u_star[0])

	p_u_star = (u_star_old+2*b_star)**(-3)*(1.0/u_star_old)**(m*lambda_star)*np.exp(-1.0*lambda_star/u_star_old*sum(u))
	p_u_star_new = (u_star+2*b_star)**(-3)*(1.0/u_star)**(m*lambda_star)*np.exp(-1.0*lambda_star/u_star*sum(u))

	ap = min(1,p_u_star_new/p_u_star)
		# step 3

	y_AP = stats.binom(1,ap)
	u_star_new = u_star_old
	temp = y_AP.rvs(1)
	if temp[0] ==0:
		u_star_new = u_star

	# step 4

	log_tao_u_star = np.log(tao**(u_star_old))+iterNum**(-1.0*eta)*(ap-alpha)
	tao_u_star_new = (np.exp(log_tao_u_star))**(1.0/u_star_new)

	return u_star_new,tao_u_star_new

def UpdateLambda(lambda_star_old,tao,s_star,u_star,m,u,eta,alpha,iterNum):
	log_lambda_star_X = stats.norm(loc =np.log(lambda_star_old),scale =tao**(lambda_star_old) )
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
	if temp[0] ==0:
		lambda_star_new = lambda_star

	# step 4

	log_tao_lambda_star = np.log(tao**(lambda_star_old))+iterNum**(-1.0*eta)*(ap-alpha)
	tao_lambda_star_new = (np.exp(log_tao_lambda_star))**(1.0/lambda_star_new)

	return lambda_star_new,tao_lambda_star_new



if __name__ =='__main__':
	#初始值
	T = 10

	phi_old = [1,1,1,1,1,1,1,1,1,1]
	tao_phi_old = [1,1,1,1,1,1,1,1,1,1]
	lambda_old = 1
	k_old = [2,1,4,1,2,1,4,1,1,1]
	z_old = [1,1,1,1,1,1,1,1,1,1]

	# 这里的beta_old 是 beta 里的一行，之后要统一形式！！
	beta_old = [1,1,1,1,1,1,1,1,1,1]
	u = 1
	rou = 0.99
	varphi = 0.99
	iterNum =1
	eta = 0.75
	alphi_phi_old =[0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]
	alpha = 0.3

	#update sigma^2
	x = [[1,2,3,2,3,2,3,2,3,1],[2,3,4,2,3,2,3,2,3,1],[5,3,1,2,3,2,3,2,3,1],[1,1,3,2,3,2,3,2,3,1]]
	m = len(x)-1
	y = [1,2,3,3,2,1,1,2,3,2]
	beta = [[1,2,3,2,3,2,3,2,3,1],[2,3,4,2,3,2,3,2,3,1],[5,3,1,2,3,2,3,2,3,1],[1,1,3,2,3,2,3,2,3,1]]

	# rou_sigma != 1
	rou_sigma = 0.95
	u_sigma = 2.0
	lambda_sigma = 1.0
	k_sigma = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]



	#update k_sigma
	z_sigma_old = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]

	#2.5 update

	m =5
	alpha_theta = 0.97
	alpha_theta_sigma = 0.97
	alpha_hat = 0.97
	alpha_hat_sigma = 0.97
	eta_sigma = 0.75

	xi_old = [[1,1,1,1]]
	xi_sigma_old = [[1,1,1]]
	s_xi_old = [1,1,1,1]
	s_xi_sigma_old = [1,1,1]



	# update U_star
	u_star_old = 2
	tao = 0.8
	b_star= 1
	lambda_star = 2

	#u_i 是在2.5得到的，等补上2.5之后需要删除
	u_i = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]

	# update lambda
	# s_star,u_star 是2.5里的
	s_star =2
	u_star =1.2


	# run sample
	print "begin \n"

	phi_new, tao_phi_new = UpdatePhi(phi_old,tao_phi_old,lambda_old,k_old,beta_old,u,rou,varphi,iterNum,eta,alphi_phi_old,T)
	k_new, z_new = UpdateK(k_old,z_old,T,lambda_old,rou,u,phi_new,alpha,iterNum,eta)

	sigma_square = UpdateSigma(x,y,T,beta,rou_sigma,u_sigma,lambda_sigma,k_sigma)


	k_sigma_new,z_sigma_new= UpdateK_sigma(k_sigma,z_sigma_old,lambda_sigma,rou_sigma,u_sigma,sigma_square,iterNum,eta,alpha,T)


	#2.5
	xi_new,xi_sigma_new,s_xi_new,s_xi_sigma_new = UpdateTheta(m,alpha_theta,alpha_theta_sigma,alpha_hat,alpha_hat_sigma,iterNum,eta,eta_sigma,
	                                                          list(np.sqrt(sigma_square)),k_sigma_new,beta_old,lambda_old,u,rou,varphi,xi_old,s_xi_old,
	                                                          lambda_sigma,u_sigma,rou_sigma,xi_sigma_old,s_xi_sigma_old,phi_new,k_new)
	#还原 xi 到变量中
	lambda_new = np.exp(xi_new[0])
	u_new = np.exp(xi_new[1])
	rou_new = np.exp()

	print xi_new
	print "\n"
	print xi_sigma_new


	beta_new = UpdateBeta(phi_new,varphi,T)

	# u_i 不等于 u，u_i是2.5得到的
	u_star_new, tao_u_star_new = UpdateU_star(u_star_old,u_i,tao,b_star,lambda_star,m,iterNum,eta,alpha)

	lambda_star_new, tao_lambda_star_new = UpdateLambda(lambda_star,tao,s_star,u_star,m,u_i,eta,alpha,iterNum)

	#print lambda_star_new
	#print tao_lambda_star_new

	print "\nfinished!\n"

