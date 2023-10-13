"""采用令中间的得分矩阵比例"""
import numpy as np
import pandas as pd
import time 
import math
# import cProfile
import psutil

def CLNF(R_1,R_2,lamta,b_1,b_2,k_2,step_1):
    #k_2为层数，k 为系数
    # 令b_2等于b_1少训练一个超参数
    # begin_time=time.time()
    mem=psutil.virtual_memory()
    # ysy_1=float(mem.used)
    
    R_1=np.array(R_1)
    R_2=np.array(R_2)
    
    Y_1=np.ones_like(R_1)
    Y_2=np.ones_like(R_2)
    Y_1[np.isnan(R_1)]=0
    Y_2[np.isnan(R_2)]=0
    
    R_1_1=R_1.copy()
    R_2_1=R_2.copy()
    R_1_1[np.isnan(R_1)]=0
    R_2_1[np.isnan(R_2)]=0


    t_1=0
    a=R_1.shape[0]
    b=int(round(k_2))
    U=np.random.uniform(0.001,1,(a,b))
    B_1=np.random.uniform(0.001,1,(b,b))
    B_2=np.random.uniform(0.001,1,(b,b))
    B_1=(B_1+B_1.T)/2
    B_2=(B_2+B_2.T)/2

    # array=[]
    # array_mae=[]
    # array_rmse=[]
    # t_1_2=1
    while t_1<step_1: #and t_1_2>0.000000000000001:
        #更新U
        U_mid1=np.dot(U,B_1).dot(U.T)
        U_mid1[np.isnan(R_1)]=0
        # print(U_mid1)
        U_mid2=np.dot(U,B_2).dot(U.T)
        U_mid2[np.isnan(R_2)]=0
        

        negative_U=4*(np.dot(U_mid1,U).dot(B_1)+lamta*np.dot(U_mid2,U).dot(B_2))+2*b_2*U

        postive_U=4*(np.dot(R_1_1,U).dot(B_1)+lamta*np.dot(R_2_1,U).dot(B_2))

        for i in range(a):
            for j in range(b):
                if negative_U[i,j]!=0:
                    U[i,j]=U[i,j]*postive_U[i,j]/negative_U[i,j]
                # else:
                #     print(1)
        # print(U)
        #更新B_1
        B_1_mid1=np.dot(U,B_1).dot(U.T)
        B_1_mid1[np.isnan(R_1)]=0
        negative_B_1=np.dot(U.T,B_1_mid1).dot(U)+b_1*B_1
        
        postive_B_1=np.dot(U.T,R_1_1).dot(U)

        for i in range(b):
            for j in range(b):
                if negative_B_1[i,j]!=0:
                    B_1[i,j]=B_1[i,j]*postive_B_1[i,j]/negative_B_1[i,j]
                    # B_1[j,i]=B_1[i,j]
        B_1=(B_1+B_1.T)/2
        # print(B_1)
        #更新B_2
        B_2_mid1=np.dot(U,B_2).dot(U.T)
        B_2_mid1[np.isnan(R_2)]=0
        negative_B_2=lamta*np.dot(U.T,B_2_mid1).dot(U)+b_1*B_2
        
        postive_B_2=lamta*np.dot(U.T,R_2_1).dot(U)

        for i in range(b):
            for j in range(b):
                if negative_B_2[i,j]!=0:
                    B_2[i,j]=B_2[i,j]*postive_B_2[i,j]/negative_B_2[i,j]
                    # B_2[j,i]=B_2[i,j]
        B_2=(B_2+B_2.T)/2
        # print(B_2)

        # if t_1==0:
        #     end_time=time.time()
        #     run_time=end_time-begin_time
        #     print('一轮用时：',run_time)	

        t_1+=1
    number=np.count_nonzero(Y_1) 
    Lf=np.dot(U,B_1).dot(U.T)
    Lf[np.isnan(R_1)]=0
    result_matrix=R_1-Lf
    result_matrix[np.isnan(R_1)]=0
    result_matrix_norm1=np.linalg.norm(result_matrix, ord=1)
    result_matrix_norm2=np.linalg.norm(result_matrix, ord=2)
    result_1=result_matrix_norm1/number
    result_2=math.sqrt(result_matrix_norm2/number)
        # print('MAE:',result_1,'\nRMSE:',result_2)
        # array.append(result_1)
        # array_mae.append(result_1)
        # array_rmse.append(result_2)

        # if t_1>1:
        # 	t_1_2=abs(array[-1]-array[-2])
        # 	# print('变化',t_1_2)
        # else:
        # 	pass

        
    # end_time=time.time()
    # run_time=end_time-begin_time
    # print('总用时：',run_time)
    zj=float(mem.total)
    ysy=float(mem.used)
    memory=(ysy)/zj
    # return1={}
    # miss1=sum(R_1[R_1==0])/(R_1.shape[0]*R_1.shape[1])
    # miss2=sum(R_2[R_2==0])/(R_2.shape[0]*R_2.shape[1])
    
    # return1['CLNLF模型缺失度:'+str(miss1)+';'+str(miss2)+'下MAE和RMSE']=[array_mae[-1],array_rmse[-1]]
    
    return result_1,result_2,memory    #array_mae[-1],array_rmse[-1]

# cProfile.run('CLNF()', "re")
    