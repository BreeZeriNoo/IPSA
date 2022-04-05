#!/usr/bin/env python
# coding: utf-8

# Lecture 17 (Visualization and optimization)
# 

# In[4]:


print("Hello World!")


# In[31]:


import matplotlib.pyplot as plt
import math
import numpy as np
import scipy.optimize as opt
def f(x):
    return 100*(x - 3)**2+math.exp(x) 

xvals = np.linspace(0,10,100) 
yvals = list(map(f, xvals))

plt.plot(xvals, yvals,color="Red")

opt.minimize(f,0)


# In[87]:


"""
Exercise 18.1 (numpy)
Generate two 5x5 matrices with entries drawn from a standard normal distribution (mean 0, variance 1).
numpy.random.normal might come in handy here. Find the inverse of the matrices. 
Check by matrix multiplication, that it is in fact the inverses you have found.

"""

A=np.array([np.random.normal(0, 1, size=(5,5))])
B=np.array([np.random.normal(0, 1, size=(5,5))])

As=np.linalg.inv(A)
Bs=np.linalg.inv(B)

#Test om inverse: 
print("A=",+np.round(np.matmul(A,As),decimals=0, out=None))
print("B=",+np.round(np.matmul(B,Bs),decimals=0, out=None))

#Altså er de inverser.

A.shape


# In[103]:


"""
Find the mean of each row and each column of the matrices without using loops. numpy.mean might come in handy here.

"""
def mean_row(A):
    result_row=[]
    s_row=A.shape[1]
    a_row=np.zeros(s_row)
    for i in range(s_row):
        a_row[i]=1
        result_row.append(np.mean(np.matmul(a_row,A)))
    
    return result_row

def mean_col(A):
    result_col=[]
    s_col=A.shape[2]
    a_col=np.zeros(s_col)
    for j in range(s_col):
        a_col[j]=1
        result_col.append(np.mean(np.matmul(A,np.transpose(a_col))))
    return result_col

#mean_mat(A)        


# In[188]:


"""
Create a 100x100 matrix with entries drawn from the uniform distribution between 0 and 1. Find all rows where the sum exceeds 55. 
Find both the row index and the sum. 
(Try to have a look at the functions numpy.sum and numpy.where)
"""
A=np.array([np.random.uniform(0, 1, size=(1000,1000))])

t=mean_row(A)


def sol_55(t):
    sol=[]
    for i in range(len(t)):
        if t[i]>=55:
            sol.append((i,t[i]))
    return sol
#print(len(sol_55(t))/10)


# In[214]:


import numpy as np
np.random.uniform(0, 1, size=3)

def approximate_pi(number_of_throws):
    hits = 0
    points=[]
    for i in range(number_of_throws):
        x, y ,z = np.random.uniform(0, 1, size=3)
        r = np.sqrt(x**2 + y**2 + z**2)
        hits = hits+(r < 1)
        points.append(r)
    return [hits/number_of_throws*6,points]
#print(approximate_pi(10000))

A=np.array([np.random.uniform(0, 1, size=(1000,1000))])
B=np.array([np.random.uniform(0, 1, size=(1000,1000))])

def points_inred(A,B):
    points_inred=[]
    for i in range(A.shape[1]):
        for j in range(A.shape[1]):
            if np.sqrt(A[i,j]**2 + B[i,j]**2)<=1:
                points_inred.append[(A[np.array([i[i],j[j]])],B[np.array([i[i],j[j]])])]
    return points_inred            
print(points_inred(A,B))


# In[ ]:





# In[ ]:




