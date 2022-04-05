#Lecture 18 (Multi-dimensional data)


#Exercise 18.1 (numpy)
import numpy as np
"""
a. Generate two 5x5 matrices with entries drawn from a standard normal distribution (mean 0, variance 1).
numpy.random.normal might come in handy here. Find the inverse of the matrices.
Check by matrix multiplication, that it is in fact the inverses you have found.
"""

A=np.array([np.random.normal(0, 1, size=(5,5))])
B=np.array([np.random.normal(0, 1, size=(5,5))])

As=np.linalg.inv(A)
Bs=np.linalg.inv(B)

#Test om inverse:
#print("A=",+np.round(np.matmul(A,As),decimals=0, out=None))
#print("B=",+np.round(np.matmul(B,Bs),decimals=0, out=None))

#Altså er de inverser.

"""
b. Find the mean of each row and each column of the matrices without using loops. numpy.mean might come in handy here.
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

"""
c. Create a 100x100 matrix with entries drawn from the uniform distribution between 0 and 1. Find all rows where the sum exceeds 55.
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


#18.2



#18.3
"""
Consider the below list L of 2D points.
Apply numpy.polyfit for different degrees of the polymial fitting, and find the minimum degree polynomial where the maximal vertical distance between a point (x, y) in L and the polynomial poly is at most 1, i.e.
abs(y - poly(x)) <= 1 for all (x, y) in L. Plot the data points and the different polynomial approximations. Present your work in Jupyter.
"""
import matplotlib.pyplot as plt

L = [(2, 3), (5, 7), (4, 10), (10, 15), (2.5, 5), (8, 4), (9, 10), (6, 6)]
x,y=zip(*L)

def min_degree_pol(x,y,n):
    Polynomier=[(np.polyfit(x,y,deg=i)) for i in range(2,n)]
    Muligheder=[]
    for c in Polynomier:
        p=np.poly1d(c)
        for (x,y) in L:
            if np.all(np.abs(y-p(x))<=0.1):
                print((np.abs(y-p(x))))
                Muligheder.append(c)
    return min(Muligheder,key=len)

def plot_polynomium(k,r,koordinater):
    p=np.poly1d(k)
    range=np.arange(r)
    y=p(range)
    plt.plot(range, y)
    x,y=zip(*koordinater)
    plt.scatter(x,y)
    return plt.show()
plot_polynomium(min_degree_pol(x,y,10),12,L)



from scipy.optimize import linprog

"""
Exercise 19.1 (linear program)
Solve the below linear program using scipy.optimize.linprog.

Maximize (objective function)

350·x1 + 300·x2

subject to

x1 + x2 ≤ 200
9·x1 + 6·x2 ≤ 1566
12·x1 + 16·x2 ≤ 2880
x1 ≥ 3
x2 ≥ 7
"""
#Opstil i matrix/vektorform. og omskrives til min ved ændring af fortegn i objektfunktionen.
obs=[-350,-300]
Amatrix=[[1,1],[9,6],[12,16]]
Bvektor=[200,1566,2880]

res=linprog(obs,A_ub=Amatrix,b_ub=Bvektor,bounds=[(3,None),(7,None)],method='revised simplex')
#print(res)
