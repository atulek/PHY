# Import libraries, problems and algorithm
import numpy as np

from DE import DE
from ABC import ABC
from PSO import PSO
from GA import GA
from TestFunc import TestFunc
# import matplotlib.pyplot as plt
from mpi4py import MPI

# MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Global Parameters
testFunction = TestFunc().rosenbrock()
repeat = 3
maxIt = 500
nPop = 100
nVar = 50
filename = testFunction.name

# PSO Parameters
c1 = 1.4962
c2 = 1.4962
w = 0.7298
wdamp = 1.0

# DE Parameters
F = 0.2
crossoverProbability = 0.2

# GA Parameters
beta = 0.2
pc = 1
gamma = 0.1
mu = 0.1
sigma = 0.1
k = 7

# Migration Parameters
migRate = 25
migInterval = 25
mRate = int(nPop/100*migRate)

# File name
if migInterval > maxIt:
    fileName = "IPH_%s_rate-%d_int-%d_no.txt" % (filename, migRate, migInterval)
else:
    fileName = "IPH_%s_rate-%d_int-%d_yes.txt" % (filename, migRate, migInterval)
# PSO Default parametreleri
pso = PSO(
    test_func=testFunction,
    maxit=maxIt,
    npop=nPop,
    nvar=nVar,
    c1=c1,
    c2=c2,
    w=w,
    wdamp=wdamp,
    comm=comm,
    psize=size,
    myrank=rank,
    mrate=mRate,
    minterval=migInterval
)

abc = ABC(
    test_func=testFunction,
    maxit=maxIt,
    npop=nPop,
    nvar=nVar,
    comm=comm,
    psize=size,
    myrank=rank,
    mrate=mRate,
    minterval=migInterval
)

de = DE(
    test_func=testFunction,
    maxit=maxIt,
    npop=nPop,
    nvar=nVar,
    F=F,
    cp=crossoverProbability,
    comm=comm,
    psize=size,
    myrank=rank,
    mrate=mRate,
    minterval=migInterval
)

ga = GA(
    test_func=testFunction,
    maxit=maxIt,
    npop=nPop,
    nvar=nVar,
    beta=beta,
    pc=pc,
    gamma=gamma,
    mu=mu,
    sigma=sigma,
    k=k,
    comm=comm,
    psize=size,
    myrank=rank,
    mrate=mRate,
    minterval=migInterval
)

if rank == 0:
    meanCosts = np.empty(maxIt, float)
    allCosts = np.empty((repeat, maxIt), float)
    allLastCosts = np.empty(repeat, float)

for rpt in range(repeat):
    if rank % 3 == 0:
        bestcost = abc.run()
    elif rank % 3 == 1:
        bestcost = de.run()
    elif rank % 3 == 2:
        bestcost = pso.run()
    # elif rank % 4 == 3:
    #     bestcost = pso.run()
    if rank == 0:
        allLastCosts[rpt] = bestcost[-1]
        allCosts[rpt] = bestcost

if rank == 0:
    meanCosts = np.mean(allCosts, axis=0)
    minAllLastCosts = np.min(allLastCosts)
    maxAllLastCosts = np.max(allLastCosts)
    meanAllLastCosts = np.mean(allLastCosts)
    print(allLastCosts, "\n", minAllLastCosts, "\n", maxAllLastCosts, "\n", meanAllLastCosts)


def writefile(filename):
    f = open(filename, "w")
    f.write(filename)
    f.write("\n")
    f.write(f"Sub pop sayisi= {size}\n")
    f.write(f"Problem boyutu= {nVar}\n")
    f.write(f"Tekrar sayisi= {repeat}\n")
    f.write("\n")
    f.write(f"Min value=  {minAllLastCosts}\n")
    f.write(f"Max value=  {maxAllLastCosts}\n")
    f.write(f"Mean value= {meanAllLastCosts}\n")
    f.write("\n\n")
    f.write("Her tekrarın sonucu")
    f.write("\n")
    for i in range(repeat):
        f.write(str(allLastCosts[i]))
        f.write("\n")
    f.write("\n\n")
    f.write("En iyi sonucun itersayon adımları")
    f.write("\n")
    for i in range(maxIt):
        f.write(str(meanCosts[i]))
        f.write("\n")
    f.close()


if rank == 0:
    writefile(fileName)
