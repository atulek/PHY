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
repeat = 5
maxIt = 500
nPop = 100
nVar = 50

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

# bestcost = np.empty(repeat, float)
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
        # meanCosts = np.mean(bestcost, float)
if rank == 0:
    meanCosts = np.mean(allCosts, axis=0)
    print(allLastCosts)
    print(meanCosts)




# bestcost = pso.run()
# result = de.run()
# print(gbest['cost'])


# plt.semilogy(bestcost)
# plt.xlim(0, 1000)
# plt.xlabel('İterasyonlar')
# plt.ylabel('Maliyet Fonksiyonu')
# plt.title('Particle Swarm Optimization (PSO)')
# plt.grid(True)
# plt.show()

# PSO için maliyet fonksiyonu GA olarak atanıyor
# PSO.CostFunction = gao
# GA parametre aralıkları tanımlanıyor. Sırasıyla:
# pc, gamma, mu, sigma, selectionType
# PSO parametreleri belirleniyor
# c1, c2 [0,5 - 2,5] w = [0,4 -0,9]
