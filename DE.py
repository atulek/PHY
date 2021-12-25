import numpy as np
import copy
from mpi4py import MPI
from TestFunc import TestFunc

# Git deneme
# DE sınıfı oluşturuluyor


class DE:

    # Default parametreler
    def __init__(self, test_func, maxit, npop, nvar, F, cp, comm, psize, myrank, mrate, minterval):

        self.costfunc = test_func.costfunc
        self.nvar = nvar
        self.varmin = test_func.varmin
        self.varmax = test_func.varmax
        self.maxit = maxit
        self.npop = npop
        self.F = F
        self.cp = cp
        self.comm = comm
        self.psize = psize
        self.myrank = myrank
        self.mrate = mrate
        self.minterval = minterval

    # DE çalıştırılıyor
    def run(self):

        # Boş bireyler oluşturuluyor
        individual = {}
        individual["position"] = None
        individual["cost"] = None

        # individual = {}
        # individual["position"] = None
        # individual["cost"] = None
        # individual["velocity"] = None
        # individual["best_position"] = None
        # individual["best_cost"] = None

        # En iyi çözümü tutan dictionary
        bestsol = individual
        bestsol["cost"] = np.inf
        # İlk popülasyon rasgele oluşturuluyor
        pop = [0] * self.npop

        for i in range(self.npop):
            pop[i] = {}
            pop[i]["position"] = np.random.uniform(self.varmin, self.varmax, self.nvar)
            pop[i]["cost"] = self.costfunc(pop[i]["position"])
            pop[i]["velocity"] = np.zeros(self.nvar)
            pop[i]["best_position"] = pop[i]["position"].copy()
            # En iyi çözüm (gerekliyse) güncelleniyor
            if pop[i]["cost"] < bestsol["cost"]:
                bestsol = copy.deepcopy(pop[i])

        # maxit boyutunda dizi tanımlanıyor (en iyi çözümlerin sonuçları tutulacak)
        bestcost = np.empty(self.maxit)
        gBests = np.empty(self.maxit)

        # Algoritma çalışmaya başlıyor
        for it in range(self.maxit):
            for i in range(self.npop):
                # Mutasyon işlemi için 3 adet birey seçiliyor

                x = pop[i]["position"]
                K = np.random.permutation(self.npop)
                X = np.where(K == i)
                K = np.delete(K, X)
                a = K[1]
                b = K[2]
                c = K[3]
                # Mutasyon işlemi

                y = pop[a]["position"] + self.F * (pop[b]["position"] - pop[c]["position"])
                y = self.apply_bound(y, self.varmin, self.varmax)
                # Rekombinasyon işlemi

                z = np.zeros(len(x))
                j0 = np.random.randint(len(x))
                for j in range(len(x)):
                    if j0 == j or np.random.uniform(0, 1) <= self.cp:
                        z[j] = y[j]
                    else:
                        z[j] = x[j]

                newSol = {}
                newSol["position"] = z
                newSol["cost"] = self.costfunc(newSol["position"])

                if newSol["cost"] < pop[i]["cost"]:
                    pop[i] = newSol
                    if pop[i]["cost"] < bestsol["cost"]:
                        bestsol = pop[i]

            bestcost[it] = bestsol["cost"]
            # Migration
            # received = np.empty(popArr.shape, dtype=np.float)
            if it % self.minterval == 0 and it != 0 and it != self.maxit:
                pop = sorted(pop, key=lambda s: s["cost"])
                popArr = self.dicttonp(pop)
                # received = np.empty(popArr.shape, dtype=np.float)
                received = self.migrate(popArr, it)
                popArr[-self.mrate:] = received[:self.mrate].copy()
                pop = self.nptodict(popArr, pop)
            # print(it, "cost", bestcost[it])
            if self.myrank == 0:
                gBest = np.empty(1, float)
            else:
                gBest = None
            lBest = bestcost[it]
            self.comm.Reduce([lBest, MPI.FLOAT], [gBest, MPI.FLOAT], op=MPI.MIN, root=0)
            gBests[it] = gBest
        print("DE=", bestcost[-1])
        print(self.myrank)
        print("DE---=", gBests[-1])

        # Elde edilen çıktılar döndürülüyor
        # out = {}
        # out["pop"] = pop
        # out["bestsol"] = bestsol
        # out["bestcost"] = bestcost
        # out["bests"] = gBests
        # return out
        return gBests

        # Çözümleri problem uzayında tutan metot

    def apply_bound(self, x, varmin, varmax):
        min = np.zeros(self.nvar)
        min.fill(varmin)
        max = np.zeros(self.nvar)
        max.fill(varmax)
        x = np.maximum(x, min)
        x = np.minimum(x, max)
        return x

    def dicttonp(self, pop):
        ret = np.empty((self.npop, self.nvar))
        for i in range(self.npop):
            ret[i] = pop[i]["position"].copy()
        return ret

    def nptodict(self, arr, ret):
        for i in range(self.npop):
            ret[i]["position"] = arr[i]
        return ret

    # def dicttonp(self, pop):
    #     ret = np.empty((3, self.npop, self.nvar))
    #     # ret = np.empty((self.npop, self.nvar))
    #
    #     for i in range(self.npop):
    #         ret[0][i] = pop[i]["position"].copy()
    #         ret[1][i] = pop[i]["best_position"].copy()
    #         ret[2][i] = pop[i]["velocity"].copy()
    #     return ret
    #
    # def nptodict(self, arr, ret):
    #     for i in range(self.npop):
    #         ret[i]["position"] = arr[0][i]
    #         ret[i]["best_position"] = arr[1][i]
    #         ret[i]["velocity"] = arr[2][i]
    #     return ret

    def migrate(self, pop, itr):
        received = np.empty(pop.shape, dtype=np.float)
        for i in range(self.psize):
            if self.myrank == i:
                hedef = (self.myrank + 1) % self.psize
                self.comm.Send(pop, dest=hedef, tag=itr)
            if self.myrank == (i+1) % self.psize:
                kaynak = (self.myrank-1) % self.psize
                if kaynak < 0:
                    kaynak += self.psize
                tmp = np.empty(pop.shape, dtype=np.float)
                self.comm.Recv(tmp, source=kaynak, tag=itr)
                received = tmp.copy()
        return received

