import numpy as np
import copy
from mpi4py import MPI
from TestFunc import TestFunc

# ABC sınıfı oluşturuluyor


class ABC:

    # Default parametreler
    def __init__(self, test_func, maxit, npop, nvar, comm, psize, myrank, mrate, minterval):

        self.costfunc = test_func.costfunc
        self.varmin = test_func.varmin
        self.varmax = test_func.varmax
        self.maxit = maxit
        self.npop = npop
        self.nvar = nvar
        self.comm = comm
        self.psize = psize
        self.myrank = myrank
        self.mrate = mrate
        self.minterval = minterval
        self.nonlookerpop = npop
        self.L = self.npop * self.nvar  # failure limit
        self.a = 1

    # ABC çalıştırılıyor
    def run(self):

        # Boş bireyler oluşturuluyor
        # empty_bee = {}
        # empty_bee["position"] = None
        # empty_bee["cost"] = None

        empty_bee = {}
        empty_bee["position"] = None
        empty_bee["cost"] = None
        empty_bee["velocity"] = None
        empty_bee["best_position"] = None
        empty_bee["best_cost"] = None

        # En iyi çözümü tutan dictionary
        bestsol = empty_bee
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

        # Failure counter
        fc = np.zeros(self.npop, int)

        # Recruited Bees
        for it in range(self.maxit):
            for i in range(self.npop):
                # Choose k randomly, not equal to i
                K = np.random.permutation(self.npop)
                X = np.where(K == i)
                K = np.delete(K, X)
                k = K[0]
                # Define Acceleration Coefficient
                phi = np.random.uniform(-1, 1, self.nvar)
                newbee = {}
                # New Bee Position
                newbee["position"] = pop[i]["position"] + phi * (pop[i]["position"] - pop[k]["position"])
                # Apply Bounds
                newbee["position"] = self.apply_bound(newbee["position"], self.varmin, self.varmax)
                # Evaluation
                newbee["cost"] = self.costfunc(newbee["position"])
                # Comparision
                if newbee["cost"] <= pop[i]["cost"]:
                    pop[i] = newbee.copy()
                else:
                    fc[i] += 1
            # Calculate Fitness Values and Selection Probabilities
            F = np.zeros(self.npop)
            totalcost = 0
            for i in range(self.npop):
                totalcost += pop[i]["cost"]
            meancost = totalcost / self.npop
            for i in range(self.npop):
                F[i] = np.exp(-pop[i]["cost"]/meancost)
            P = F / np.sum(F)

            # Onlooker bees
            for m in range(self.nonlookerpop):
                # Select Source Site
                i = self.roulette_wheel_selection(P)
                # Choose k randomly, not equal to i
                K = np.random.permutation(self.npop)
                X = np.where(K == i)
                K = np.delete(K, X)
                k = K[0]
                # Define Acceleration Coefficient
                phi = np.random.uniform(-1, 1, self.nvar)
                newbee = {}
                # New Bee Position
                newbee["position"] = pop[i]["position"] + phi * (pop[i]["position"] - pop[k]["position"])
                # Apply Bounds
                newbee["position"] = self.apply_bound(newbee["position"], self.varmin, self.varmax)
                # Evaluation
                newbee["cost"] = self.costfunc(newbee["position"])
                # Comparision
                if newbee["cost"] <= pop[i]["cost"]:
                    pop[i] = newbee.copy()
                else:
                    fc[i] += 1

            # Scout Bees
            for i in range(self.npop):
                if fc[i] >= self.L:
                    pop[i]["position"] = np.random.uniform(self.varmin, self.varmax, self.nvar)
                    pop[i]["cost"] = self.costfunc(pop[i]["position"])
                    fc[i] = 0
            # Update Best Solution Ever Found
            for i in range(self.npop):
                if pop[i]["cost"] < bestsol["cost"]:
                    bestsol = pop[i].copy()
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
        # print("ABC=", bestcost[-1])
        # print(self.myrank)
        # print("ABC--=", gBests[-1])

        # Elde edilen çıktılar döndürülüyor
        # out = {}
        # out["pop"] = pop
        # out["bestsol"] = bestsol
        # out["bestcost"] = bestcost
        # out["gbests"] = gBests
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

    def roulette_wheel_selection(self, p):
        c = np.cumsum(p)
        r = sum(p)*np.random.rand()
        ind = np.argwhere(r <= c)
        return ind[0][0]

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
            if self.myrank == (i + 1) % self.psize:
                kaynak = (self.myrank - 1) % self.psize
                if kaynak < 0:
                    kaynak += self.psize
                tmp = np.empty(pop.shape, dtype=np.float)
                self.comm.Recv(tmp, source=kaynak, tag=itr)
                received = tmp.copy()
        return received
