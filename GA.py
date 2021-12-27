import numpy as np
import copy
from mpi4py import MPI
from TestFunc import TestFunc


class GA:
    # Default parametreler
    def __init__(self, test_func, maxit, npop, nvar, beta, pc, gamma, mu, sigma, k, comm, psize, myrank, mrate, minterval):

        self.costfunc = test_func.costfunc
        self.varmin = test_func.varmin
        self.varmax = test_func.varmax
        self.selectionType = 1

        self.maxit = maxit
        self.npop = npop
        self.nvar = nvar
        self.beta = beta
        self.pc = pc
        self.gamma = gamma
        self.mu = mu
        self.sigma = sigma
        self.nc = int(np.round(self.pc*self.npop/2)*2)
        self.k = k
        self.comm = comm
        self.psize = psize
        self.myrank = myrank
        self.mrate = mrate
        self.minterval = minterval

    # GA çalıştırılıyor
    def run(self):
        # Boş bireyler oluşturuluyor
        individual = {}
        individual["position"] = None
        individual["cost"] = None


        # En iyi çözümü tutan dictionary
        bestsol = individual
        bestsol["cost"] = np.inf
        # İlk popülasyon rasgele oluşturuluyor
        pop = [0] * self.npop

        for i in range(self.npop):
            pop[i] = {}
            pop[i]["position"] = np.random.uniform(self.varmin, self.varmax, self.nvar)
            pop[i]["cost"] = self.costfunc(pop[i]["position"])
            # En iyi çözüm (gerekliyse) güncelleniyor
            if pop[i]["cost"] < bestsol["cost"]:
                bestsol = copy.deepcopy(pop[i])

        # maxit boyutunda dizi tanımlanıyor (en iyi çözümlerin sonuçları tutulacak)
        bestcost = np.empty(self.maxit)
        gBests = np.empty(self.maxit)


        # Algoritma çalışmaya başlıyor
        for it in range(self.maxit):
            costs = [0] *len(pop)
            #Rulet tekerleği
            for t in range(len(pop)):
                costs[t] = pop[t]["cost"]

            costs = np.array(costs)
            #costs = np.array([x["cost"] for x in pop])
            avg_cost = np.mean(costs)
            if avg_cost != 0:
                costs = costs / avg_cost
            probs = np.exp(-self.beta * costs)
            # Selection işlemi
            popc = []
            for _ in range(self.nc // 2):
                if self.selectionType == 0:   # Rasgele seçim
                    q = np.random.permutation(self.npop)
                    p1 = pop[q[0]]
                    p2 = pop[q[1]]
                elif self.selectionType == 1:   # Rulet tekerleği
                    p1 = pop[self.roulette_wheel_selection(probs)]
                    p2 = pop[self.roulette_wheel_selection(probs)]
                elif self.selectionType == 2:   # Turnuva seçimi
                    p1 = self.tournament_selection(pop, self.k)
                    p2 = self.tournament_selection(pop, self.k)

                # Crossover İşlemi
                c1, c2 = self.crossover(p1, p2, self.gamma)

                # Mutation işlemi
                c1 = self.mutate(c1, self.mu, self.sigma)
                c2 = self.mutate(c2, self.mu, self.sigma)

                # Çözüm uzayının dışına çıkıldığında sınırlara çekme işlemi
                self.apply_bound(c1, self.varmin, self.varmax)
                self.apply_bound(c2, self.varmin, self.varmax)

                # İlk çocuk problemi çözüyor, gerekliyse en iyi çözüm güncelleniyor
                c1["cost"] = self.costfunc(c1["position"])
                if c1["cost"] < bestsol["cost"]:
                    bestsol = copy.deepcopy(c1)

                # İkinci çocuk problemi çözüyor, gerekliyse en iyi çözüm güncelleniyor
                c2["cost"] = self.costfunc(c2["position"])
                if c2["cost"] < bestsol["cost"]:
                    bestsol = copy.deepcopy(c2)


                # Yeni çocuklar popülasyona dahil ediliyor
                popc.append(c1)
                popc.append(c2)

            # yeni oluşan büyük poplasyon maliyete göre sıralanıp popülasyon boyutundan fazla olan bireyler atılıyor
            pop += popc
            pop = sorted(pop, key=lambda x: x["cost"])
            pop = pop[0:self.npop]

            # İterasyonun en iyi çözümü depolanıyor
            bestcost[it] = bestsol["cost"]
            if it % self.minterval == 0 and it != 0 and it != self.maxit:
                pop = sorted(pop, key=lambda s: s["cost"])
                popArr = self.dicttonp(pop)
                # received = np.empty(popArr.shape, dtype=np.float)
                received = self.migrate(popArr, it)
                popArr[-self.mrate:] = received[:self.mrate].copy()
                pop = self.nptodict(popArr, pop)
            # print(it, bestcost[it])
            if self.myrank == 0:
                gBest = np.empty(1, float)
            else:
                gBest = None
            lBest = bestcost[it]
            self.comm.Reduce([lBest, MPI.FLOAT], [gBest, MPI.FLOAT], op=MPI.MIN, root=0)
            gBests[it] = gBest
        # print("GA=", bestcost[-1])
        # print(self.myrank)
        # print("GA---=", gBests[-1])

        # Elde edilen çıktılar döndürülüyor
        # out = {}
        # out["pop"] = pop
        # out["bestsol"] = bestsol
        # out["bestcost"] = bestcost
        # out["bests"] = gBests
        # return out
        return gBests
        # Çaprazlam işlemi raporda anlatıldığı gibi uniform olarak yapılıyor
    def crossover(self, p1, p2, gamma=0.1):
        c1 = copy.deepcopy(p1)
        c2 = copy.deepcopy(p1)
        alpha = np.random.uniform(-gamma, 1 + gamma, *c1["position"].shape)
        c1["position"] = alpha * p1["position"] + (1 - alpha) * p2["position"]
        c2["position"] = alpha * p2["position"] + (1 - alpha) * p1["position"]
        return c1, c2
        # Mutasyon oranına göre rasgele seçilen bireylere mutasyon uygulanıyor
    def mutate(self, x, mu, sigma):
        y = copy.deepcopy(x)
        flag = np.random.rand(*x["position"].shape) <= mu
        ind = np.argwhere(flag)
        y["position"][ind] += sigma * np.random.randn(*ind.shape)
        return y
        # Çözümleri problem uzayında tutan metot
    def apply_bound(self, x, varmin, varmax):
        x["position"] = np.maximum(x["position"], varmin)
        x["position"] = np.minimum(x["position"], varmax)
        # Rulet tekerleği seçimi
    def roulette_wheel_selection(self, p):
        c = np.cumsum(p)
        r = sum(p) * np.random.rand()
        ind = np.argwhere(r <= c)
        return ind[0][0]

    def tournament_selection(self, pop, k):
        best = np.inf
        q = np.random.permutation(self.npop)
        # print(pop[q[0]]["cost"])
        for i in range(k):
            if pop[q[i]]["cost"] < best:
                best = pop[q[i]]["cost"]
                ret = pop[q[i]]
        return ret

    def dicttonp(self, pop):
        ret = np.empty((self.npop, self.nvar))
        for i in range(self.npop):
            ret[i] = pop[i]["position"].copy()
        return ret

    def nptodict(self, arr, ret):
        for i in range(self.npop):
            ret[i]["position"] = arr[i]
        return ret

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
