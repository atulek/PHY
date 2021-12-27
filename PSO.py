import numpy as np
from mpi4py import MPI
from TestFunc import TestFunc

class PSO:
	def __init__(self, test_func, maxit, npop, nvar, c1, c2, w, wdamp, comm, psize, myrank, mrate, minterval):
	
		self.costfunc = test_func.costfunc
		self.nvar = nvar
		self.varmin = test_func.varmin
		self.varmax = test_func.varmax
		self.maxit = maxit
		self.npop = npop
		self.c1 = c1
		self.c2 = c2
		self.w = w
		self.wdamp = wdamp
		self.comm = comm
		self.psize = psize
		self.myrank = myrank
		self.mrate = mrate
		self.minterval = minterval

	def run(self):

		# Empty Particle Template
		particle = {}
		particle["position"] = None
		particle["cost"] = None
		particle["velocity"] = None
		particle["best_position"] = None
		particle["best_cost"] = None
		# empty_particle = {
		# 	'position': None,
		# 	'velocity': None,
		# 	'cost': None,
		# 	'best_position': None,
		# 	'best_cost': None,
		# }

		# Extract Problem Info
		# CostFunction = problem['CostFunction']
		# VarMin = problem['VarMin']
		# VarMax = problem['VarMax']
		# nVar = problem['nVar']

		# Initialize Global Best
		# gbest = {'position': None, 'cost': np.inf}
		gbest = particle
		gbest["cost"] = np.inf

		# Create Initial Population
		# pop = []
		pop = [0] * self.npop
		for i in range(0, self.npop):
			pop[i] = {}
			pop[i]["position"] = np.random.uniform(self.varmin, self.varmax, self.nvar)
			pop[i]["velocity"] = np.zeros(self.nvar)
			pop[i]["cost"] = self.costfunc(pop[i]["position"])
			pop[i]["best_position"] = pop[i]["position"].copy()
			pop[i]["best_cost"] = pop[i]["cost"]

			if pop[i]["best_cost"] < gbest["cost"]:
				gbest["position"] = pop[i]["best_position"].copy()
				gbest["cost"] = pop[i]["best_cost"]

		bestcost = np.empty(self.maxit)
		gBests = np.empty(self.maxit)

		# PSO Loop
		for it in range(0, self.maxit):
			for i in range(0, self.npop):

				pop[i]["velocity"] = self.w * pop[i]["velocity"] \
									 + self.c1 * np.random.rand(self.nvar) * (pop[i]["best_position"] - pop[i]["position"]) \
									 + self.c2 * np.random.rand(self.nvar) * (gbest["position"] - pop[i]["position"])

				pop[i]["position"] += pop[i]["velocity"]
				pop[i]["position"] = np.maximum(pop[i]["position"], self.varmin)
				pop[i]["position"] = np.minimum(pop[i]["position"], self.varmax)

				pop[i]["cost"] = self.costfunc(pop[i]["position"])

				if pop[i]["cost"] < pop[i]["best_cost"]:
					pop[i]["best_position"] = pop[i]["position"].copy()
					pop[i]["best_cost"] = pop[i]["cost"]

					if pop[i]["best_cost"] < gbest["cost"]:
						gbest["position"] = pop[i]["best_position"].copy()
						gbest["cost"] = pop[i]["best_cost"]

			self.w *= self.wdamp
			# print('Iteration {}: Best Cost = {}'.format(it, gbest['cost']))
			bestcost[it] = gbest["cost"]
			# Migration
			# received = np.empty(popArr.shape, dtype=np.float)
			if it % self.minterval == 0 and it != 0 and it != self.maxit:
				pop = sorted(pop, key=lambda s: s["cost"])
				popArr = self.dicttonp(pop)
				# received = np.empty(popArr.shape, dtype=np.float)
				received = self.migrate(popArr, it)
				popArr[-self.mrate:] = received[:self.mrate].copy()
				pop = self.nptodict(popArr, pop)
			if self.myrank == 0:
				gBest = np.empty(1, float)
			else:
				gBest = None
			lBest = bestcost[it]
			self.comm.Reduce([lBest, MPI.FLOAT], [gBest, MPI.FLOAT], op=MPI.MIN, root=0)
			gBests[it] = gBest
		# print("PSO=", bestcost[-1])
		# print(self.myrank)
		# print("PSO--=", gBests[-1])
		# Elde edilen çıktılar döndürülüyor
		# out = {}
		# out["pop"] = pop
		# out["bestcost"] = bestcost
		# out["bests"] = gBests
		# return out
		return gBests

	# def dicttonp(self, pop):
	# 	ret = np.empty((3, self.npop, self.nvar))
	# 	# ret = np.empty((self.npop, self.nvar))
	#
	# 	for i in range(self.npop):
	# 		ret[0][i] = pop[i]["position"].copy()
	# 		ret[1][i] = pop[i]["best_position"].copy()
	# 		ret[2][i] = pop[i]["velocity"].copy()
	# 	return ret
	#
	# def nptodict(self, arr, ret):
	# 	for i in range(self.npop):
	# 		ret[i]["position"] = arr[0][i]
	# 		ret[i]["best_position"] = arr[1][i]
	# 		ret[i]["velocity"] = arr[2][i]
	# 	return ret

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
			if self.myrank == (i + 1) % self.psize:
				kaynak = (self.myrank - 1) % self.psize
				if kaynak < 0:
					kaynak += self.psize
				tmp = np.empty(pop.shape, dtype=np.float)
				self.comm.Recv(tmp, source=kaynak, tag=itr)
				received = tmp.copy()
		return received

# Start Time for tic and tov functions
# startTime_for_tictoc = 0
#
# # Start measuring time elapsed
# def tic():
# 	import time
# 	global startTime_for_tictoc
# 	startTime_for_tictoc = time.time()
#
# # End mesuring time elapsed
# def toc():
# 	import time, math
# 	if 'startTime_for_tictoc' in globals():
# 		dt = math.floor(100*(time.time() - startTime_for_tictoc))/100.
# 		print('Elapsed time is {} second(s).'.format(dt))
# 	else:
# 		print('Start time not set. You should call tic before toc.')
