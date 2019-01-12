import numpy as np
import subprocess
from datetime import datetime as dt

class PSO:
    def __init__(self, forexType):
        # Version info
        try:
            self.revision = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('ascii')
            self.branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip().decode('ascii')
        except Exception:
            self.revision = ""
            self.branch = ""
        self.date = dt.now().strftime("%Y-%m-%d %H:%M:%S")
        self.forexType = forexType

        self.l1Size = 4
        self.l2Size = 8
        self.lstmSize = 6
        self.outputSize = 2
        self.sequenceSize = 30
        self.sequenceOverlap = 120
        self.batchSize = 100
        self.amountOfParticles = 100
        self.amountOfEpochs = 100

        self.omega = 0.7
        self.phiP = 0.8
        self.phiG = 0.6

    def print_hyper_parameters(self):
        print('\033[94m'+"====================")
        print("initialized =", self.date)
        print("branch =", self.branch, "@", self.revision)
        print("forexType =", self.forexType)
        print("l1Size =", self.l1Size, "/ l2Size =", self.l2Size, "/ lstmSize =", self.lstmSize, "/ outSize =", self.outputSize)
        print("sequenceSize =", self.sequenceSize, "/ sequenceOverlap =", self.sequenceOverlap)
        print("batchSize =", self.batchSize, "/ epochs =", self.amountOfEpochs)
        print("amountParticles =",self.amountOfParticles, "/ omega =", self.omega, "/ phiP =", self.phiP, " / phiG =", self.phiG)
        print("====================" + '\033[0m')

    def reset_particles(self, dims):
        print("Reset PSO")

        self.dims = dims

        self.pos = np.random.rand(self.amountOfParticles, self.dims) * 2 - 1
        self.vel = np.random.rand(self.amountOfParticles, self.dims) * 2 - 1
        self.cost = np.full(self.amountOfParticles, 10000.0)

        # self.load_info()
        self.best_pos = self.pos
        self.best_cost = np.full(self.amountOfParticles, 10000.0)
        self.best_swarm_pos = None
        self.best_swarm_cost = None
        self.best_swarm_index = None
        self.rogueParticle = -1

    def get_particles(self):
        # assert list(np.shape(self.pos)) == [self.amount_of_particles, self.dims]
        return self.pos

    def get_best_particle(self):
        return self.best_swarm_pos

    def update(self, cost):
        # assert len(cost) == self.amount_of_particles

        self.update_bests(cost)

        rp = np.random.rand(self.amountOfParticles, self.dims)
        rg = np.random.rand(self.amountOfParticles, self.dims)
        self.vel = self.omega * self.vel + self.phiP * rp * (self.best_pos - self.pos) + self.phiG * rg * (
                self.best_swarm_pos - self.pos)
        self.pos = self.pos + self.vel

        self.rogueParticle = np.random.randint(self.amountOfParticles)
        while self.rogueParticle == self.best_swarm_index:
            self.rogueParticle = np.random.randint(self.amountOfParticles)

        posStd = np.std(self.pos)
        self.pos[self.rogueParticle] = (np.random.rand(1, self.dims) * 2 - 1) * (posStd * 3)

    def update_bests(self, cost):
        # take the moment of the cost to smooth out outliers (particles need to build up credibility)
        # self.cost = self.cost * 0.8 + cost * 0.2
        self.cost = cost # uncomment to use old method

        for i in range(self.amountOfParticles):
            if self.cost[i] < self.best_cost[i]:
                self.best_pos[i, :] = self.pos[i, :]
                self.best_cost[i] = self.cost[i]

                if self.best_swarm_cost is None or self.cost[i] < self.best_swarm_cost:
                    self.best_swarm_pos = self.pos[i, :]
                    self.best_swarm_cost = self.cost[i]
                    self.best_swarm_index = i

    def getStats(self):
        stats = {}
        stats["best"] = self.best_swarm_index
        stats["bestCost"] = "%.3f" % self.best_swarm_cost
        stats["bestCurrentCost"] = "%.3f" % self.cost[self.best_swarm_index]  # How much profit did the best particle make in the current iteration
        stats["rogue"] = self.rogueParticle
        stats["avgPos"] = "%.5f" % np.mean(self.pos)
        stats["varPos"] = "%.5f" % np.mean(np.power(self.pos - np.mean(self.pos, axis=1, keepdims=True), 2))
        stats["avgBestDistance"] = "%.3f" % np.mean(np.sqrt(np.sum(np.power(self.pos - self.best_pos, 2), axis=1)))
        stats["avgSwarmBestDistance"] = "%.3f" % np.mean(np.sqrt(np.sum(np.power(self.pos - self.best_swarm_pos, 2), axis=1)))
        stats["avgVelocity"] = "%.3f" % np.mean(np.sqrt(np.sum(np.power(self.vel, 2), axis=1)))
        return stats
