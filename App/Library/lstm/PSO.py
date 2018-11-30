import numpy as np


class PSO:
    def __init__(self, amountOfParticles, dims):
        self.amountOfParticles = amountOfParticles
        self.dims = dims

        # TODO: properly select random distributions and initial cost
        self.pos = np.random.rand(self.amountOfParticles, self.dims)
        self.vel = np.random.rand(self.amountOfParticles, self.dims)

        self.bestPos = self.pos
        self.bestCost = np.full(self.amountOfParticles, 1000)

        self.bestSwarmPos = self.pos[0] # TODO: should not be the first
        self.bestSwarmCost = 1000

        # Just some random hyper parameters i found somewhere
        self.omega = 0.7
        self.phiP = 2
        self.phiG = 2

    def getParticles(self):
        assert list(np.shape(self.pos)) == [self.amountOfParticles, self.dims]
        return self.pos

    def update(self, cost):
        assert len(cost) == self.amountOfParticles

        rp = np.random.rand(self.amountOfParticles, self.dims)
        rg = np.random.rand(self.amountOfParticles, self.dims)
        self.vel = self.omega * self.vel + self.phiP * rp * (self.bestPos - self.pos) + self.phiG * rg * (self.bestSwarmPos - self.pos)
        self.pos = self.pos + self.vel

        for i in range(self.amountOfParticles):
            if cost[i] < self.bestCost[i]:
                self.bestPos[p, :] = self.pos[p, :]
                self.bestCost[i] = cost[i]

                if cost[i] < self.bestSwarmCost:
                    self.bestSwarmPos = self.pos[p, :]
                    self.bestSwarmCost = cost[i]