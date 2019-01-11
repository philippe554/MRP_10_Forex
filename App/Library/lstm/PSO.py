import numpy as np


class PSO:
    def __init__(self, amount_of_particles, dims):
        self.amount_of_particles = amount_of_particles
        self.dims = dims

        # TODO: properly select random distributions and initial cost
        self.pos = np.random.rand(self.amount_of_particles, self.dims) * 2 - 1
        self.vel = np.random.rand(self.amount_of_particles, self.dims) * 2 - 1
        self.cost = np.full(self.amount_of_particles, 1000.0)

        # self.load_info()
        self.best_pos = self.pos
        self.best_cost = np.full(self.amount_of_particles, 1000.0)
        self.best_swarm_pos = self.pos[0]  # TODO: should not be the first
        self.best_swarm_cost = 1000.0
        self.best_swarm_index = 0
        self.rogueParticle = -1

        # Just some random hyper parameters i found somewhere
        self.omega = 0.5
        self.phiP = 0.6
        self.phiG = 0.8

    def get_particles(self):
        # assert list(np.shape(self.pos)) == [self.amount_of_particles, self.dims]
        return self.pos

    def get_best_particle(self):
        return self.best_swarm_pos

    def update(self, cost):
        # assert len(cost) == self.amount_of_particles

        rp = np.random.rand(self.amount_of_particles, self.dims)
        rg = np.random.rand(self.amount_of_particles, self.dims)
        self.vel = self.omega * self.vel + self.phiP * rp * (self.best_pos - self.pos) + self.phiG * rg * (
                self.best_swarm_pos - self.pos)
        self.pos = self.pos + self.vel

        self.update_bests(cost)

        self.rogueParticle = np.random.randint(self.amount_of_particles)
        while self.rogueParticle ==  self.best_swarm_index:
            self.rogueParticle = np.random.randint(self.amount_of_particles)

        posStd = np.std(self.pos)
        self.pos[self.rogueParticle] = (np.random.rand(1, self.dims) * 2 - 1) * (posStd * 3)

    def update_bests(self, cost):
        #take the moment of the cost to smooth out outliers (particles need to build up credibility)
        self.cost = self.cost * 0.8 + cost * 0.2

        #self.cost = cost # uncomment to use old method

        for i in range(self.amount_of_particles):
            if self.cost[i] < self.best_cost[i]:
                self.best_pos[i, :] = self.pos[i, :]
                self.best_cost[i] = self.cost[i]

                if self.cost[i] < self.best_swarm_cost:
                    self.best_swarm_pos = self.pos[i, :]
                    self.best_swarm_cost = self.cost[i]
                    self.best_swarm_index = i

    def getStats(self):
        stats = {}
        stats["best"] = self.best_swarm_index
        stats["rogue"] = self.rogueParticle
        stats["avgPos"] = "%.5f" % np.mean(self.pos)
        stats["varPos"] = "%.5f" % np.mean(np.power(self.pos - np.mean(self.pos, axis=1, keepdims=True), 2))
        stats["avgBestDistance"] = "%.3f" % np.mean(np.sqrt(np.sum(np.power(self.pos - self.best_pos, 2), axis=1)))
        stats["avgSwarmBestDistance"] = "%.3f" % np.mean(np.sqrt(np.sum(np.power(self.pos - self.best_swarm_pos, 2), axis=1)))
        stats["avgVelocity"] = "%.3f" % np.mean(np.sqrt(np.sum(np.power(self.vel, 2), axis=1)))
        return stats
