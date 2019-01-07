import numpy as np


class PSO:
    def __init__(self, amount_of_particles, dims):
        self.amount_of_particles = amount_of_particles
        self.dims = dims

        # TODO: properly select random distributions and initial cost
        self.pos = np.random.rand(self.amount_of_particles, self.dims)
        self.vel = np.random.rand(self.amount_of_particles, self.dims)
        # self.load_info()
        self.best_pos = self.pos
        self.best_cost = np.full(self.amount_of_particles, 1000)
        self.best_swarm_pos = self.pos[0]  # TODO: should not be the first
        self.best_swarm_cost = 1000

        # Just some random hyper parameters i found somewhere
        self.omega = 0.7
        self.phiP = 2
        self.phiG = 2

    def get_particles(self):
        assert list(np.shape(self.pos)) == [self.amount_of_particles, self.dims]
        return self.pos

    def update(self, cost):
        assert len(cost) == self.amount_of_particles

        rp = np.random.rand(self.amount_of_particles, self.dims)
        rg = np.random.rand(self.amount_of_particles, self.dims)
        self.vel = self.omega * self.vel + self.phiP * rp * (self.best_pos - self.pos) + self.phiG * rg * (
                self.best_swarm_pos - self.pos)
        self.pos = self.pos + self.vel

        self.update_bests(cost)

    def update_bests(self, cost):
        for i in range(self.amount_of_particles):
            if cost[i] < self.best_cost[i]:
                self.best_pos[i, :] = self.pos[i, :]
                self.best_cost[i] = cost[i]

                if cost[i] < self.best_swarm_cost:
                    self.best_swarm_pos = self.pos[i, :]
                    self.best_swarm_cost = cost[i]

    def getStats(self):
        stats = {}
        stats["avgBestDistance"] = np.mean(np.sqrt(np.sum(np.power(self.pos - self.best_pos, 2), axis=1)))
        stats["avgSwarmBestDistance"] = np.mean(np.sqrt(np.sum(np.power(self.pos - self.best_swarm_cost, 2), axis=1)))
        stats["avgVelocity"] = np.mean(np.sqrt(np.sum(np.power(self.vel, 2), axis=1)))
        return stats
