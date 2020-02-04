"""
Implementation of continuous state fluid neural network from
Information at the edge of chaos in fluid neural networks, Sole & Miramontes
with open boundaries inspired by
Self-organized criticality in ant brood tending, O'Toole et al.
"""

import random
import math
import numpy as np
import matplotlib.pyplot as plt


class Grid:
    def __init__(self, size, num_agents):
        self.size = size
        self.grid = -np.ones((size, size))
        self.num_agents = num_agents
        self.next_agent_idx = num_agents + 1
        self.ant_dict = {}
        self.empty_idx = -1
        self._populate()

    def _clear_boundary(self):
        boundary_sites = boundary(self.size)
        for site in boundary_sites:
            idx = self.grid[site]
            if idx in self.ant_dict:
                del self.ant_dict[idx]
            self.grid[site] = self.empty_idx

    def _populate(self):
        for idx in range(self.num_agents):
            open_squares_x, open_squares_y = np.where(self.grid == self.empty_idx)
            location_index = random.randint(0, len(open_squares_x)-1)
            x_location = open_squares_x[location_index]
            y_location = open_squares_y[location_index]
            new_ant = Ant(x_location, y_location, idx)
            self.ant_dict[idx] = new_ant
            self.grid[x_location, y_location] = idx
        self._clear_boundary()

    def _von_neumann_neighbors(self, x, y):
        """
        Returns active ants in a Von Neumann (4 nearest) neighborhood for a given location
        """
        options = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        neighbors = []
        for opt in options:
            i, j = opt
            if 0 <= i < self.size:
                if 0 <= j < self.size:
                    # if there is an ant at this location
                    if self.grid[i, j] > self.empty_idx:
                        neighbors.append(self.grid[i, j])
        return neighbors

    def _moore_neighbors(self, x, y):
        """
        Returns the indices of (up to) 8 neighbors (Moore neighborhood).
        """
        x_options = [x - 1, x, x + 1]
        y_options = [y - 1, y, y + 1]
        valid_x = []
        valid_y = []
        valid_neighbors = []
        for i in x_options:
            for j in y_options:
                if 0 <= i < self.size:
                    if 0 <= j < self.size:
                        if i != x or j != y:
                            # if there is an ant at this location
                            if self.grid[i, j] > self.empty_idx:
                                valid_x.append(i)
                                valid_y.append(j)
                                valid_neighbors.append(self.grid[i, j])
        return valid_neighbors

    def update(self, g):
        # find state at t+1 based on neighbors' states at t
        for ant_idx, ant in self.ant_dict.items():
            neighbors = self._moore_neighbors(ant.x, ant.y)
            neighbor_sum = ant.state
            for n in neighbors:
                neighbor_sum += self.ant_dict[n].state
            ant.next_state = np.tanh(g * neighbor_sum)
        # update state to t+1
        for ant_idx, ant in self.ant_dict.items():
            ant.state = ant.next_state

    def spontaneous_activity(self, threshold, activation_probability, spontaneous_activity_level):
        for ant_idx, ant in self.ant_dict.items():
            if ant.state < threshold:
                if random.random() < activation_probability:
                    ant.state = spontaneous_activity_level

    def move_4(self, ant):
        """
        moves an ant to a random adjacent square if available
        """
        x, y = ant.x, ant.y
        options = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        valid_options = []
        for opt in options:
            i, j = opt
            if 0 <= i < self.size:
                if 0 <= j < self.size:
                    # if there is not an ant at this location
                    if self.grid[i, j] == self.empty_idx:
                        valid_options.append(opt)
        if len(valid_options) > 0:
            new_x, new_y = random.choice(valid_options)
            self.grid[x, y] = self.empty_idx
            self.grid[new_x, new_y] = ant.idx
            ant.x = new_x
            ant.y = new_y

    def move_8(self, threshold):
        """
        Moves each ant to a randomly chosen open neighboring square (uses 8 neighbors)
        """
        boundary_sites = boundary(self.size)
        indices = list(self.ant_dict.keys())
        to_delete = []
        random.shuffle(indices)
        for idx in indices:
            ant = self.ant_dict[idx]
            if ant.state > threshold:
                x, y = ant.x, ant.y
                x_options = [x - 1, x, x + 1]
                y_options = [y - 1, y, y + 1]
                valid_x = []
                valid_y = []
                for i in x_options:
                    for j in y_options:
                        if 0 <= i < self.size:
                            if 0 <= j < self.size:
                                if i != x or j != y:
                                    # if there is not an ant at this location
                                    if self.grid[i, j] == self.empty_idx:
                                        valid_x.append(i)
                                        valid_y.append(j)

                if len(valid_x) > 0:
                    index = random.randint(0, len(valid_x)-1)  # randint is inclusive on both ends
                    new_x = valid_x[index]
                    new_y = valid_y[index]
                    self.grid[x, y] = self.empty_idx
                    if (new_x, new_y) in boundary_sites:
                        to_delete.append(ant.idx)
                    else:
                        self.grid[new_x, new_y] = ant.idx
                        ant.x = new_x
                        ant.y = new_y
        for ant_idx in to_delete:
            del self.ant_dict[ant_idx]

    def enter(self, enter_prob, enter_state):
        boundary_sites = boundary(self.size)
        for site in boundary_sites:
            x, y = site
            if random.random() < enter_prob:
                if self.grid[site] == self.empty_idx:
                    self.grid[site] = self.next_agent_idx
                    new_ant = Ant(x, y, self.next_agent_idx, state=enter_state)
                    self.ant_dict[self.next_agent_idx] = new_ant
                    self.next_agent_idx += 1

    def count_actives(self, threshold):
        num_active = 0
        for ant_idx, ant in self.ant_dict.items():
            if ant.state > threshold:
                num_active += 1
        return num_active


class Ant:
    def __init__(self, x, y, idx, state=0.0):
        self.x = x
        self.y = y
        self.idx = idx
        self.state = state


def boundary(size):
    boundary_sites = set()
    to_clear = {0, size-1}
    for r in range(size):
        for c in range(size):
            if r in to_clear or c in to_clear:
                boundary_sites.add((r, c))
    return boundary_sites


def _ent(p):
    return 0 if p == 0 else (-p * math.log(p, 2))


def _entropy(activity):
    T = len(activity)
    T_j = {}
    for activity_level in activity:
        if activity_level in T_j:
            T_j[activity_level] += 1
        else:
            T_j[activity_level] = 1
    S = 0
    for k, v in T_j.items():
        p = v/T
        S += _ent(p)
    return S


def _fractional_entropy(activity):
    rounded_fraction_active = [round(i, 4) for i in activity]
    entropy = _entropy(rounded_fraction_active)
    return entropy


def run(timesteps, g, density, grid_size, enter_prob, enter_state, activation_prob, burn_in, plotting):
    activity_threshold = 10e-16
    activation_probability = activation_prob  # for 50x50: 0.00001  # 0.025
    spontaneous_activity_level = 0.1
    num_agents = int(round(density * grid_size ** 2))
    if num_agents < 2: num_agents = 2
    grid = Grid(grid_size, num_agents)

    num_active = []
    total_ants = []
    for t in range(timesteps):
        # update ants based on neighbors
        grid.update(g=g)
        # spontaneous activations
        grid.spontaneous_activity(threshold=activity_threshold,
                                  activation_probability=activation_probability,
                                  spontaneous_activity_level=spontaneous_activity_level)
        # ants enter on boundary with probability p
        grid.enter(enter_prob, enter_state)
        # all ants move
        grid.move_8(threshold=activity_threshold)
        total_on_grid = len(grid.ant_dict)
        total_ants.append(total_on_grid)
        count_active = grid.count_actives(threshold=activity_threshold)
        num_active.append(count_active)

    # don't use the first n timesteps so system has time to reach typical behavior
    percentage_active = [num_active[i] / total_ants[i] if total_ants[i] > 0 else 0 for i in range(burn_in, len(num_active))]
    density = [total_ants[i]/(grid.size ** 2) for i in range(burn_in, len(num_active))]
    #### What we're using for entropy here
    ent = _entropy(num_active[burn_in:])
    if plotting:
        plt.plot(percentage_active, alpha=0.5)
        plt.scatter([i for i in range(len(percentage_active))], percentage_active, s=2, label='density')
        plt.title(f'g={round(g, 4)}, enter={round(enter_prob, 3)}, ent={round(ent, 4)}')
        plt.xlabel('Timesteps')
        plt.ylabel('Activity')
        plt.ylim((0, 1))
        # plt.savefig(f'grid_size_{grid_size}/{grid_size}_{enter_state}_{activation_prob}_{g}_{enter_prob}.png')
        # plt.clf()
        plt.show()
    return ent, np.mean(density)


def param_search(grid_size, letter, plotting):
    enter_state = 10e-16
    activation_prob = 0.0001
    with open(f'grid_{grid_size}_activity_{enter_state}_prob_{activation_prob}_{letter}.txt', 'w+') as f:
        for g in [0.4, 0.45, 0.5, 0.6, 0.7]:
            for ep in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                ent, mean_density = run(timesteps=4000,
                                        g=g,
                                        density=0.5,
                                        grid_size=grid_size,
                                        enter_prob=ep,
                                        enter_state=enter_state,
                                        activation_prob=activation_prob,
                                        burn_in=2000,
                                        plotting=plotting)
                print(g, ep, ent)
                f.write(f'{g} {ep} {ent}\n')


def main():
    param_search(grid_size=15, letter='a', plotting=True)


if __name__ == '__main__':
    main()
