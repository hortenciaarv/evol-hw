'''
HW03
Differential Evolution algorithm

Hortencia Alejandra Ramírez Vázquez
Amanda Valdez Calderón
'''

import numpy as np

def differential_evolution(func, bounds, pop_size, max_gen, F, cr):
    dim = len(bounds)

    #initialize population
    pop = np.random.rand(pop_size, dim)
    for i in range(dim):
        pop[:, i] = bounds[i][0] + pop[:, i] * (bounds[i][1] - bounds[i][0])

    #evaluante initial population
    fitness = np.array([func(ind) for ind in pop])

    for gen in range(max_gen):
        for i in range(pop_size):

            # Mutation: DE/rand/1
            idxs = [idx for idx in range(pop_size) if idx != i]
            # v(i,g) = x(r0,g) + F * (x(r1,g) - x(r2,g))
            r0, r1, r2 = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(r0 + F * (r1 - r2), 
                             [b[0] for b in bounds], 
                             [b[1] for b in bounds])
            
            #crossover
            cross_points = np.random.rand(dim) < cr
            j_rand = np.random.randint(dim)
            cross_points[j_rand] = True  
            trial = np.where(cross_points, mutant, pop[i])

            #selection
            f = func(trial)
            if f < fitness[i]:
                pop[i] = trial
                fitness[i] = f
    
    # Return best solution
    best_idx = np.argmin(fitness)
    return pop[best_idx], fitness[best_idx]

if __name__ == "__main__":
    def rastrigin(x):
        A = 10
        return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

    bounds = [(-5.12, 5.12)] * 2
    best, score = differential_evolution(rastrigin, bounds, pop_size=50, max_gen=100, F=0.8, cr=0.9)
    print("Best solution:", best)
    print("Fitness:", score)