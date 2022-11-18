import random
import numpy as np
import functools

import co_functions as cf
import utils

DIMENSION = 10 # dimension of the problems
POP_SIZE = 100 # population size
MAX_GEN = 500 # maximum number of generations
REPEATS = 10 # number of runs of algorithm (should be at least 10)
OUT_DIR = 'continuous' # output directory for logs
EXP_ID = 'P2Differential' # the ID of this experiment (used to create log names)
CR = 0.9 # Change propability
F = 0.8 # Multiplier of difference
DIFF_COUNT = 1

# creates the individual
def create_ind(ind_len):
    return np.random.uniform(-5, 5, size=(ind_len,))

# creates the population using the create individual function
def create_pop(pop_size, create_individual):
    return [create_individual() for _ in range(pop_size)]

def select_random_unique(idx, pop, diff_count):
    while True:
        inds = random.sample(range(len(pop)), diff_count*2 + 1)
        if idx not in inds:
            return inds

def create_donor(inds, pop, fm, diff_count):
    donor = pop[inds.pop(0)][:]
    for idx1, idx2 in zip(inds[0::2], inds[1::2]):
        donor = donor + fm*(pop[idx1] - pop[idx2])
    return donor

def select_random_dim(dimensions):
    return random.randint(0, dimensions-1)

def select_better(donor, parent, fitness):
    if fitness(donor).fitness > fitness(parent).fitness:
        return donor[:]
    else:
        return parent[:]

def uniform_cx(donor, parent, cr_prob):
    must_change_idx = select_random_dim()
    for i, v in np.ndenumerate(parent):
        if random.random() >= cr_prob and must_change_idx != i:
            donor[i] = v
    return donor[:]

# implements the evolutionary algorithm
# arguments:
#   pop_size  - the initial population
#   max_gen   - maximum number of generation
#   fitness   - fitness function (takes individual as argument and returns 
#               FitObjPair)
#   map_fn    - function to use to map fitness evaluation over the whole 
#               population (default `map`)
#   log       - a utils.Log structure to log the evolution run
def evolutionary_algorithm(pop, max_gen, fitness, select, mutate, crossover, select_off,*, map_fn=map, log=None):
    evals = 0
    for G in range(max_gen):
        fits_objs = list(map_fn(fitness, pop))
        evals += len(pop)
        if log:
            log.add_gen(fits_objs, evals)
        fits = [f.fitness for f in fits_objs]
        objs = [f.objective for f in fits_objs]
        
        offspring = []
        for idx in range(len(pop)):
            inds = select(idx, pop)
            donor = crossover(mutate(inds), pop[idx])
            offspring.append(select_off(donor, pop[idx]))
        
        pop = offspring[:]
    return pop

if __name__ == '__main__':

    # use `functool.partial` to create fix some arguments of the functions 
    # and create functions with required signatures
    cr_ind = functools.partial(create_ind, ind_len=DIMENSION)
    select = functools.partial(select_random_unique, diff_count=DIFF_COUNT)
    mutate = functools.partial(create_donor, fm=F ,diff_count=DIFF_COUNT)
    crossover = functools.partial(uniform_cx, cr_prob=CR)

    # we will run the experiment on a number of different functions
    fit_generators = [cf.make_f01_sphere,
                      cf.make_f02_ellipsoidal,
                      cf.make_f06_attractive_sector,
                      cf.make_f08_rosenbrock,
                      cf.make_f10_rotated_ellipsoidal]
    fit_names = ['f01', 'f02', 'f06', 'f08', 'f10']

    for fit_gen, fit_name in zip(fit_generators, fit_names):
        fit = fit_gen(DIMENSION)
        select_off = functools.partial(select_better, fitness=fit)

        # run the algorithm `REPEATS` times and remember the best solutions from 
        # last generations
    
        best_inds = []
        for run in range(REPEATS):
            # initialize the log structure
            log = utils.Log(OUT_DIR, EXP_ID + '.' + fit_name , run, 
                            write_immediately=True, print_frequency=5)
            # create population
            pop = create_pop(POP_SIZE, cr_ind)
            # run evolution - notice we use the pool.map as the map_fn
            pop = evolutionary_algorithm(pop, MAX_GEN, fit, select, mutate, crossover, select_off, map_fn=map, log=log)
            # remember the best individual from last generation, save it to file
            bi = max(pop, key=fit)
            best_inds.append(bi)
            
            # if we used write_immediately = False, we would need to save the 
            # files now
            # log.write_files()

        # print an overview of the best individuals from each run
        for i, bi in enumerate(best_inds):
            print(f'Run {i}: objective = {fit(bi).objective}')

        # write summary logs for the whole experiment
        utils.summarize_experiment(OUT_DIR, EXP_ID + '.' + fit_name)