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
EXP_ID = 'P2DifferentialAdaptiveCR09-F035-FD035-CRD09' # the ID of this experiment (used to create log names)
CR = 0.9 # Change propability
F = 0.8 # Multiplier of difference
DIFF_COUNT = 1
F_DEC = 0.35
CR_DEC = 0.9

# Velka F i moc mala nic moc nedelala, nejlepsi to bylo pro 0.55 az 0.6
# pridavani countu bylo k nicemu
# cx prob jelo rychle celkem pro 0.9 a v rozmezi 0.5
# jen zmensovat se to moc nedari, zkusim to udelat jako s tou jednou petinou
# s jednou petinou to celkem
# celkem fajn s 0.3 f a cr 0.8, taky 0.35 az 0.9

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


class BaseF:
    def __init__(self, fm):
        self.fm = fm
        self.init_fm = fm

    def __call__(self, val):
        return self.fm * val

    def reset(self):
        self.fm = self.init_fm

class AdaptiveFBasic(BaseF):

    def __init__(self, fm, decreaser):
        super().__init__(fm)
        self.dec = decreaser

    def update(self):
        self.fm *= self.dec


class AdaptiveOneFifthF(AdaptiveFBasic):

    def __init__(self, fm, decreaser):
        super().__init__(fm, decreaser)

    def update(self, succ_rate):
        if succ_rate != 0.0:
            if succ_rate > 0.2:
                self.fm = self.fm / self.dec
            elif succ_rate < 0.2:
                self.fm = self.fm * self.dec


class BaseCR:

    def __init__(self, cr_prob, dims):
        self.cr = cr_prob
        self.init_cr = cr_prob
        self.dims = dims

    def select_random_dim(self):
        return random.randint(0, self.dims-1)

    def __call__(self, donor, parent):
        must_change_idx = self.select_random_dim()
        for i, v in np.ndenumerate(parent):
            if random.random() >= self.cr and must_change_idx != i:
                donor[i] = v
        return donor[:]

    def reset(self):
        self.cr = self.init_cr

class AdaptiveCRBasic(BaseCR):

    def __init__(self, cr_prob, dims, decreaser):
        super().__init__(cr_prob, dims)
        self.dec = decreaser

    def update(self):
        self.cr *= self.dec


class AdaptiveOneFifthCR(AdaptiveCRBasic):

    def __init__(self, cr_prob, dims, decreaser):
        super().__init__(cr_prob, dims, decreaser)

    def update(self, succ_rate):
        if succ_rate != 0.0:
            if succ_rate > 0.2:
                self.cr = self.cr / self.dec
            elif succ_rate < 0.2:
                self.cr = self.cr * self.dec

def create_donor(inds, pop, fm):
    donor = pop[inds.pop(0)][:]
    for idx1, idx2 in zip(inds[0::2], inds[1::2]):
        donor = donor + fm(pop[idx1] - pop[idx2])
    return donor

def select_random_dim(dimensions):
    return random.randint(0, dimensions-1)

def select_better(donor, parent, fitness):
    if fitness(donor).fitness >= fitness(parent).fitness:
        return 1, donor[:]
    else:
        return 0, parent[:]


# implements the evolutionary algorithm
# arguments:
#   pop_size  - the initial population
#   max_gen   - maximum number of generation
#   fitness   - fitness function (takes individual as argument and returns 
#               FitObjPair)
#   map_fn    - function to use to map fitness evaluation over the whole 
#               population (default `map`)
#   log       - a utils.Log structure to log the evolution run
def evolutionary_algorithm(pop, max_gen, fitness, select, mutate, crossover, select_off,*, map_fn=map, log=None, mutate_f=None):
    evals = 0
    mutate_f.reset()
    crossover.reset()
    for G in range(max_gen):
        successes = 0
        fits_objs = list(map_fn(fitness, pop))
        evals += len(pop)
        if log:
            log.add_gen(fits_objs, evals)
        fits = [f.fitness for f in fits_objs]
        objs = [f.objective for f in fits_objs]
        
        offspring = []
        for idx in range(len(pop)):
            inds = select(idx, pop)
            donor = crossover(mutate(inds, pop), pop[idx])
            succ, best = select_off(donor, pop[idx])
            offspring.append(best)
            successes += succ

        if mutate_f is not None:
            mutate_f.update(successes / len(pop))
            crossover.update(successes / len(pop))

        pop = offspring[:]
    return pop

if __name__ == '__main__':

    # use `functool.partial` to create fix some arguments of the functions 
    # and create functions with required signatures
    cr_ind = functools.partial(create_ind, ind_len=DIMENSION)
    select = functools.partial(select_random_unique, diff_count=DIFF_COUNT)
    adaptive_f = AdaptiveOneFifthF(F, F_DEC)
    mutate = functools.partial(create_donor, fm=adaptive_f)
    crossover = AdaptiveOneFifthCR(CR, DIMENSION, CR_DEC)

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
            pop = evolutionary_algorithm(pop, MAX_GEN, fit, select, mutate, crossover, select_off, map_fn=map, log=log, mutate_f=adaptive_f)
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