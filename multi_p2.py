import copy
import functools
import numpy as np
import operator
import random

import moo_functions as mf
import multi_utils as mu
import utils

DIMENSION = 10 # dimension of the problems
POP_SIZE = 100 # population size
MAX_GEN = 50 # maximum number of generations
CX_PROB = 0.8 # crossover probability
MUT_PROB = 0.2 # mutation probability
MUT_STEP = 0.55 # size of the mutation steps
AD_MUT_C = 0.8 # Multiplier in adaptive stepsize
REPEATS = 10 # number of runs of algorithm (should be at least 10)
AR_CX_WEIGHT = 0.75
DIFF_COUNT = 1
F = 0.8 # Multiplier of difference
OUT_DIR = 'multi' # output directory for logs
EXP_ID = f'HyperVolume+CXUniform+DiffAdaptiveMut+CXP{CX_PROB}+MP{MUT_PROB}+MS{MUT_STEP}' # the ID of this experiment (used to create log names)

#large step size did good stuff

class MultiIndividual:

    def __init__(self, x):
        self.x = x
        self.fitness = None
        self.ssc = None
        self.front = None

# creates the individual
def create_ind(ind_len):
    return MultiIndividual(np.random.uniform(0, 1, size=(ind_len,)))

# creates the population using the create individual function
def create_pop(pop_size, create_individual):
    return [create_individual() for _ in range(pop_size)]

# the tournament selection (roulette wheell would not work, because we can have 
# negative fitness)
def tournament_selection_NSGA2(pop, k):
    selected = []
    for i in range(k):
        p1 = random.randrange(0, len(pop))
        p2 = random.randrange(0, len(pop))
        if (pop[p1].front, -pop[p1].ssc) < (pop[p2].front, -pop[p2].ssc): # lexicographic comparison
            selected.append(copy.deepcopy(pop[p1]))
        else:
            selected.append(copy.deepcopy(pop[p2]))

    return selected

def nsga2_select(pop, k):
    fronts = mu.divide_fronts(pop)
    selected = []
    for i, f in enumerate(fronts):
        mu.assign_hypervolume_addition(f)
        for ind in f:
            ind.front = i
        if len(selected) + len(f) <= k:
            selected += f
        else:
            break
    
    assert len(selected) <= k
    assert len(f) + len(selected) >= k

    if len(selected) != k:
        # f is now the front that did not fit fully
        selected += list(sorted(f, key=lambda x: -x.ssc))[:k - len(selected)]

    assert len(selected) == k

    return selected

# to the whole population with probability mut_prob)
def mutation(pop, mutate, mut_prob):
    return [mutate(idx, p, pop) if random.random() < mut_prob else copy.deepcopy(p) for idx, p in enumerate(pop)]


class DiffMutation:
    
    def __init__(self, diff_count, F, fitness):
        self.diff_count = diff_count
        self.F = F
        self.fit = fitness
        self.init_f = F
        self.mut_count = 0
        self.succ_mut_count = 0

    def select_random_unique(self, idx, pop):
        while True:
            inds = random.sample(range(len(pop)), self.diff_count*2)
            if idx not in inds:
                return inds

    def __call__(self, idx, ind, pop):
        inds = self.select_random_unique(idx, pop)
        donor = copy.deepcopy(ind)
        # donor = pop[inds.pop(0)][:]
        for idx1, idx2 in zip(inds[0::2], inds[1::2]):
            a = donor.x + self.F*(copy.deepcopy(pop[idx1]).x - copy.deepcopy(pop[idx2]).x)
            np.clip(a, 0, 1, donor.x)
        of1, of2 = self.fit(ind)
        nf1, nf2 = self.fit(donor)
        if nf1 >= of1 and nf2 >= of2:
            self.succ_mut_count += 1
        self.mut_count += 1   
        return donor

    def update(self):
        if self.mut_count != 0:
            succ_rate = self.succ_mut_count / self.mut_count
            if succ_rate > 0.2:
                self.F = self.F / AD_MUT_C
            elif succ_rate < 0.2:
                self.F = self.F * AD_MUT_C
        self.succ_mut_count = 0
        self.mut_count = 0

    def reset(self):
        self.F = self.init_f

# gaussian mutation - we need a class because we want to change the step
# size of the mutation adaptively
class BasicMutation:

    def update_step_size(self):
        pass
    
    def __init__(self, step_size, fitness):
        self.step_size = step_size
        self.init_step_size = step_size
        self.fit = fitness

    def __call__(self, idx, ind, pop=None):
        a = ind.x + self.step_size*np.random.normal(size=ind.x.shape)
        np.clip(a, 0, 1, ind.x)
        return ind
    
    def reset(self):
        self.step_size = self.init_step_size


class AdaptiveOneFifthMutation(BasicMutation):

    def __init__(self, step_size, fitness):
        super().__init__(step_size, fitness)
        self.mut_count = 0
        self.succ_mut_count = 0

    def __call__(self, idx, ind, pop=None):
        old_ind = copy.deepcopy(ind)
        a = ind.x + self.step_size*np.random.normal(size=ind.x.shape)
        np.clip(a, 0, 1, ind.x)
        of1, of2 = self.fit(old_ind)
        nf1, nf2 = self.fit(ind)
        if nf1 >= of1 and nf2 >= of2:
        #if nf1 + 3*nf2 >= of1 + 3*of2:
            self.succ_mut_count += 1
        self.mut_count += 1
        return ind

    def update(self):
        if self.mut_count != 0:
            succ_rate = self.succ_mut_count / self.mut_count
            if succ_rate > 0.2:
                self.step_size = self.step_size / AD_MUT_C
            elif succ_rate < 0.2:
                self.step_size = self.step_size * AD_MUT_C
        self.succ_mut_count = 0
        self.mut_count = 0
 

# applies a list of genetic operators (functions with 1 argument - population) 
# to the population

# applies the cross function (implementing the crossover of two individuals)
# to the whole population (with probability cx_prob)
def crossover(pop, cross, cx_prob):
    off = []
    for p1, p2 in zip(pop[0::2], pop[1::2]):
        if random.random() < cx_prob:
            o1, o2 = cross(p1, p2)
        else:
            o1, o2 = copy.deepcopy(p1), copy.deepcopy(p2)
        off.append(o1)
        off.append(o2)
    return off

# implements the one-point crossover of two individuals
def one_pt_cross(p1, p2):
    point = random.randrange(1, len(p1.x))
    p1 = copy.deepcopy(p1)
    p2 = copy.deepcopy(p2)
    o1 = np.append(p1.x[:point], p2.x[point:])
    o2 = np.append(p2.x[:point], p1.x[point:])
    p1.x = o1
    p2.x = o2
    return p1, p2


def uniform_cross(p1, p2):
    for idx in range(len(p1.x)):
        if random.randint(0, 1):
            p1.x[idx], p2.x[idx] = p2.x[idx], p1.x[idx]
    return p1, p2

def arithmetic_cross(p1, p2):
    for idx in range(len(p1.x)):
        p1n = AR_CX_WEIGHT*p1.x[idx] + (1-AR_CX_WEIGHT)*p2.x[idx]
        p2n = AR_CX_WEIGHT*p2.x[idx] + (1-AR_CX_WEIGHT)*p1.x[idx]
        p1.x[idx], p2.x[idx] = p1n, p2n
    return p1, p2


def mate(pop, operators):
    for o in operators:
        pop = o(pop)
    return pop

# applies the mutate function (implementing the mutation of a single individual)

# implements the evolutionary algorithm
# arguments:
#   pop_size  - the initial population
#   max_gen   - maximum number of generation
#   fitness   - fitness function (takes individual as argument and returns 
#               FitObjPair)
#   operators - list of genetic operators (functions with one arguments - 
#               population; returning a population)
#   mate_sel  - mating selection (funtion with three arguments - population, 
#               fitness values, number of individuals to select; returning the 
#               selected population)
#   mutate_ind - reference to the class to mutate an individual - can be used to 
#               change the mutation step adaptively
#   map_fn    - function to use to map fitness evaluation over the whole 
#               population (default `map`)
#   log       - a utils.Log structure to log the evolution run
def evolutionary_algorithm(pop, max_gen, fitness, operators, mate_sel, mutate_ind, *, map_fn=map, log=None, opt_hv = np.product(mu.HYP_REF)):
    evals = 0
    mutate_ind.reset()
    for G in range(max_gen):        

        if G == 0:
            fits_objs = list(map_fn(fitness, pop))
            for ind, fit in zip(pop, fits_objs):
                ind.fitness = fit
            evals += len(pop)
            fronts = mu.divide_fronts(pop)
            for i,f in enumerate(fronts):
                mu.assign_hypervolume_addition(f)
                for ind in f:
                    ind.front = i
            
        if log:
            log.add_multi_gen(pop, evals, opt_hv)

        mating_pool = mate_sel(pop, POP_SIZE)
        offspring = mate(mating_pool, operators)
        fits_objs = list(map_fn(fitness, offspring))
        for ind, fit in zip(offspring, fits_objs):
            ind.fitness = fit
            
            
        evals += len(offspring)
        pop = nsga2_select(pop + offspring, POP_SIZE)
        mutate_ind.update()

    return pop

if __name__ == '__main__':

    # use `functool.partial` to create fix some arguments of the functions 
    # and create functions with required signatures
    cr_ind = functools.partial(create_ind, ind_len=DIMENSION)
    # we will run the experiment on a number of different functions
    fit_names = ['ZDT1', 'ZDT2', 'ZDT3', 'ZDT4', 'ZDT6']

    for fit_name in fit_names:
        fit = mf.get_function_by_name(fit_name)
        opt_hv = mf.get_opt_hypervolume(fit_name)
        mutate_ind = DiffMutation(diff_count=DIFF_COUNT, F=F, fitness=fit) #AdaptiveOneFifthMutation(step_size=MUT_STEP, fitness=fit) #DiffMutation(diff_count=DIFF_COUNT, F=F, fitness=fit)
        xover = functools.partial(crossover, cross=uniform_cross, cx_prob=CX_PROB)
        mut = functools.partial(mutation, mut_prob=MUT_PROB, mutate=mutate_ind)

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
            pop = evolutionary_algorithm(pop, MAX_GEN, fit, [xover, mut], tournament_selection_NSGA2, mutate_ind, map_fn=map, log=log, opt_hv=opt_hv)
            # remember the best individual from last generation, save it to file
            best_inds.append(mu.hypervolume(pop))
            
            # if we used write_immediately = False, we would need to save the 
            # files now
            # log.write_files()

        # print an overview of the best individuals from each run
        for i, bi in enumerate(best_inds):
            print(f'Run {i}: objective = {opt_hv - bi}')

        # write summary logs for the whole experiment
        utils.summarize_experiment(OUT_DIR, EXP_ID + '.' + fit_name)