import random
import numpy as np
import functools
import sys
import utils

K = 10 #number of piles
POP_SIZE = 100 # population size
MAX_GEN = 500 # maximum number of generations
CX_PROB = 0.4 # crossover probability
MUT_PROB = 1 # mutation probability
MUT_FLIP_PROB = 0.002 # probability of chaninging value during mutation
REPEATS = 10 # number of runs of algorithm (should be at least 10)
ELITISM = 0.07 
ELITE_COUNT = round(POP_SIZE*ELITISM)
OUT_DIR = 'partition' # output directory for logs
#EXP_ID = 'test'
EXP_ID = f'hw3-diffavg-tournament-swapmaxminbest-popsize{POP_SIZE}-maxgen{MAX_GEN}-mutprob{MUT_PROB}-cxprob{CX_PROB}--elite{ELITISM}' # the ID of this experiment (used to create log names)


# reads the input set of values of objects
def read_weights(filename):
    with open(filename) as f:
        return list(map(int, f.readlines()))

# computes the bin weights
# - bins are the indices of bins into which the object belongs
def bin_weights(weights, bins):
    bw = [0]*K
    for w, b in zip(weights, bins):
        bw[b] += w
    return bw

# the fitness function
def fitness(ind, weights):
    bw = bin_weights(weights, ind)
    return utils.FitObjPair(fitness=1/(max(bw) - min(bw) + 1), 
                            objective=max(bw) - min(bw))

def fitness_with_average(ind, weights):
    bw = bin_weights(weights, ind)
    fit_value = 0
    avg_weight = sum(bw) / len(bw)
    for w in bw:
        fit_value += abs(w - avg_weight)

    return utils.FitObjPair(1 / (fit_value + 1),
                            objective=max(bw) - min(bw))

# creates the individual
def create_ind(ind_len):
    return [random.randrange(0, K) for _ in range(ind_len)]

# creates the population using the create individual function
def create_pop(pop_size, create_individual):
    return [create_individual() for _ in range(pop_size)]

# the roulette wheel selection
def roulette_wheel_selection(pop, fits, k):
    return random.choices(pop, fits, k=k)

# tournament selection
def tournament_selection(pop, fits, k):
    selected = []
    for _ in range(k):
        i1, i2 = random.randrange(0, len(pop)), random.randrange(0, len(pop))
        if fits[i1] > fits[i2]:
            selected.append(pop[i1])
        else: 
            selected.append(pop[i2])
    return selected

# implements the one-point crossover of two individuals
def one_pt_cross(p1, p2):
    point = random.randrange(1, len(p1))
    o1 = p1[:point] + p2[point:]
    o2 = p2[:point] + p1[point:]
    return o1, o2

# applies the cross function (implementing the crossover of two individuals)
# to the whole population (with probability cx_prob)
def crossover(pop, cross, cx_prob):
    off = []
    for p1, p2 in zip(pop[0::2], pop[1::2]):
        if random.random() < cx_prob:
            o1, o2 = cross(p1, p2)
        else:
            o1, o2 = p1[:], p2[:]
        off.append(o1)
        off.append(o2)
    return off

# implements the "bit-flip" mutation of one individual
def flip_mutate(p, prob, upper):
    return [random.randrange(0, upper) if random.random() < prob else i for i in p]

def bucket_items_indeces(ind, bin_id):
    return [idx for idx, b in enumerate(ind) if b == bin_id]

def swap_max_min_mutate(ind, weights):
    bw = bin_weights(weights, ind)
    max_bin_id, min_bin_id = np.argmax(bw), np.argmin(bw)
    max_item_idx_max_bin = max(bucket_items_indeces(ind, max_bin_id), key=lambda x: weights[x])
    min_item_idx_min_bin = min(bucket_items_indeces(ind, min_bin_id), key=lambda x: weights[x])
    ind[max_item_idx_max_bin] = min_bin_id
    ind[min_item_idx_min_bin] = max_bin_id
    return ind[:]

def swap_max_min_best_mutate(ind, weights):
    bw = bin_weights(weights, ind)
    max_bin_id, min_bin_id = np.argmax(bw), np.argmin(bw)
    max_bin_items = bucket_items_indeces(ind, max_bin_id)
    min_bin_items = bucket_items_indeces(ind, min_bin_id)
    
    ibmax_idx, ibmin_idx = 0, 0
    min_diff = sys.maxsize
    for max_bin_item in max_bin_items:
        for min_bin_item in min_bin_items:
            diffax = bw[max_bin_id] - weights[max_bin_item] + weights[min_bin_item]
            diffin = bw[min_bin_id] - weights[min_bin_item] + weights[max_bin_item]
            diff = abs(diffax - diffin)
            if diff < min_diff:
                min_diff = diff
                ibmax_idx, ibmin_idx = max_bin_item, min_bin_item
    
    ind[ibmax_idx], ind[ibmin_idx] = min_bin_id, max_bin_id
    return ind[:] 


# applies the mutate function (implementing the mutation of a single individual)
# to the whole population with probability mut_prob)
def mutation(pop, mutate, mut_prob):
    return [mutate(p) if random.random() < mut_prob else p[:] for p in pop]

# applies a list of genetic operators (functions with 1 argument - population) 
# to the population
def mate(pop, operators):
    for o in operators:
        pop = o(pop)
    return pop

def elitism(pop, offspring, fitness):
    return sorted(pop, key=fitness)[-ELITE_COUNT:] + sorted(offspring, key=fitness)[ELITE_COUNT:]

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
#   map_fn    - function to use to map fitness evaluation over the whole 
#               population (default `map`)
#   log       - a utils.Log structure to log the evolution run
def evolutionary_algorithm(pop, max_gen, fitness, operators, mate_sel, *, map_fn=map, log=None):
    evals = 0
    for G in range(max_gen):
        fits_objs = list(map_fn(fitness, pop))
        evals += len(pop)
        if log:
            log.add_gen(fits_objs, evals)
        fits = [f.fitness for f in fits_objs]
        objs = [f.objective for f in fits_objs]

        mating_pool = mate_sel(pop, fits, POP_SIZE)
        offspring = mate(mating_pool, operators)
        pop = elitism(pop, offspring, fitness)[:]

    return pop


if __name__ == '__main__':
    # read the weights from input
    weights = read_weights('inputs/partition-easy.txt')

    # use `functool.partial` to create fix some arguments of the functions 
    # and create functions with required signatures
    cr_ind = functools.partial(create_ind, ind_len=len(weights))
    fit = functools.partial(fitness_with_average, weights=weights)
    xover = functools.partial(crossover, cross=one_pt_cross, cx_prob=CX_PROB)
    mut = functools.partial(mutation, mut_prob=MUT_PROB, mutate=functools.partial(swap_max_min_best_mutate, weights=weights))

    #mut = functools.partial(mutation, mut_prob=MUT_PROB, 
    #                       mutate=functools.partial(flip_mutate, prob=MUT_FLIP_PROB, upper=K))

    # we can use multiprocessing to evaluate fitness in parallel
    import multiprocessing
    pool = multiprocessing.Pool()

    import matplotlib.pyplot as plt

    # run the algorithm `REPEATS` times and remember the best solutions from 
    # last generations
    best_inds = []
    for run in range(REPEATS):
        # initialize the log structure
        log = utils.Log(OUT_DIR, EXP_ID, run, 
                        write_immediately=True, print_frequency=5)
        # create population
        pop = create_pop(POP_SIZE, cr_ind)
        # run evolution - notice we use the pool.map as the map_fn
        pop = evolutionary_algorithm(pop, MAX_GEN, fit, [xover, mut], tournament_selection, map_fn=pool.map, log=log)
        # remember the best individual from last generation, save it to file
        bi = max(pop, key=fit)
        best_inds.append(bi)

        with open(f'{OUT_DIR}/{EXP_ID}_{run}.best', 'w') as f:
            for w, b in zip(weights, bi):
                f.write(f'{w} {b}\n')
        
        # if we used write_immediately = False, we would need to save the 
        # files now
        # log.write_files()

    # print an overview of the best individuals from each run
    for i, bi in enumerate(best_inds):
        print(f'Run {i}: difference = {fit(bi).objective}, bin weights = {bin_weights(weights, bi)}')

    # write summary logs for the whole experiment
    utils.summarize_experiment(OUT_DIR, EXP_ID)

    # read the summary log and plot the experiment
    evals, lower, mean, upper = utils.get_plot_data(OUT_DIR, EXP_ID)
    plt.figure(figsize=(12, 8))
    utils.plot_experiment(evals, lower, mean, upper, legend_name = 'Default settings')
    plt.legend()
    plt.show()

    # you can also plot mutiple experiments at the same time using 
    # utils.plot_experiments, e.g. if you have two experiments 'default' and 
    # 'tuned' both in the 'partition' directory, you can call
    # utils.plot_experiments('partition', ['default', 'tuned'], 
    #                        rename_dict={'default': 'Default setting'})
    # the rename_dict can be used to make reasonable entries in the legend - 
    # experiments that are not in the dict use their id (in this case, the 
    # legend entries would be 'Default settings' and 'tuned') 