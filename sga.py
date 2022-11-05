import random
import pprint
import numpy as np
import matplotlib.pyplot as plt


POP_SIZE = 200
IND_LEN = 25
#CX_PROB = 0.8
#MUT_PROB = 0.05
MUT_FLIP_PROB = 0.1

# creates a single individual of lenght `lenght`
def create_ind(length):
    return [random.randint(0, 1) for _ in range(length)]

# creates a population of `size` individuals
def create_population(size):
    return [create_ind(IND_LEN) for _ in range(size)]

# tournament selection
# def selection(pop, fits):
#     selected = []
#     for _ in range(len(pop)):
#         i1, i2 = random.randrange(0, len(pop)), random.randrange(0, len(pop))
#         if fits[i1] > fits[i2]:
#             selected.append(pop[i1])
#         else: 
#             selected.append(pop[i2])
#     return selected

# roulette wheel selection
def selection(pop, fits):
    return random.choices(pop, fits, k=POP_SIZE)

# one point crossover
def cross(p1, p2):
    point = random.randint(0, len(p1))
    o1 = p1[:point] + p2[point:]
    o2 = p2[:point] + p1[point:]
    return o1, o2

# applies crossover to all individuals
def crossover(pop, cx_prob):
    off = []
    for p1, p2 in zip(pop[0::2], pop[1::2]):
        o1, o2 = p1[:], p2[:]
        if random.random() < cx_prob:
            o1, o2 = cross(p1[:], p2[:])
        off.append(o1)
        off.append(o2)
    return off

# bit flip mutation
def mutate(p):
    o = []
    for g in p:
        if random.random() < MUT_FLIP_PROB:
            g = 1-g
        o.append(g)
    return o

def mutation(pop, mut_prob):
    off = []
    for p in pop:
        if random.random() < mut_prob:
            o = mutate(p)
        else:
            o = p[:]
        off.append(o)
    return off

# applies crossover and mutation
def operators(pop, cx_prob, mut_prob):
    pop1 = crossover(pop, cx_prob)
    return mutation(pop1, mut_prob)

def fit_plc(ind):
    even_idx, odd_idx = ind[0::2], ind[1::2]
    even_ones_fitness = sum(even_idx) + (len(odd_idx) - sum(odd_idx))
    odd_ones_fitness = sum(odd_idx) + (len(even_idx) - sum(even_idx))
    return max(even_ones_fitness, odd_ones_fitness)

def fit_con(ind):
    last_val = 0
    fitness = 0
    for idx, x in enumerate(ind):
        if idx == 0:
            last_val = not x
        if last_val != x:
            fitness += 1
        last_val = x
    return fitness

# implements the whole EA
def evolutionary_algorithm(fitness, cx_prob, mut_prob):
    pop = create_population(POP_SIZE)
    log = []
    for G in range(100):
        fits = list(map(fitness, pop))
        log.append((G, max(fits), sum(fits)/100, G*POP_SIZE))
        #print(G, sum(fits), max(fits)) # prints fitness to console
        mating_pool = selection(pop, fits)
        offspring = operators(mating_pool, cx_prob, mut_prob)
	  #pop = offspring[:-1]+[max(pop, key=fitness)] #SGA + elitism
        pop = offspring[:] #SGA

    return pop, log

# i1, i2 = create_ind(10), create_ind(10)
# print((i1, i2))
# print(cross(i1, i2))
# print(mutate(i1))


def run_experiment(fitness, cx_prob, mut_prob):
    # run the EA 10 times and aggregate the logs, show the last gen in last run
    logs = []
    for i in range(10):
        random.seed(i)
        pop,log = evolutionary_algorithm(fitness, cx_prob, mut_prob)
        logs.append(log)
    # fits = list(map(fitness, pop))
    # pprint.pprint(list(zip(fits, pop)))
    # print(sum(fits), max(fits))
    # pprint.pprint(log)

    # extract fitness evaluations and best fitnesses from logs
    evals = []
    best_fit = []
    for log in logs:
        evals.append([l[3] for l in log])
        best_fit.append([l[1] for l in log])

    evals = np.array(evals)
    best_fit = np.array(best_fit)
    
    return evals, best_fit


def convertTuple(tup):
    st = ''.join(map(str, tup))
    return st

def experiment():
    exps = [(fit_con, 0.8, 0.05), 
    (fit_con, 0.4, 0.05), 
    (fit_con, 0.8, 0.3), 
    (fit_con, 0.4, 0.3)]
    
    #exps = [(fit_plc, 0.8, 0.05),
    # (fit_plc, 0.4, 0.05), 
    # (fit_plc, 0.8, 0.3), 
    # (fit_plc, 0.4, 0.3)] 
    
    for exp in exps:
        evals, best_fit = run_experiment(*exp)

        # plot the converegence graph and quartiles
        plt.plot(evals[0,:], np.median(best_fit, axis=0), label=exp[0].__name__ + ' ' + str(exp[1]) + ' ' + str(exp[2]))
        #plt.fill_between(evals[0,:], np.percentile(best_fit, q=25, axis=0), np.percentile(best_fit, q=75, axis=0), alpha = 0.25)

    plt.title('SGA - Convergence of the fitness value')
    plt.xlabel('Evaluations')
    plt.ylabel('Fitness value')
    plt.legend()
    plt.gcf().canvas.manager.set_window_title('SGA Convergence of the fitness value')
    plt.show()


experiment()