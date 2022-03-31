# Imports
import sklearn
import csv
import os
import subprocess
import random
import numpy as np
from scipy import stats
from statistics import mean
from copy import deepcopy
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from itertools import product

import utils
import scoring

def main():
    # reuse initial state
    initial_restart = 'n'
    # flag for restart from save
    restart = 'n'

    # scoring property. Can be 'polar', 'opt_bg', or 'solv_eng'
    scoring_prop = 'polar'

    # number of polymers in population, can be 16, 32, 48, 64, 80, 96
    pop_size = 32
    # selection method. Can be 'random', 'tournament', 'roulette', 'rank', 'SUS'
    selection_method = 'random'
    # mutation rate. Can be 0.1-0.9, in increments of 0.1
    mutation_rate = 0.4
    # elitism percentage. Percentage of top candidates to pass on to next generation. Can be 0, 0.25, 0.5
    elitism_perc = 0.5

    # GA run file name, with the format of "parameter_changed parameter_value fitness_property"
    run_name = 'base' # lets change

    # name for parameter changed where we want the same initial state
    initial_randstate_filename = 'convergence'

    # Create list of possible building block unit SMILES in specific format
    unit_list = utils.make_unit_list()

    if restart == 'y':
        # reload parameters and random state from restart file
        last_gen_filename = 'last_gen_' + run_name + '.p'
        open_params = open(last_gen_filename, 'rb')
        params = pickle.load(open_params)
        open_params.close()

        randstate_filename = 'randstate' + run_name + '.p'
        open_rand = open(randstate_filename, 'rb')
        randstate = pickle.load(open_rand)
        random.setstate(randstate)
        open_rand.close()

    else:
        if initial_restart == 'n':
            # sets initial state
            randstate = random.getstate()
            initial_randstate = 'initial_randstate' + initial_randstate_filename + '.p'
            rand_file = open(initial_randstate, 'wb')
            pickle.dump(randstate, rand_file)
            rand_file.close()
        else:
            # re-opens intial state during troubleshooting
            initial_randstate = 'initial_randstate' + initial_randstate_filename + '.p'
            open_rand = open(initial_randstate, 'rb')
            randstate = pickle.load(open_rand)
            random.setstate(randstate)
            open_rand.close()

        # run initial generation if NOT loading from restart file
        params = init_gen(pop_size, init_type, selection_method, mutation_rate, elitism_perc, run_name, scoring_prop, unit_list)

        # pickle parameters needed for restart
        last_gen_filename = 'last_gen_' + run_name + '.p'
        params_file = open(last_gen_filename, 'wb')
        pickle.dump(params, params_file)
        params_file.close()

        # pickle random state for restart
        randstate = random.getstate()
        randstate_filename = 'randstate' + run_name + '.p'
        rand_file = open(randstate_filename, 'wb')
        pickle.dump(randstate, rand_file)
        rand_file.close()


    for x in range(99):
        # run next generation of GA
        params = next_gen(params)

        # pickle parameters needed for restart
        last_gen_filename = 'last_gen_' + run_name + '.p'
        params_file = open(last_gen_filename, 'wb')
        pickle.dump(params, params_file)
        params_file.close()

        # pickle random state for restart
        randstate = random.getstate()
        randstate_filename = 'randstate' + run_name + '.p'
        rand_file = open(randstate_filename, 'wb')
        pickle.dump(randstate, rand_file)
        rand_file.close()

def next_gen(params):
   # params = [pop_size, unit_list, gen_counter, fitness_list, selection_method, mutation_rate, elitism_perc, run_name, scoring_prop]
    pop_size = params[0]
    unit_list = params[1]
    gen_counter = params[2]
    fitness_list = params[3]
    selection_method = params[4]
    mutation_rate = params[5]
    elitism_perc = params[6]
    run_name = params[7]
    scoring_prop = params[8]

    gen_counter +=1
    ranked_population = fitness_list[2]
    ranked_scores = fitness_list[0]

    if scoring_prop == 'polar':
        ranked_population.reverse()
        ranked_scores.reverse()

    # Select perectnage of top performers for next geenration - "elitism"
    elitist_population = elitist_select(ranked_population, elitism_perc, scoring_prop)

    # Selection, Crossover & Mutation - create children to repopulate bottom 50% of NFAs in population
    new_population, new_population_smiles = select_crossover_mutate(ranked_population, ranked_scores, elitist_population, selection_method, mutation_rate, scoring_prop, pop_size, unit_list)

    fitness_list = scoring.fitness_function(new_population, new_population_smiles, scoring_prop) # [ranked_score, ranked_poly_SMILES, ranked_poly_population]

    min_score = fitness_list[0][0]
    median = int((len(fitness_list[0])-1)/2)
    med_score = fitness_list[0][median]
    max_score = fitness_list[0][-1]

    quick_filename = 'quick_analysis' + run_name + '.csv'
    with open(quick_filename, mode='w+') as quick_file:
        # write to quick analysis file
        quick_writer = csv.writer(quick_file)
        quick_writer.writerow([gen_counter, min_score, med_score, max_score])

    for x in range(len(fitness_list[0])):
        poly = fitness_list[2][x]
        poly_SMILES = fitness_list[1][x]
        score = fitness_list[0][x]

        # write full analysis file
        full_filename = 'full_analysis' + run_name + '.csv'
        with open(full_filename, mode='w+') as full_file:
            full_writer = csv.writer(full_file)
            full_writer.writerow([gen_counter, poly, poly_SMILES, score])

    params = [pop_size, unit_list, gen_counter, fitness_list, selection_method, mutation_rate, elitism_perc, run_name, scoring_prop]
    
    return(params)

def parent_select(ranked_population, ranked_scores, selection_method, scoring_prop):
    if selection_method == 'random':
        parents = []
        # randomly select two parents (as indexes from population) to cross
        parent_a = random.randint(0, len(ranked_population) - 1)
        parent_b = random.randint(0, len(ranked_population) - 1)
        # ensure parents are unique indiviudals
        if len(ranked_population) > 1:
            while parent_b == parent_a:
                parent_b = random.randint(0, len(ranked_population) - 1)

        parents.append(ranked_population[parent_a])
        parents.append(ranked_population[parent_b])

    elif selection_method == 'tournament':
        parents = []
        # select 2 parents
        while len(parents) < 2:
            individuals = []
            # select random individual 1
            individual_1 =  random.randint(0, len(ranked_population) - 1)
            individuals.append(individual_1)

            # select random individual 2
            individual_2 =  random.randint(0, len(ranked_population) - 1)
            while individual_1 == individual_2:
                individual_2 = random.randint(0, len(ranked_population) - 1)
            individuals.append(individual_2)

            # select random individual 3
            individual_3 =  random.randint(0, len(ranked_population) - 1)
            while ((individual_3 == individual_1) or (individual_3 == individual_2)):
                individual_3 = random.randint(0, len(ranked_population) - 1)
            individuals.append(individual_3)
            # Make list of their fitness
            scores = [ranked_scores[individual_1], ranked_scores[individual_2], ranked_scores[individual_3]]
            
            # find the index of the best fitness score
            if scoring_prop == 'polar':
                best_index = np.argmax(scores)
            else:
                best_index = np.argmin(scores)

            best_individual = individuals[best_index]
            parent = ranked_population[best_individual]

            if parent not in parents:
                parents.append(parent)

    elif selection_method == 'roulette':

        '''# sum of scores
        total_fitness = sum(ranked_scores)

        if scoring_prop == 'polar':
            # size of each pie, proportional to its fitness
            relative_fitness = [f/total_fitness for f in ranked_scores]
        else:
            # givers larger pie size to smaller scores
            relative_fitness = [1- f/total_fitness for f in ranked_scores]

        # probabilities of each pie
        probs = [sum(relative_fitness[:i+1]) for i in range(len(relative_fitness))]

        parents = []
        # select 2 parents
        while len(parents) < 2:
            # point on "wheel" where it should stop
            point = random.uniform(0, sum(relative_fitness))
            
            for (i, individual) in enumerate(ranked_population):
                if point <= probs[i]:
                    parent = individual
                    break
                    
            # ensures different parents
            if parent not in parents:
                parents.append(parent)'''


        parents = []

        # create wheel 
        wheel = []
        # sum of scores
        total = sum(ranked_scores)
        # bottom limit
        limit = 0
        for x in range(len(ranked_scores)):
            # fitness proportion
            fitness = ranked_scores[x]/total
            # appends the bottom and top limits of the pie, the score, and the polymer
            wheel.append((limit, limit+fitness, ranked_scores[x], ranked_population[x]))
            limit += fitness

        # random number between 0 and 1
        r = random.random()
        # search for polymer in that pie
        score, polymer = utils.binSearch(wheel, r)
        parents.append(polymer)
        while len(parents) < 2:
            r = random.random()
            score, polymer = utils.binSearch(wheel, r)
            if polymer not in parents:
                parents.append(polymer)
            
    
    elif selection_method == 'SUS': #stochastic universal sampling
        parents = []

        # create wheel 
        wheel = []
        # sum of scores
        total = sum(ranked_scores)
        # bottom limit
        limit = 0
        for x in range(len(ranked_scores)):
            # fitness proportion
            fitness = ranked_scores[x]/total
            # appends the bottom and top limits of the pie, the score, and the polymer
            wheel.append((limit, limit+fitness, ranked_scores[x], ranked_population[x]))
            limit += fitness

        # separation between selected points on wheel
        stepSize = 0.5
        # random number between 0 and 1
        r = random.random()

        # search for polymer in that pie
        score, polymer = utils.binSearch(wheel, r)
        parents.append(polymer)
        while len(parents) < 2:
            r += stepSize
            if r>1:
                r %= 1
            score, polymer = utils.binSearch(wheel, r)
            parents.append(polymer)

    elif selection_method == 'rank':
        parents = []
        # reverse population so worst scores get smallest slice of pie
        ranked_population.reverse()

        # creates wheel 
        wheel = []
        # total sum of the indicies 
        total = sum(range(1, len(ranked_population)+1))
        top = 0
        for p in range(len(ranked_population)):
            # adds 1 so the first element does not have a pie slice of size 0
            x = p+1
            # fraction of total pie
            f = x/total
            wheel.append((top, top+f, ranked_population[p]))
            top += f

        # pick random parents from wheel
        while len(parents) < 2:    
            r = random.random()
            polymer = utils.rank_binSearch(wheel, r)

            if polymer not in parents:
                parents.append(polymer)
    else:
        print('not a valid selection method')



def select_crossover_mutate(ranked_population, ranked_scores, elitist_population, selection_method, mutation_rate, scoring_prop, pop_size, unit_list):
    new_pop = deepcopy(elitist_population)
    new_pop_smiles = []

    # initialize new population with elitists
    for poly in new_pop:
        smiles = utils.make_polymer_smi(poly, unit_list)
        new_pop_smiles.append(smiles)

    # loop until enough children have been added to reach population size
    while len(new_pop) < pop_size:

        # select two parents
        parents = parent_select(ranked_population, ranked_scores, selection_method, scoring_prop)
        
        # create hybrid child
        temp_child = []

        # take first unit from parent 1 and second unit from parent 2
        temp_child.append(parents[0][0])
        temp_child.append(parents[1][1])

        # give child opportunity for mutation
        temp_child = mutate(temp_child, unit_list, mutation_rate)

        # check for duplication
        if temp_child in new_pop:
            pass
        else:
            temp_poly_smi = utils.make_polymer_smi(temp_child, unit_list)
            new_pop_smiles.append(temp_poly_smi)
            new_pop.append(temp_child)       

    return new_pop, new_pop_smiles


def mutate(temp_child, unit_list, mut_rate):

    # determine whether to mutate based on mutation rate
    rand = random.randint(1, 100)
    if rand <= (mut_rate * 100):
        pass
    else:
        return temp_child

   # choose point of mutation
    point = random.randint(0, 1)

    new_mono = random.randint(0, len(unit_list) - 1)
    temp_child[point] = new_mono

    return temp_child

def elitist_select(ranked_population, elitism_perc, scoring_prop):
    # find number of parents to pass to next generation
    elitist_count = int(len(ranked_population) * elitism_perc)
    elitist_list = []

    for x in range(elitist_count):
        elitist_list.append(ranked_population[x])

    return elitist_list
    


def init_gen(pop_size, selection_method, mutation_rate, elitism_perc, run_name, scoring_prop, unit_list):
    # initialize generation counter
    gen_counter = 1

    # create inital population as list of polymers
    population = []
    population_str = []


    while len(population) < pop_size:
        temp_poly = []
        # select monomer types for polymer
        for num in range(2):
            # randomly select a monomer index
            poly_monomer = random.randint(0, len(unit_list) - 1)
            temp_poly.append(poly_monomer)

        # make SMILES string of polymer
        temp_poly_smi = utils.make_polymer_smi(temp_poly, unit_list)

        # checks molecule for errors RDKit would catch
        try:
            # convert to canonical SMILES to check for duplication
            canonical_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(temp_poly_smi))
        except:
            # prevents molecules with incorrect valence, which canonical smiles will catch and throw error
            print(temp_poly_smi)
            print('Incorrect valence, could not perform canonical smiles')
            continue

        # check for duplication
        if canonical_smiles in population_str:
            continue
        else:
            population.append(temp_poly)
            population_str.append(temp_poly_smi)


    # create new analysis files
    quick_filename = 'quick_analysis' + run_name + '.csv'
    with open(quick_filename, mode='w+') as quick:
        quick_writer = csv.writer(quick)
        quick_writer.writerow(['gen', 'min_score', 'med_score', 'max_score'])

    full_filename = 'full_analysis' + run_name + '.csv'
    with open(full_filename, mode='w+') as full:
        full_writer = csv.writer(full)
        full_writer.writerow(['gen', 'filename', 'SMILES', 'score'])

    fitness_list = scoring.fitness_function(population, population_str, scoring_prop) # [ranked_score, ranked_poly_SMILES, ranked_poly_population] 

    min_score = fitness_list[0][0]
    median = int((len(fitness_list[0])-1)/2)
    med_score = fitness_list[0][median]
    max_score = fitness_list[0][-1]

    quick_filename = 'quick_analysis' + run_name + '.csv'
    with open(quick_filename, mode='w+') as quick_file:
        # write to quick analysis file
        quick_writer = csv.writer(quick_file)
        quick_writer.writerow([1, min_score, med_score, max_score])

    for x in range(len(fitness_list[0])):
        poly = fitness_list[2][x]
        poly_SMILES = fitness_list[1][x]
        score = fitness_list[0][x]

        # write full analysis file
        full_filename = 'full_analysis' + run_name + '.csv'
        with open(full_filename, mode='w+') as full_file:
            full_writer = csv.writer(full_file)
            full_writer.writerow([gen_counter, poly, poly_SMILES, score])

    params = [pop_size, unit_list, gen_counter, fitness_list, selection_method, mutation_rate, elitism_perc, run_name, scoring_prop]
    
    return(params)


if __name__ == '__main__':
    main()