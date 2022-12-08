# Imports
import csv
import random
from re import S
import numpy as np
from copy import deepcopy
import pickle
import pandas as pd
from scipy.stats import spearmanr
from itertools import product
import argparse
import os
from rdkit import Chem
import sys
import glob
import subprocess


import utils
import scoring


def main(chem_property, run_label, gen_counter, first_run):
    # reuse initial state
    initial_restart = 'y'
    # flag for restart from save
    restart = 'y'

    # optimized GA hyperparameters
    pop_size = 32
    selection_method = 'tournament_3'
    mutation_rate = 0.4
    elitism_perc = 0.5


    # Spearman coefficient threshold - spearman coefficent must be greater than this value to trip the convergence counter
    spear_thresh = 0.8

    # Convergence threshold - # of consecutive generations that need to meet Spearman coefficient threshold before termination
    conv_gen = 50
    
    # GA run file name, with the format of "parameter_changed parameter_value fitness_property run_label(ABCDE)"
    run_name =  'optimized_GA_' + chem_property + '_' + run_label 

    # Create list of possible building block unit SMILES in specific format
    unit_list = utils.make_unit_list()

    # Create all possible numerical sequences for hexamers with 1 or 2 monomer species
    sequence_list = list(product(range(2), repeat=6))
    # sequence_list = [(0,1,0,1,0,1)]


    # Check for standard output directories & create them if necessary
    directories = ['quick_files', 'full_files', 'mono_freq', 'input_mol', 'opt_mol', 'opt_xyz', 'output_GFN2', 'output_std_dft_xtb', 'output_water', 'output_hexane', 'last_gen_params', 'rand_states', 'generations_A_polar', 'generations_B_polar', 'generations_C_polar', 'generations_D_polar', 'generations_E_polar']
    for dir in directories:
        exists = os.path.isdir('/ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/%s' % dir)
        if not exists:
            os.makedirs('/ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/%s' % dir)

    if restart == 'y':
        # reload parameters and random state from restart file
        last_gen_filename = '/ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/last_gen_params/last_gen_' + run_name + '.p'
        open_params = open(last_gen_filename, 'rb')
        params = pickle.load(open_params)
        open_params.close()

        randstate_filename = '/ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/rand_states/randstate_' + run_name + '.p'
        open_rand = open(randstate_filename, 'rb')
        randstate = pickle.load(open_rand)
        random.setstate(randstate)
        open_rand.close()

    else:
        if initial_restart == 'n':
            # sets initial state
            randstate = random.getstate()
            initial_randstate = '/ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/initial_randstates/initial_randstate_' + run_label +'.p'
            rand_file = open(initial_randstate, 'wb')
            pickle.dump(randstate, rand_file)
            rand_file.close()
        else:
            # re-opens intial state during troubleshooting
            initial_randstate = '/ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/initial_randstates/initial_randstate_' + run_label + '.p'
            open_rand = open(initial_randstate, 'rb')
            randstate = pickle.load(open_rand)
            random.setstate(randstate)
            open_rand.close()

        # run initial generation if NOT loading from restart file
        params = init_gen(pop_size, selection_method, mutation_rate, elitism_perc, run_name, chem_property, unit_list, sequence_list, spear_thresh)

        # pickle parameters needed for restart
        last_gen_filename = '/ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/last_gen_params/last_gen_' + run_name + '.p'
        params_file = open(last_gen_filename, 'wb')
        pickle.dump(params, params_file)
        params_file.close()

        # pickle random state for restart
        randstate = random.getstate()
        randstate_filename = '/ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/rand_states/randstate_' + run_name + '.p'
        rand_file = open(randstate_filename, 'wb')
        pickle.dump(randstate, rand_file)
        rand_file.close()

    # get convergence counter from inital parameters
    spear_counter = params[12]

    # for x in range(1):
    if spear_counter < conv_gen:

        print(first_run)
        params = evaluate_prev_gen(gen_counter, params, first_run, run_name)

        # run next generation of GA
        next_gen(params, gen_counter, run_name)

        # pickle parameters needed for restart
        last_gen_filename = '/ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/last_gen_params/last_gen_' + run_name + '.p'
        params_file = open(last_gen_filename, 'wb')
        pickle.dump(params, params_file)
        params_file.close()

        # pickle random state for restart
        randstate = random.getstate()
        randstate_filename = '/ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/rand_states/randstate_' + run_name + '.p'
        rand_file = open(randstate_filename, 'wb')
        pickle.dump(randstate, rand_file)
        rand_file.close()

    else:
        print('GA has reached convergence')
        sys.exit()

def evaluate_prev_gen(gen_counter, params, first_run, run_name):
    '''
    Evaluate the output files of the previous generation and determine their fitness

    Parameters
    ----------
    gen_counter: int
        current generation

    Returns
    -------
    fitness_list: list
        list of ranked properties from previous generation
    '''
    pop_size = params[0]
    unit_list = params[1]
    sequence_list = params[2]
    fitness_list = params[4]
    selection_method = params[5]
    mutation_rate = params[6]
    elitism_perc = params[7]
    scoring_prop = params[9]
    mono_df = params[10]
    spear_thresh = params[11]
    spear_counter = params[12]

    run_label = run_name.split('_')[-1]

    prev_gen = gen_counter-1

    if first_run == 'True':
        pass

    else:
        print('in else')
        prev_pop_filename = []
        for output in glob.iglob('/ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/generations_%s_polar/%s/*.txt' % (run_label, prev_gen)):
            # split filename to obtain polymer name
            poly_filename = output.split('/')[-1].split('.')[0]
            # append the filenames to the previous gen filename list
            prev_pop_filename.append(poly_filename)

        prev_pop = []
        for x in prev_pop_filename:
            # adds the polymer to the generation, in the format [seq, mon1, mon2]
            prev_pop.append(utils.filename_to_poly(x))

        
        # Calculate fitness of previous generation. fitness_list = [ranked_score, ranked_polymers]
        fitness_list = scoring.fitness_function(prev_pop, scoring_prop, unit_list, run_name, gen_counter) # [ranked_score, ranked_poly_population]

        min_score = fitness_list[0][0]
        median = int((len(fitness_list[0])-1)/2)
        med_score = fitness_list[0][median]
        max_score = fitness_list[0][-1]

        # calculate new monomer frequencies
        old_freq = deepcopy(mono_df)
        mono_df = utils.get_monomer_freq(prev_pop, mono_df, prev_gen)

        # generate monomer frequency csv
        mono_df.to_csv('/ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/mono_freq/%s_monomer_frequency.csv' % run_name)

        # get ranked index arrays for Spearman correlation
        old_ranks = utils.get_ranked_idx(old_freq, prev_gen-1)[:10]
        new_ranks = utils.get_ranked_idx(mono_df, prev_gen)[:10]


        # calculate Spearman correlation coefficient
        spear = spearmanr(old_ranks, new_ranks)[0]

        # keep track of number of successive generations meeting Spearman criterion
        if spear > spear_thresh:
            spear_counter += 1
        else:
            spear_counter = 0

        print('here')
        print(run_name)
        quick_filename = '/ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/quick_files/quick_analysis_' + run_name + '.csv'
        with open(quick_filename, mode='a+') as quick_file:
            # write to quick analysis file
            quick_writer = csv.writer(quick_file)
            quick_writer.writerow([prev_gen, min_score, med_score, max_score, spear, spear_counter])

        for x in range(len(fitness_list[0])):
            poly = fitness_list[1][x]
            filename = utils.make_file_name(poly)

            score = fitness_list[0][x]

            # write full analysis file
            full_filename = '/ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/full_files/full_analysis_' + run_name + '.csv'
            with open(full_filename, mode='a+') as full_file:
                full_writer = csv.writer(full_file)
                full_writer.writerow([prev_gen, filename, score])
    
    params = [pop_size, unit_list, sequence_list, gen_counter, fitness_list, selection_method, mutation_rate, elitism_perc, run_name, scoring_prop, mono_df, spear_thresh, spear_counter]
    
    return params

def next_gen(params, gen_counter, run_name):
    '''
    Runs the next generation of the GA

    Paramaters
    ----------
    params: list
        list with specific order
        params = [pop_size, unit_list, sequence_list, gen_counter, fitness_list, selection_method, mutation_rate, elitism_perc, run_name, scoring_prop, mono_df, spear_thresh, spear_counter]
    
    Returns
    --------
    params: list
        list with specific order
        params = [pop_size, unit_list, sequence_list, gen_counter, fitness_list, selection_method, mutation_rate, elitism_perc, run_name, scoring_prop, mono_df, spear_thresh, spear_counter]
    '''
    pop_size = params[0]
    unit_list = params[1]
    sequence_list = params[2]
    fitness_list = params[4]
    selection_method = params[5]
    mutation_rate = params[6]
    elitism_perc = params[7]
    scoring_prop = params[9]
    mono_df = params[10]
    spear_thresh = params[11]
    spear_counter = params[12]

    run_label = run_name.split('_')[-1]
    # makes a directory in generations directory with current genetation number
    setup = subprocess.call('mkdir /ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/generations_%s_polar/%s' % (run_label, gen_counter), shell=True)

    ranked_population = fitness_list[1]
    ranked_scores = fitness_list[0]

    if scoring_prop == 'polar':
        ranked_population.reverse()
        ranked_scores.reverse()

    # Select percentage of top performers for next genration - "elitism"
    elitist_population = elitist_select(ranked_population, elitism_perc)

    # Selection, Crossover & Mutation - create children to repopulate bottom 50% of NFAs in population
    new_population = select_crossover_mutate(ranked_population, ranked_scores, elitist_population, selection_method, mutation_rate, scoring_prop, pop_size, unit_list, sequence_list)

    print('new_population_len')
    print(len(new_population))
    for poly in new_population:
        fname = utils.make_file_name(poly)
        print(fname)


    for x in new_population:
        scoring.run_calculations(x, scoring_prop, unit_list, run_name, gen_counter)

def parent_select(ranked_population, ranked_scores, selection_method, scoring_prop):
    '''
    Selects two parents. Method of selection depends on selection_method
    
    Parameters
    ----------
    ranked_population: list
        ordered list of polymer names, ranked from best to worst
    ranked_scores: list
        ordered list of scores, ranked from best to worst
    selection_method: str
        Type of selection operation. Options are 'random', 'random_top50', 'tournament', 'roulette', 'SUS', or 'rank'
    scoring_prop: str
        Fitness function property. Options are 'polar', 'opt_bg', 'solv_eng'

    Returns
    -------
    parents: list
        list of length 2 containing the two parent indicies
    '''
    # randomly select parents
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

    # randomly select parents from top 50% of population
    elif selection_method == 'random_top50':
        parents = []
        # randomly select two parents (as indexes from population) to cross
        parent_a = random.randint(0, len(ranked_population)/2 - 1)
        parent_b = random.randint(0, len(ranked_population)/2 - 1)
        # ensure parents are unique indiviudals
        if len(ranked_population) > 1:
            while parent_b == parent_a:
                parent_b = random.randint(0, len(ranked_population)/2 - 1)

        parents.append(ranked_population[parent_a])
        parents.append(ranked_population[parent_b])

    # 3-way tournament selection
    elif selection_method == 'tournament_3':
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

    # 4-way tournament selection
    elif selection_method == 'tournament_4':
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

            # select random individual 4
            individual_4 =  random.randint(0, len(ranked_population) - 1)
            while ((individual_4 == individual_1) or (individual_4 == individual_2) or (individual_4 == individual_3)):
                individual_4 = random.randint(0, len(ranked_population) - 1)
            individuals.append(individual_4)
            # Make list of their fitness
            scores = [ranked_scores[individual_1], ranked_scores[individual_2], ranked_scores[individual_3], ranked_scores[individual_4]]
            
            # find the index of the best fitness score
            if scoring_prop == 'polar':
                best_index = np.argmax(scores)
            else:
                best_index = np.argmin(scores)

            best_individual = individuals[best_index]
            parent = ranked_population[best_individual]

            if parent not in parents:
                parents.append(parent)

    elif selection_method == 'tournament_2':
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

            # Make list of their fitness
            scores = [ranked_scores[individual_1], ranked_scores[individual_2]]
            
            # find the index of the best fitness score
            if scoring_prop == 'polar':
                best_index = np.argmax(scores)
            else:
                best_index = np.argmin(scores)

            best_individual = individuals[best_index]
            parent = ranked_population[best_individual]

            if parent not in parents:
                parents.append(parent)

    elif selection_method == 'roulette': #ranked_population, ranked_scores, selection_method, scoring_prop
        parents = []
        # create wheel 
        wheel = []
        # bottom limit
        limit = 0

        if scoring_prop == 'opt_bg':
            inversed_scores = [1/x for x in ranked_scores]
            total = sum(inversed_scores)

            for x in range(len(inversed_scores)):
                # fitness proportion
                fitness = inversed_scores[x]/total
                # appends the bottom and top limits of the pie, the score, and the polymer
                wheel.append((limit, limit+fitness, inversed_scores[x], ranked_population[x]))
                limit += fitness


        elif scoring_prop == 'polar':
            # sum of scores
            total = sum(ranked_scores)
            for x in range(len(ranked_scores)):
                # fitness proportion
                fitness = ranked_scores[x]/total
                # appends the bottom and top limits of the pie, the score, and the polymer
                wheel.append((limit, limit+fitness, ranked_scores[x], ranked_population[x]))
                limit += fitness
            
        else:
            print('selection method does not work with negative numbers (solvation ratios)')

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
        # bottom limit
        limit = 0

        if scoring_prop == 'opt_bg':
            inversed_scores = [1/x for x in ranked_scores]
            total = sum(inversed_scores)

            for x in range(len(inversed_scores)):
                # fitness proportion
                fitness = inversed_scores[x]/total
                # appends the bottom and top limits of the pie, the score, and the polymer
                wheel.append((limit, limit+fitness, inversed_scores[x], ranked_population[x]))
                limit += fitness


        elif scoring_prop == 'polar':
            # sum of scores
            total = sum(ranked_scores)
            for x in range(len(ranked_scores)):
                # fitness proportion
                fitness = ranked_scores[x]/total
                # appends the bottom and top limits of the pie, the score, and the polymer
                wheel.append((limit, limit+fitness, ranked_scores[x], ranked_population[x]))
                limit += fitness

        else:
            print('selection method does not work with negative numbers (solvation ratios)')
        

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

    return parents


def select_crossover_mutate(ranked_population, ranked_scores, elitist_population, selection_method, mutation_rate, scoring_prop, pop_size, unit_list, sequence_list):
    '''
    Perform selection, crossover, and mutation operations

    Parameters
    ----------
    ranked_population: list
        ordered list containing lists of polymers of format [mon_1_index, mon_2_index]
    ranked_scores: list
        ordered list of scores of population
    elitist_population: list
        list of elite polymers to pass to next generation
    selection_method: str
        Type of selection operation. Options are 'random', 'tournament', 'roulette', 'SUS', or 'rank'
    mutation_rate: int
        Chance of polymer to undergo mutation
    scoring_prop: str
        Fitness function property. Options are 'polar', 'opt_bg', 'solv_eng'
    pop_size: int
        number of individuals in the population
    unit_list: Dataframe
        dataframe containing the SMILES of all monomers
    sequence_list: list
        list of all possible sequences
    
    Returns
    -------
    new_pop: list
        list of new polymers of format [mon_1_index, mon_2_index]
    new_pop_smiles: list
        list of the SMILES of the new polymers
    '''
    
    new_pop = deepcopy(elitist_population)

    # create canonical smiles for new_pop
    new_pop_can = []
    for polymer in new_pop:
        temp_poly_str = utils.make_polymer_smi(polymer, unit_list)
        temp_poly_can = Chem.CanonSmiles(temp_poly_str)
        new_pop_can.append(temp_poly_can)

    # loop until enough children have been added to reach population size
    while len(new_pop) < pop_size:
        # select two parents
        parents = parent_select(ranked_population, ranked_scores, selection_method, scoring_prop)
        
        # create hybrid child
        temp_child = []

        # randomly determine which parent's sequence will be used
        par_seq = random.randint(0, 1)

        # give child appropriate parent's sequence
        temp_child.append(parents[par_seq][0])

        # take first unit from parent 1 and second unit from parent 2
        temp_child.append(parents[0][1])
        temp_child.append(parents[1][2])

        # give child opportunity for mutation
        temp_child = mutate(temp_child, unit_list, sequence_list, mutation_rate)

        # DCH - don't need to order monomers for naming convention
        # temp_child.sort()

        # make canonical SMILES string of polymer
        temp_child_str = utils.make_polymer_smi(
            temp_child, unit_list)

        try:
            temp_child_can = Chem.CanonSmiles(temp_child_str)
        except:
            print('Error creating canonical smiles for molecule:')
            print(temp_child)
            continue

        # check for duplication - use str for comparison to avoid homopolymer, etc. type duplicates
        if temp_child_can in new_pop_can:
            pass
        elif temp_child in new_pop:
            pass
        elif utils.not_valid(temp_child, unit_list) == True:
            pass
        else:
            new_pop.append(temp_child)
            new_pop_can.append(temp_poly_can)       

    return new_pop


def mutate(temp_child, unit_list, sequence_list, mut_rate):
    '''
    Mutation operator. Replaces one monomer with a new one

    Parameters
    -----------
    temp_child: list
        format is [(#,#,...), A, B]
    unit_list: Dataframe
        dataframe containing the SMILES of all monomers
    mut_rate: int
        Chance of polymer to undergo mutation
    
    Return
    ------
    temp_child: list
        format is [(#,#,...), A, B]
    '''

    # determine whether to mutate based on mutation rate
    rand = random.randint(1, 100)
    if rand <= (mut_rate * 100):
        pass
    else:
        return temp_child

    # choose point of mutation
    point = random.randint(0, 2)

    # replace sequence
    if point == 0:
        temp_child[point] = sequence_list[random.randint(
            0, len(sequence_list) - 1)]
    # or replace specific monomer
    else:
        new_mono = random.randint(0, len(unit_list) - 1)
        temp_child[point] = new_mono


    return temp_child


def elitist_select(ranked_population, elitism_perc):
    '''
    Selects a percentage of the top polymers to ensure in next generation
    
    Parameters
    ----------
    ranked_population: list
        ordered list containing lists of polymers of format [mon_1_index, mon_2_index]
    elitism_perc: float
        percentage of generation to pass to next generation. Can range from 0-1 (although 1 is the entire generation)

    Returns
    -------
    elitist_list: list
        list of polymers each of format [mon_1_index, mon_2_index]
    '''

    # find number of parents to pass to next generation
    elitist_count = int(len(ranked_population) * elitism_perc)
    elitist_list = []

    for x in range(elitist_count):
        elitist_list.append(ranked_population[x])

    return elitist_list
    

def init_gen(pop_size, selection_method, mutation_rate, elitism_perc, run_name, scoring_prop, unit_list, sequence_list, spear_thresh):
    '''
    Create initial population

    Parameters
    -----------
    pop_size: int
        number of individuals in the population
    selection_method: str
        Type of selection operation. Options are 'random', 'tournament', 'roulette', 'SUS', or 'rank'
    mutation_rate: int
        Chance of polymer to undergo mutation
    elitism_perc: float
        percentage of generation to pass to next generation. Can range from 0-1 (although 1 is the entire generation)
    run_name: str
        name of this GA run
    scoring_prop: str
        Fitness function property. Options are 'polar', 'opt_bg', 'solv_eng'
    unit_list: Dataframe
        dataframe containing the SMILES of all monomers
    sequence_list: list
        list containing all possible sequences
    spear_thresh: float
        minimum spearman correlation coefficient to trip convergence counter
    
    Returns
    -------
    params: list
        format is [pop_size, unit_list, gen_counter, fitness_list, selection_method, mutation_rate, elitism_perc, run_name, scoring_prop, spear_counter, mono_df]
    '''
    # initialize generation counter
    gen_counter = 1

    # initialize convergence counter
    spear_counter = 0

    # create monomer frequency df [freq_0]
    mono_df = pd.DataFrame(0, index=np.arange(len(unit_list)), columns=np.arange(1))
    mono_df = mono_df.rename_axis('mono_idx')
    mono_df = mono_df.rename(columns={0:'freq_0'})

    # create inital population as list of polymers
    population = []
    population_can = []

    while len(population) < pop_size:
        temp_poly = []

        # select sequence type for polymer
        poly_seq = sequence_list[random.randint(0, len(sequence_list) - 1)]
        temp_poly.append(poly_seq)

        # select monomer types for polymer
        for num in range(2):
            # randomly select a monomer index
            poly_monomer = random.randint(0, len(unit_list) - 1)
            temp_poly.append(poly_monomer)

        # DCH - don't need this
        # sorts the monomer indices in ascending order (for file naming convention)
        # temp_poly.sort()

        # make canonical SMILES string of polymer
        temp_poly_str = utils.make_polymer_smi(
            temp_poly, unit_list)

        try:
            temp_poly_can = Chem.CanonSmiles(temp_poly_str)
        except:
            print('Error creating canonical smiles for molecule:')
            print(temp_poly)
            continue

        # add polymer to population
        # check for duplication - use str for comparison to avoid homopolymer, etc. type duplicates
        if temp_poly_can in population_can:
            continue
        elif utils.not_valid(temp_poly, unit_list) == True:
            continue
        else:
            population.append(temp_poly)
            population_can.append(temp_poly_can)

    # calculate new monomer frequencies
    mono_df = utils.get_monomer_freq(population, mono_df, gen_counter)

    # create new analysis files
    quick_filename = '/ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/quick_files/quick_analysis_' + run_name + '.csv'
    with open(quick_filename, mode='w+') as quick:
        quick_writer = csv.writer(quick)
        quick_writer.writerow(['gen', 'min_score', 'med_score', 'max_score', 'spearman', 'conv_counter'])

    full_filename = '/ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/full_files/full_analysis_' + run_name + '.csv'
    with open(full_filename, mode='w+') as full:
        full_writer = csv.writer(full)
        full_writer.writerow(['gen', 'filename', 'score'])

    fitness_list = scoring.fitness_function(population, scoring_prop, unit_list, run_name, gen_counter) # [ranked_score, ranked_poly_population] 

    min_score = fitness_list[0][0]
    median = int((len(fitness_list[0])-1)/2)
    med_score = fitness_list[0][median]
    max_score = fitness_list[0][-1]

    # inital spearman coefficent for output file
    spear = 0

    quick_filename = '/ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/quick_files/quick_analysis_' + run_name + '.csv'
    with open(quick_filename, mode='a+') as quick_file:
        # write to quick analysis file
        quick_writer = csv.writer(quick_file)
        quick_writer.writerow([1, min_score, med_score, max_score, spear, spear_counter])

    for x in range(len(fitness_list[0])):
        poly = fitness_list[1][x]
        filename = utils.make_file_name(poly)
        score = fitness_list[0][x]

        # write full analysis file
        full_filename = '/ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/full_files/full_analysis_' + run_name + '.csv'
        with open(full_filename, mode='a+') as full_file:
            full_writer = csv.writer(full_file)
            full_writer.writerow([gen_counter, filename, score])

    params = [pop_size, unit_list, sequence_list, gen_counter, fitness_list, selection_method, mutation_rate, elitism_perc, run_name, scoring_prop, mono_df, spear_thresh, spear_counter]
    
    return(params)

def check_prev_gen_complete(gen_counter, run_label, first_run):
    '''
    checks to see if the previous generation is complete. If so, it will run the next generation

    Parameters
    ----------
    gen_counter (int)
        generation to run next
    '''
    
    if gen_counter == 1:
        return True
    elif first_run == 'True':
        return True
    else:
        pop_size = 32

        # checks to see if all of the GFN2 calculations are finished. Counts number of files in directory
        DIR = '/ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/generations_%s_polar/%s' %(run_label, gen_counter)
        num_calcs = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

        if num_calcs == pop_size:
            return True
        else:
            return False


if __name__ == '__main__':
    usage = "usage: %prog [options] "
    parser = argparse.ArgumentParser(usage)

    # sets input arguments
    # 'polar', 'opt_bg', or 'solv_eng'
    parser.add_argument('chem_property', action='store', type=str)
    # 'A', 'B', 'C', 'D', or 'E'
    parser.add_argument('run_label', action='store', type=str)
    # is this the first run using cron? True or False
    parser.add_argument('first_run', action='store', type=str)
    
    args = parser.parse_args()

    # parse what generation we are up to
    with open('/ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/current_gen_polar_%s.txt' %(args.run_label), 'r')  as f:
        prev_gen = f.readline()

    # checks to make sure all of the previous generation's GFN2 calculations are complete
    if check_prev_gen_complete(prev_gen, args.run_label, args.first_run) != True:
        sys.exit()
    # runs next generation
    else:    
        # update the current generation
        current_gen = int(prev_gen) + 1

        main(args.chem_property, args.run_label, current_gen, args.first_run)

        # update the generation file with current generation
        with open('/ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/current_gen_polar_%s.txt' %(args.run_label), 'w') as f:
            f.write(str(current_gen))