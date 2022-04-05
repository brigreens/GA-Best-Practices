# Imports
import csv
import subprocess
import random
import numpy as np
from copy import deepcopy
import pickle
from rdkit import Chem
import math

import utils
import scoring

def main():
    # reuse initial state
    initial_restart = 'n'
    # flag for restart from save
    restart = 'n'
    # Run number (for use with same initial states), can be A, B, C, D, or E
    run_label = 'A'

    # number of islands, try 5, 10, 20
    num_islands = 10

    # number of polymers on each island. Try 10, 20, 32, 50, 100. Papers usually have sub-pop size = num islands
    island_pop_size = 10

    # Percentage of each island to swap with other island. Cannot be 50% or higher
    migration_rate = 30

    # Number of generations to pass before migration. Can try 1, 5, 10, 20
    migration_interval = 5

    # topology of islands, can try connected_network, elite_island, and ring
    topology = 'connected_network'

    # scoring property. Can be 'polar', 'opt_bg', or 'solv_eng'
    scoring_prop = 'polar'


    # selection method. Can be 'random', 'tournament', 'roulette', 'rank', 'SUS'
    selection_method = 'random'
    # mutation rate. Can be 0.1-0.9, in increments of 0.1
    mutation_rate = 0.4
    # elitism percentage. Percentage of top candidates to pass on to next generation. Can be 0, 0.25, 0.5
    elitism_perc = 0.5

    # GA run file name, with the format of "parameter_changed parameter_value fitness_property run_label(ABCDE)"
    run_name = 'base_' + run_label # lets change

    # name for parameter changed where we want the same initial state
    initial_randstate_filename = 'convergence'

    # Create list of possible building block unit SMILES in specific format
    unit_list = utils.make_unit_list()

    if restart == 'y':
        # reload parameters and random state from restart file
        last_gen_filename = '../last_gen_params/last_gen_' + run_name + '.p'
        open_params = open(last_gen_filename, 'rb')
        params = pickle.load(open_params)
        open_params.close()

        randstate_filename = '../rand_states/randstate_' + run_name + '.p'
        open_rand = open(randstate_filename, 'rb')
        randstate = pickle.load(open_rand)
        random.setstate(randstate)
        open_rand.close()

    else:
        if initial_restart == 'n':
            # sets initial state
            randstate = random.getstate()
            initial_randstate = '../initial_randstates/initial_randstate_' + run_label +'.p'
            rand_file = open(initial_randstate, 'wb')
            pickle.dump(randstate, rand_file)
            rand_file.close()
        else:
            # re-opens intial state during troubleshooting
            initial_randstate = '../initial_randstates/initial_randstate_' + run_label + '.p'
            open_rand = open(initial_randstate, 'rb')
            randstate = pickle.load(open_rand)
            random.setstate(randstate)
            open_rand.close()

        # run initial generation if NOT loading from restart file
        params = init_gen(island_pop_size, num_islands, migration_rate, migration_interval,topology,  scoring_prop, selection_method, mutation_rate,  elitism_perc ,run_name, unit_list)

        # pickle parameters needed for restart
        last_gen_filename = '../last_gen_params/last_gen_' + run_name + '.p'
        params_file = open(last_gen_filename, 'wb')
        pickle.dump(params, params_file)
        params_file.close()

        # pickle random state for restart
        randstate = random.getstate()
        randstate_filename = '../rand_states/randstate_' + run_name + '.p'
        rand_file = open(randstate_filename, 'wb')
        pickle.dump(randstate, rand_file)
        rand_file.close()


    for x in range(99):
        if x % migration_interval == 0:
            params = migration(params)
        else:
            # run next generation of GA
            params = next_gen(params)

        # pickle parameters needed for restart
        last_gen_filename = '../last_gen_params/last_gen_' + run_name + '.p'
        params_file = open(last_gen_filename, 'wb')
        pickle.dump(params, params_file)
        params_file.close()

        # pickle random state for restart
        randstate = random.getstate()
        randstate_filename = '../rand_states/randstate' + run_name + '.p'
        rand_file = open(randstate_filename, 'wb')
        pickle.dump(randstate, rand_file)
        rand_file.close()


def migration(params):
    '''
    Performs migration among the islands

    Parameters
    ----------
    params: list
        list with specific order
        params = [unit_list, gen_counter, island_fitness_lists, island_pop_size, num_islands, migration_rate, migration_interval,topology,  scoring_prop, selection_method, mutation_rate,  elitism_perc ,run_name]
    
    Returns
    -------
    params: list
        list with specific order
        params = [unit_list, gen_counter, ranked_island_fitness_lists, island_pop_size, num_islands, migration_rate, migration_interval,topology,  scoring_prop, selection_method, mutation_rate,  elitism_perc ,run_name]

    '''
    # params = [unit_list, gen_counter, island_fitness_lists, island_pop_size, num_islands, migration_rate, migration_interval,topology,  scoring_prop, selection_method, mutation_rate,  elitism_perc ,run_name]
    unit_list = params[0]
    gen_counter = params[1]
    island_fitness_lists = params[2]
    island_pop_size = params[3]
    num_islands = params[4]
    migration_rate = params[5]
    migration_interval = params[6]
    topology = params[7]
    scoring_prop = params[8]
    selection_method = params[9]
    mutation_rate = params[10]
    elitism_perc = params[11]
    run_name = params[12]

    new_island_fitness_lists = deepcopy(island_fitness_lists)

    # number of individuals to migrate
    num_to_migrate =  math.ceil(migration_rate/100 * island_pop_size)

    if topology == 'elite_island':
    # central elite island that accepts the local best from each island and distributes the global best to other islands
        # deleting the individuals to be replaced
        for x in range(1, num_islands):
            if scoring_prop != 'polar':
                new_island_fitness_lists[x][2] = new_island_fitness_lists[x][2][:-1]
                new_island_fitness_lists[x][1]= new_island_fitness_lists[x][1][:-1]
                new_island_fitness_lists[x][0]= new_island_fitness_lists[x][0][:-1]
            else:
                new_island_fitness_lists[x][2] = new_island_fitness_lists[x][2][1:]
                new_island_fitness_lists[x][1]= new_island_fitness_lists[x][1][1:]
                new_island_fitness_lists[x][0]= new_island_fitness_lists[x][0][1:]
                
        if scoring_prop != 'polar':
                new_island_fitness_lists[0][2] = new_island_fitness_lists[0][2][:-(num_islands-1)]
                new_island_fitness_lists[0][1]= new_island_fitness_lists[0][1][:-(num_islands-1)]
                new_island_fitness_lists[0][0]= new_island_fitness_lists[0][0][:-(num_islands-1)]
        else:
            new_island_fitness_lists[0][2] = new_island_fitness_lists[0][2][(num_islands-1):]
            new_island_fitness_lists[0][1]= new_island_fitness_lists[0][1][(num_islands-1):]
            new_island_fitness_lists[0][0]= new_island_fitness_lists[0][0][(num_islands-1):]
    
        for x in range(1, num_islands):
            # move local best individual to elite island
            # polarazibality scoring function is maximizing score, so list is backwards
            if scoring_prop != 'polar':
                # defining the migrant polymer to be moved
                migrant_poly = island_fitness_lists[x][2][0]
                migrant_SMILES = island_fitness_lists[x][1][0]
                migrant_score = island_fitness_lists[x][0][0]
            else:
                # defining the migrant polymer to be moved
                migrant_poly = island_fitness_lists[x][2][-1]
                migrant_SMILES = island_fitness_lists[x][1][-1]
                migrant_score = island_fitness_lists[x][0][-1]

            # replaces worst individuals in next neighboring island
            new_island_fitness_lists[0][2].append(migrant_poly)
            new_island_fitness_lists[0][1].append(migrant_SMILES)
            new_island_fitness_lists[0][0].append(migrant_score)
            
            # move global best to each island
            if scoring_prop != 'polar':
                # defining the migrant polymer to be moved
                migrant_poly = island_fitness_lists[0][2][0]
                migrant_SMILES = island_fitness_lists[0][1][0]
                migrant_score = island_fitness_lists[0][0][0]
            else:
                # defining the migrant polymer to be moved
                migrant_poly = island_fitness_lists[0][2][-1]
                migrant_SMILES = island_fitness_lists[0][1][-1]
                migrant_score = island_fitness_lists[0][0][-1]

            # replaces worst individuals in next neighboring island
            new_island_fitness_lists[x][2].append(migrant_poly)
            new_island_fitness_lists[x][1].append(migrant_SMILES)
            new_island_fitness_lists[x][0].append(migrant_score)


    
    elif topology == 'connect_network':
        # islands to transfer to elite island are island 0 and random island
        random_island = random.randint(1, num_islands-4)

        # deleting the individuals to be replaced
        for x in range(num_islands):
            if scoring_prop != 'polar':
                new_island_fitness_lists[x][2] = new_island_fitness_lists[x][2][:-num_to_migrate]
                new_island_fitness_lists[x][1]= new_island_fitness_lists[x][1][:-num_to_migrate]
                new_island_fitness_lists[x][0]= new_island_fitness_lists[x][0][:-num_to_migrate]
            else:
                new_island_fitness_lists[x][2] = new_island_fitness_lists[x][2][num_to_migrate:]
                new_island_fitness_lists[x][1]= new_island_fitness_lists[x][1][num_to_migrate:]
                new_island_fitness_lists[x][0]= new_island_fitness_lists[x][0][num_to_migrate:]


        # num_islands except 2 will create circle. Remaining 2 are "elites"
        for x in range(0, num_islands-2):
            # loop through the number of migrants
            for i in range(num_to_migrate):
                # island 0 pass to island 1
                if x == 0:
                    # polarazibality scoring function is maximizing score, so list is backwards
                    if scoring_prop != 'polar':
                        # defining the migrant polymer to be moved
                        migrant_poly = island_fitness_lists[x][2][i]
                        migrant_SMILES = island_fitness_lists[x][1][i]
                        migrant_score = island_fitness_lists[x][0][i]

                    else:
                        # defining the migrant polymer to be moved
                        migrant_poly = island_fitness_lists[x][2][-1-i]
                        migrant_SMILES = island_fitness_lists[x][1][-1-i]
                        migrant_score = island_fitness_lists[x][0][-1-i]
                    # replaces worst individuals in next neighboring island
                    new_island_fitness_lists[x+1][2].append(migrant_poly)
                    new_island_fitness_lists[x+1][1].append(migrant_SMILES)
                    new_island_fitness_lists[x+1][0].append(migrant_score)

                    new_island_fitness_lists[num_islands-1][2].append(migrant_poly)
                    new_island_fitness_lists[num_islands-1][1].append(migrant_SMILES)
                    new_island_fitness_lists[num_islands-1][0].append(migrant_score)
                        
                elif x == num_islands-3:
                    if scoring_prop != 'polar':
                        # defining the migrant polymer to be moved
                        migrant_poly = island_fitness_lists[x][2][i]
                        migrant_SMILES = island_fitness_lists[x][1][i]
                        migrant_score = island_fitness_lists[x][0][i]

                    else:
                        # defining the migrant polymer to be moved
                        migrant_poly = island_fitness_lists[x][2][-1-i]
                        migrant_SMILES = island_fitness_lists[x][1][-1-i]
                        migrant_score = island_fitness_lists[x][0][-1-i]
                        # replaces worst individuals in next neighboring island
                    new_island_fitness_lists[0][2].append(migrant_poly)
                    new_island_fitness_lists[0][1].append(migrant_SMILES)
                    new_island_fitness_lists[0][0].append(migrant_score)
                        
                elif x == random_island:
                    if scoring_prop != 'polar':
                        # defining the migrant polymer to be moved
                        migrant_poly = island_fitness_lists[x][2][i]
                        migrant_SMILES = island_fitness_lists[x][1][i]
                        migrant_score = island_fitness_lists[x][0][i]

                    else:
                        # defining the migrant polymer to be moved
                        migrant_poly = island_fitness_lists[x][2][-1-i]
                        migrant_SMILES = island_fitness_lists[x][1][-1-i]
                        migrant_score = island_fitness_lists[x][0][-1-i]
                    # replaces worst individuals in next neighboring island
                    new_island_fitness_lists[x+1][2].append(migrant_poly)
                    new_island_fitness_lists[x+1][1].append(migrant_SMILES)
                    new_island_fitness_lists[x+1][0].append(migrant_score)

                    new_island_fitness_lists[num_islands-2][2].append(migrant_poly)
                    new_island_fitness_lists[num_islands-2][1].append(migrant_SMILES)
                    new_island_fitness_lists[num_islands-2][0].append(migrant_score)
                else:
                    if scoring_prop != 'polar':
                        # defining the migrant polymer to be moved
                        migrant_poly = island_fitness_lists[x][2][i]
                        migrant_SMILES = island_fitness_lists[x][1][i]
                        migrant_score = island_fitness_lists[x][0][i]
                    else:
                        # defining the migrant polymer to be moved
                        migrant_poly = island_fitness_lists[x][2][-1-i]
                        migrant_SMILES = island_fitness_lists[x][1][-1-i]
                        migrant_score = island_fitness_lists[x][0][-1-i]
                    # replaces worst individuals in next neighboring island
                    new_island_fitness_lists[x+1][2].append(migrant_poly)
                    new_island_fitness_lists[x+1][1].append(migrant_SMILES)
                    new_island_fitness_lists[x+1][0].append(migrant_score)
           
    elif topology == 'ring':
        # ring of islands where each island can exchange with both neighbors

        # deleting the individuals to be replaced
        for x in range(num_islands):
            if scoring_prop != 'polar':
                new_island_fitness_lists[x][2] = new_island_fitness_lists[x][2][:-num_to_migrate*2]
                new_island_fitness_lists[x][1]= new_island_fitness_lists[x][1][:-num_to_migrate*2]
                new_island_fitness_lists[x][0]= new_island_fitness_lists[x][0][:-num_to_migrate*2]
            else:
                new_island_fitness_lists[x][2] = new_island_fitness_lists[x][2][num_to_migrate*2:]
                new_island_fitness_lists[x][1]= new_island_fitness_lists[x][1][num_to_migrate*2:]
                new_island_fitness_lists[x][0]= new_island_fitness_lists[x][0][num_to_migrate*2:]

        # loop through each island except the last one
        for x in range(1, num_islands-1):
            # loop through the number of migrants
            for i in range(num_to_migrate):
                # polarazibality scoring function is maximizing score, so list is backwards
                if scoring_prop != 'polar':
                    # defining the migrant polymer to be moved
                    migrant_poly = island_fitness_lists[x][2][i]
                    migrant_SMILES = island_fitness_lists[x][1][i]
                    migrant_score = island_fitness_lists[x][0][i]
                else:
                    # defining the migrant polymer to be moved
                    migrant_poly = island_fitness_lists[x][2][-1-i]
                    migrant_SMILES = island_fitness_lists[x][1][-1-i]
                    migrant_score = island_fitness_lists[x][0][-1-i]
                # replaces worst individuals in neighboring island 1
                new_island_fitness_lists[x+1][2].append(migrant_poly)
                new_island_fitness_lists[x+1][1].append(migrant_SMILES)
                new_island_fitness_lists[x+1][0].append(migrant_score)
                # replaces worst individuals in neighboring island 2
                new_island_fitness_lists[x-1][2].append(migrant_poly)
                new_island_fitness_lists[x-1][1].append(migrant_SMILES)
                new_island_fitness_lists[x-1][0].append(migrant_score)
        # completing the ring
        for i in range(num_to_migrate): 
            if scoring_prop != 'polar':
                # defining the migrant polymer to be moved from the last island
                migrant_poly = island_fitness_lists[-1][2][i]
                migrant_SMILES = island_fitness_lists[-1][1][i]
                migrant_score = island_fitness_lists[-1][0][i]
                # replaces worst individuals in the first island
                new_island_fitness_lists[0][2].append(migrant_poly)
                new_island_fitness_lists[0][1].append(migrant_SMILES)
                new_island_fitness_lists[0][0].append(migrant_score)
                # replaces worst individuals in the second to last island
                new_island_fitness_lists[-2][2].append(migrant_poly)
                new_island_fitness_lists[-2][1].append(migrant_SMILES)
                new_island_fitness_lists[-2][0].append(migrant_score)
            
                # defining the migrant polymer to be moved from the first island
                migrant_poly = island_fitness_lists[0][2][i]
                migrant_SMILES = island_fitness_lists[0][1][i]
                migrant_score = island_fitness_lists[0][0][i]

                # replaces worst individuals in the second island
                new_island_fitness_lists[1][2].append(migrant_poly)
                new_island_fitness_lists[1][1].append(migrant_SMILES)
                new_island_fitness_lists[1][0].append(migrant_score)
                # replaces worst individuals in the last island
                new_island_fitness_lists[-1][2].append(migrant_poly)
                new_island_fitness_lists[-1][1].append(migrant_SMILES)
                new_island_fitness_lists[-1][0].append(migrant_score)
            
            else:
                # defining the migrant polymer to be moved from the last island
                migrant_poly = island_fitness_lists[-1][2][-1-i]
                migrant_SMILES = island_fitness_lists[-1][1][-1-i]
                migrant_score = island_fitness_lists[-1][0][-1-i]
                # replaces worst individuals in the first island
                new_island_fitness_lists[0][2].append(migrant_poly)
                new_island_fitness_lists[0][1].append(migrant_SMILES)
                new_island_fitness_lists[0][0].append(migrant_score)
                # replaces worst individuals in the second to last island
                new_island_fitness_lists[-2][2].append(migrant_poly)
                new_island_fitness_lists[-2][1].append(migrant_SMILES)
                new_island_fitness_lists[-2][0].append(migrant_score)

                # defining the migrant polymer to be moved from the first island
                migrant_poly = island_fitness_lists[0][2][-1-i]
                migrant_SMILES = island_fitness_lists[0][1][-1-i]
                migrant_score = island_fitness_lists[0][0][-1-i]

                # replaces worst individuals in the second island
                new_island_fitness_lists[1][2].append(migrant_poly)
                new_island_fitness_lists[1][1].append(migrant_SMILES)
                new_island_fitness_lists[1][0].append(migrant_score)

                # replaces worst individuals in the last island
                new_island_fitness_lists[-1][2].append(migrant_poly)
                new_island_fitness_lists[-1][1].append(migrant_SMILES)
                new_island_fitness_lists[-1][0].append(migrant_score)
                
    else:
        print("not a valid topology")
        new_island_fitness_lists = 0
        
    # re-order the lists based on scores, in ascending order
    ranked_island_fitness_lists = []
    for i in new_island_fitness_lists:
        ranked_score = []
        ranked_poly_SMILES = []
        ranked_poly_population = []

        # make list of indicies of polymers in population, sorted based on score
        ranked_indices = list(np.argsort(i[0]))

        for x in ranked_indices:
            ranked_score.append(i[0][x])
            ranked_poly_SMILES.append(i[1][x])
            ranked_poly_population.append(i[2][x])

        ranked_population = [ranked_score, ranked_poly_SMILES, ranked_poly_population]
        ranked_island_fitness_lists.append(ranked_population)

        # update csv files with new populations
        quick_filename = '../MIGA/' + run_name + '/quick_files/quick_analysis_island_' + str(i) + '.csv'
        with open(quick_filename, mode='w+') as quick_file:
            # write to quick analysis file
            quick_writer = csv.writer(quick_file)
            quick_writer.writerow(['migration', ranked_score[0], ranked_score[len(ranked_score)/2], ranked_score[-1]])

        for x in range(len(ranked_score)):
            poly = ranked_score[x]
            poly_SMILES = ranked_poly_SMILES[x]
            score = ranked_score[x]

            # write full analysis file
            full_filename = '../MIGA/' + run_name + '/full_files/full_analysis_island_' + str(i) + '.csv'
            with open(full_filename, mode='w+') as full_file:
                full_writer = csv.writer(full_file)
                full_writer.writerow(['migration', poly, poly_SMILES, score])
                                                        
    params = [unit_list, gen_counter, ranked_island_fitness_lists, island_pop_size, num_islands, migration_rate, migration_interval,topology,  scoring_prop, selection_method, mutation_rate,  elitism_perc ,run_name]
    return params

        

def next_gen(params):
    '''
    Runs the next generation of the GA

    Parameters
    ----------
    params: list
        list with specific order
        params = [unit_list, gen_counter, island_fitness_lists, island_pop_size, num_islands, migration_rate, migration_interval,topology,  scoring_prop, selection_method, mutation_rate,  elitism_perc ,run_name]
    
    Returns
    -------
    params: list
        list with specific order
        params = [unit_list, gen_counter, ranked_island_fitness_lists, island_pop_size, num_islands, migration_rate, migration_interval,topology,  scoring_prop, selection_method, mutation_rate,  elitism_perc ,run_name]
    '''
    unit_list = params[0]
    gen_counter = params[1]
    island_fitness_lists = params[2]
    island_pop_size = params[3]
    num_islands = params[4]
    migration_rate = params[5]
    migration_interval = params[6]
    topology = params[7]
    scoring_prop = params[8]
    selection_method = params[9]
    mutation_rate = params[10]
    elitism_perc = params[11]
    run_name = params[12]

    gen_counter +=1
    ranked_island_fitness_lists = []

    for island in range(num_islands):

        ranked_population = island_fitness_lists[island][2]
        ranked_scores = fitness_list[island][0]

        if scoring_prop == 'polar':
            ranked_population.reverse()
            ranked_scores.reverse()

        # Select perectnage of top performers for next geenration - "elitism"
        elitist_population = elitist_select(ranked_population, elitism_perc, scoring_prop)

        # Selection, Crossover & Mutation - create children to repopulate bottom 50% of NFAs in population
        new_population, new_population_smiles = select_crossover_mutate(ranked_population, ranked_scores, elitist_population, selection_method, mutation_rate, scoring_prop, island_pop_size, unit_list)

        fitness_list = scoring.fitness_function(new_population, new_population_smiles, scoring_prop) # [ranked_score, ranked_poly_SMILES, ranked_poly_population]
        ranked_island_fitness_lists.append(fitness_list)

        min_score = fitness_list[0][0]
        median = int((len(fitness_list[0])-1)/2)
        med_score = fitness_list[0][median]
        max_score = fitness_list[0][-1]

        quick_filename = '../MIGA/' + run_name + '/quick_files/quick_analysis_island_' + str(island) + '.csv'
        with open(quick_filename, mode='w+') as quick_file:
            # write to quick analysis file
            quick_writer = csv.writer(quick_file)
            quick_writer.writerow([gen_counter, min_score, med_score, max_score])

        for x in range(len(fitness_list[0])):
            poly = fitness_list[2][x]
            poly_SMILES = fitness_list[1][x]
            score = fitness_list[0][x]

            # write full analysis file
            full_filename = '../MIGA/' + run_name + '/full_files/full_analysis_island_' + str(island) + '.csv'
            with open(full_filename, mode='w+') as full_file:
                full_writer = csv.writer(full_file)
                full_writer.writerow([gen_counter, poly, poly_SMILES, score])

    params = [unit_list, gen_counter, ranked_island_fitness_lists, island_pop_size, num_islands, migration_rate, migration_interval,topology,  scoring_prop, selection_method, mutation_rate,  elitism_perc ,run_name]
    
    return(params)

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
        Type of selection operation. Options are 'random', 'tournament', 'roulette', 'SUS', or 'rank'
    scoring_prop: str
        Fitness function property. Options are 'polar', 'opt_bg', 'solv_eng'

    Returns
    -------
    parents: list
        list of length 2 containing the two parent indicies
    '''
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
    
    Returns
    -------
    new_pop: list
        list of new polymers of format [mon_1_index, mon_2_index]
    new_pop_smiles: list
        list of the SMILES of the new polymers
    '''
    
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
    '''
    Mutation operator. Replaces one monomer with a new one

    Parameters
    -----------
    temp_child: list
        format is [mon_1_index, mon_2_index]
    unit_list: Dataframe
        dataframe containing the SMILES of all monomers
    mut_rate: int
        Chance of polymer to undergo mutation
    
    Return
    ------
    temp_child: list
        format is [mon_1_index, mon_2_index]
    '''
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
    


def init_gen(island_pop_size, num_islands, migration_rate, migration_interval,topology, scoring_prop, selection_method, mutation_rate,  elitism_perc ,run_name, unit_list):
    '''
    Create initial population

    Parameters
    -----------
    island_pop_size: int
        number of individuals in the island
    num_islands: int
        number of islands
    migration_rate: int
        percentage of individuals of each island to migrate
    migration_interval: int
        number of generations increment before migration
    topology: str
        arrangement of islands. Options are 'connected_network', 'elite_island', and 'ring'
    scoring_prop: str
        Fitness function property. Options are 'polar', 'opt_bg', 'solv_eng'
    selection_method: str
        Type of selection operation. Options are 'random', 'tournament', 'roulette', 'SUS', or 'rank'
    mutation_rate: int
        Chance of polymer to undergo mutation
    elitism_perc: float
        percentage of generation to pass to next generation. Can range from 0-1 (although 1 is the entire generation)
    run_name: str
        name of this GA run
    unit_list: Dataframe
        dataframe containing the SMILES of all monomers
    
    Returns
    -------
    params: list
        format is [unit_list, gen_counter, island_fitness_lists, island_pop_size, num_islands, migration_rate, migration_interval,topology,  scoring_prop, selection_method, mutation_rate,  elitism_perc ,run_name]

    '''
    
    
    # initialize generation counter
    gen_counter = 1

    # create inital population as list of polymers
    population = []
    population_str = []

    total_pop_size = island_pop_size * num_islands

    while len(population) < total_pop_size:
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

    # split into islands
    for x in range(0, len(total_pop_size, island_pop_size)):
        island_pops = population[x:x+island_pop_size]
        island_pops_smiles = population_str[x:x+island_pop_size]

    # create directories and files to track evolution
    setup = subprocess.call('mkdir ../%s' % (run_name), shell=True)
    setup_files = subprocess.call('mkdir ../%s/quick_files ../%s/full_files' % (run_name, run_name), shell=True)

    # lists of all islands scored populations
    island_fitness_lists = []
    for island in range(len(island_pops)):
        # create new analysis files
        quick_filename = '../MIGA/' + run_name + '/quick_files/quick_analysis_island_' + str(island) + '.csv'
        with open(quick_filename, mode='w+') as quick:
            quick_writer = csv.writer(quick)
            quick_writer.writerow(['gen', 'min_score', 'med_score', 'max_score'])

        full_filename = '../MIGA/' + run_name + '/full_files/full_analysis_island_' + str(island) + '.csv'
        with open(full_filename, mode='w+') as full:
            full_writer = csv.writer(full)
            full_writer.writerow(['gen', 'filename', 'SMILES', 'score'])


        fitness_list = scoring.fitness_function(island_pops[x], island_pops_smiles[x], scoring_prop) # [ranked_score, ranked_poly_SMILES, ranked_poly_population] 
        island_fitness_lists.append(fitness_list)

        min_score = fitness_list[0][0]
        median = int((len(fitness_list[0])-1)/2)
        med_score = fitness_list[0][median]
        max_score = fitness_list[0][-1]

        quick_filename = '../MIGA/' + run_name + '/quick_files/quick_analysis_island_' + str(island) + '.csv'
        with open(quick_filename, mode='w+') as quick_file:
            # write to quick analysis file
            quick_writer = csv.writer(quick_file)
            quick_writer.writerow([1, min_score, med_score, max_score])

        for x in range(len(fitness_list[0])):
            poly = fitness_list[2][x]
            poly_SMILES = fitness_list[1][x]
            score = fitness_list[0][x]

            # write full analysis file
            full_filename = '../MIGA/' + run_name + '/full_files/full_analysis_island_' + str(island) + '.csv'
            with open(full_filename, mode='w+') as full_file:
                full_writer = csv.writer(full_file)
                full_writer.writerow([gen_counter, poly, poly_SMILES, score])



    params = [unit_list, gen_counter, island_fitness_lists, island_pop_size, num_islands, migration_rate, migration_interval,topology,  scoring_prop, selection_method, mutation_rate,  elitism_perc ,run_name]
    
    return(params)


if __name__ == '__main__':
    main()