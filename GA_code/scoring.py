import numpy as np
import pandas as pd
import sklearn
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit import Chem
from rdkit.Chem import DataStructs
from numpy import linalg
from pickle import load
import os
import utils

def parse_GFN2(filename):
    '''
    Parses through GFN2-xTB output files

    Parameters
    -----------
    filename: str
        path to output file

    Returns
    -------
    outputs: list
        [dipole_moment, polarizability]
    '''
    
    with open(filename, 'r', encoding = 'utf-8') as file:
        line = file.readline()
        while line:
            if 'molecular dipole' in line:
                line = file.readline()
                line = file.readline()
                line = file.readline()
                    
                line_list = line.split()
                dipole_moment = float(line_list[-1])
                
            elif 'Mol. C8AA' in line:
                line = file.readline()
                line_list = line.split()
                
                polarizability = float(line_list[-1])

            line = file.readline()  
        line = file.readline()

        outputs = [dipole_moment, polarizability]
        
        return outputs
    
def parse_sTDA(filename):
    
    with open(filename, 'r', encoding = 'utf-8') as file:
        line = file.readline()
        oscs = []
        energyEV = []
        while line:
            if 'excitation energies, transition moments and TDA amplitudes' in line:
                line = file.readline()
                line = file.readline()
                line_list = line.split()
                while line != '\n':
                    line_list = line.split()
                    oscs.append(float(line_list[3]))
                    energyEV.append(float(line_list[1]))
                    line = file.readline()

            line = file.readline()  
        line = file.readline()

        if len(oscs) != 0:            
            opt_bg = round(energyEV[0], 2)
            
            # Opt bg is the energy of the first transition within the first 12 transition with an oscillator strength greater than 0.5 
            if len(oscs) < 12:
                for i in range(len(oscs)):
                    if  oscs[i] > 0.5:
                        opt_bg = round(energyEV[i], 2)
                        break
            else:
                for x in range(12):
                    if  oscs[x] > 0.5:
                        opt_bg = round(energyEV[x], 2)
                        break

            return opt_bg


def solvation(filename):
    with open(filename, 'r', encoding = 'utf-8') as file:
        line = file.readline()
        while line:
            if '-> Gsolv' in line:
                line_list = line.split()
                solvation_energy = float(line_list[3])
                break
                
            line = file.readline()  
        line = file.readline()        

    return solvation_energy


def fitness_function(population, population_str, scoring_prop):
    """
    Calculates the score of a fitness property and ranks the population

    Parameters
    ----------
    population: list
        list of polymers that each have the format [monomer_index1, monomer_index2]
    population_str: list
        list of SMILES of each polymer
    scoring_prop: str
        can be 'polar', 'opt_bg', or 'solv_eng'

    Return
    ------
    ranked_population: nested list
        lists of NFAs and their PCE ranked in order of highest PCE first. Also contains the best donor
        [ranked_NFA_names, ranked_PCE, ranked_best_donor]
    """

    score_list = []

    for x in range(len(population)):
        filename = utils.make_file_name(population[x])
        if scoring_prop == 'polar':
            GFN2_file = '../calculations/GFN2/' + filename + '.out'
            GFN2_props = parse_GFN2(GFN2_file) #dipole_moment, polarizability
            polarizability = GFN2_props[1]
            score_list.append(polarizability)
            

        elif scoring_prop == 'opt_bg':
            stda_file = '../calculations/sTDDFTxtb/' + filename + '.stda'
            opt_bg = parse_sTDA(stda_file)
            score_list.append(opt_bg)

        elif scoring_prop == 'solv_eng':
            solv_water_file = '../calculations/solvation_water/' + filename + '.out'
            solv_hexane_file = '../calculations/solvation_hexane/' + filename + '.out'

            # calculate solvation free energy of acceptor in water
            solv_water = solvation(solv_water_file)
            # calculate solvation free energy of acceptor in hexane
            solv_hexane = solvation(solv_hexane_file) 
            # ratio of water solvation energy to hexane solvation energy
            ratio_water_hexane = solv_water / solv_hexane
            score_list.append(ratio_water_hexane)
        else:
            print('Not a valid scoring property')


    ranked_score = []
    ranked_poly_SMILES = []
    ranked_poly_population = []

    # make list of indicies of polymers in population, sorted based on score
    ranked_indices = list(np.argsort(score_list))

    #if scoring_prop == 'polar':
        # reverse list so highest property value = 0th
        #ranked_indices.reverse()

    for x in ranked_indices:
        ranked_score.append(score_list[x])
        ranked_poly_SMILES.append(population_str[x])
        ranked_poly_population.append(population[x])

    ranked_population = [ranked_score, ranked_poly_SMILES, ranked_poly_population]

    return ranked_population


    



