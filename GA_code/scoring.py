import numpy as np
import utils
import gzip

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

def parse_GFN2_gzip(filename):
    '''
    Parses through gzipped GFN2-xTB output files

    Parameters
    -----------
    filename: str
        path to output file

    Returns
    -------
    outputs: list
        [dipole_moment, polarizability]
    '''
    
    with gzip.open(filename, 'rt') as file:
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
    '''
    Parses through sTD-DFT-xTB output files

    Parameters
    -----------
    filename: str
        path to output file

    Returns
    -------
    opt_bg: float
        optical bandgap 
        energy of the first transition within the first 12 transition with an oscillator strength greater than 0.5 
    '''
    with open(filename, 'r', encoding = 'utf-8') as file:
        line = file.readline()
        oscs = []
        energyEV = []
        potential_no_absoroption = False
        while line:
            if 'excitation energy, transition moments and TDA amplitudes' in line:
                line = file.readline()
                line = file.readline()
                line = file.readline()
                line_list = line.split()
                while line != '\n':
                    line_list = line.split()
                    oscs.append(float(line_list[3]))
                    energyEV.append(float(line_list[1]))
                    line = file.readline()
            elif 'excitation energies, transition moments and TDA amplitudes' in line:
                line = file.readline()
                line = file.readline()
                line_list = line.split()
                while line != '\n':
                    line_list = line.split()
                    oscs.append(float(line_list[3]))
                    energyEV.append(float(line_list[1]))
                    line = file.readline()

            elif '0 CSF included by energy' in line:
                potential_no_absoroption = True

            line = file.readline()  
        line = file.readline()

        if len(oscs) != 0:            
            opt_bg = round(energyEV[0], 4)
            
            # Opt bg is the energy of the first transition within the first 12 transition with an oscillator strength greater than 0.5 
            if len(oscs) < 12:
                for i in range(len(oscs)):
                    if  oscs[i] > 0.5:
                        opt_bg = round(energyEV[i], 4)
                        break
            else:
                for x in range(12):
                    if  oscs[x] > 0.5:
                        opt_bg = round(energyEV[x], 4)
                        break

            return opt_bg
        else:
            if potential_no_absoroption == True:
                print('no absorption below 5 eV')
            else:
                print('error with file')
            print(filename)
            opt_bg = 10
            return opt_bg


def solvation(filename):
    '''
    Parses through xTB output files for solvation energy

    Parameters
    -----------
    filename: str
        path to output file

    Returns
    -------
    solvation_energy: float
        solvation energy 
    '''
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


def fitness_function(population, scoring_prop):
    """
    Calculates the score of a fitness property and ranks the population

    Parameters
    ----------
    population: list
        list of polymers that each have the format [monomer_index1, monomer_index2]
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
            try:
                GFN2_file = '../Calculations/GFN2/' + filename + '.out'
                GFN2_props = parse_GFN2(GFN2_file) #dipole_moment, polarizability
                polarizability = GFN2_props[1]
            except:
                GFN2_file = '../Calculations/GFN2/' + filename + '.out.gz'
                GFN2_props = parse_GFN2_gzip(GFN2_file) #dipole_moment, polarizability
                polarizability = GFN2_props[1]
            score_list.append(polarizability)
            

        elif scoring_prop == 'opt_bg':
            try:
                stda_file = '../Calculations/sTDDFTxtb/' + filename + '.stda'
                opt_bg = parse_sTDA(stda_file)
            except:
                print('error with this sTDA file')
                print(filename)
                opt_bg = 10
            score_list.append(opt_bg)

        elif scoring_prop == 'solv_eng':
            solv_water_file = '../Calculations/solvation_water/' + filename + '.out'
            solv_hexane_file = '../Calculations/solvation_hexane/' + filename + '.out'

            try:
                # calculate solvation free energy of acceptor in water
                solv_water = solvation(solv_water_file)
                # calculate solvation free energy of acceptor in hexane
                solv_hexane = solvation(solv_hexane_file) 
                # ratio of water solvation energy to hexane solvation energy
                if solv_hexane <= 0:
                    ratio_water_hexane = 100000
                else:
                    ratio_water_hexane = solv_water / solv_hexane
            except: 
                print('error with solvation file')
                print(filename)
                ratio_water_hexane = 100000
            score_list.append(ratio_water_hexane)
        else:
            print('Not a valid scoring property')


    ranked_score = []
    ranked_poly_population = []

    # make list of indicies of polymers in population, sorted based on score
    ranked_indices = list(np.argsort(score_list))

    #if scoring_prop == 'polar':
        # reverse list so highest property value = 0th
        #ranked_indices.reverse()

    for x in ranked_indices:
        ranked_score.append(score_list[x])
        ranked_poly_population.append(population[x])

    ranked_population = [ranked_score, ranked_poly_population]

    return ranked_population


def fitness_individual(polymer, scoring_prop):
    '''
    Returns the fitness score of an individual 
    
    Parameters
    ----------
    polymer: list
        specific order [monomer 1 index, monomer 2 index]
    scoring_prop: str
        can be 'polar', 'opt_bg', or 'solv_eng'

    Returns
    -------
    Returns the score depending on the property (polarizability, optical bandgap, or solvation ratio)
    '''

    filename = utils.make_file_name(polymer)
    if scoring_prop == 'polar':
        GFN2_file = '../Calculations/GFN2/' + filename + '.out'
        GFN2_props = parse_GFN2(GFN2_file) #dipole_moment, polarizability
        polarizability = GFN2_props[1]
        return polarizability

    elif scoring_prop == 'opt_bg':
        stda_file = '../Calculations/sTDDFTxtb/' + filename + '.stda'
        opt_bg = parse_sTDA(stda_file)
        return opt_bg

    elif scoring_prop == 'solv_eng':
        solv_water_file = '../Calculations/solvation_water/' + filename + '.out'
        solv_hexane_file = '../Calculations/solvation_hexane/' + filename + '.out'

        # calculate solvation free energy of acceptor in water
        solv_water = solvation(solv_water_file)
        # calculate solvation free energy of acceptor in hexane
        solv_hexane = solvation(solv_hexane_file) 
        # ratio of water solvation energy to hexane solvation energy
        ratio_water_hexane = solv_water / solv_hexane

        return ratio_water_hexane
    else:
        print('Not a valid scoring property')
        return None




