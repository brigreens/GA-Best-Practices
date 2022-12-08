from ast import excepthandler
import numpy as np
import utils
import gzip
import os
import pybel
import subprocess
from rdkit import Chem


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

def parse_sTDA_gzip(filename):
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
    with gzip.open(filename, 'rt', encoding = 'utf-8') as file:
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

def solvation_gzip(filename):
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
    with gzip.open(filename, 'rt', encoding = 'utf-8') as file:
        line = file.readline()
        while line:
            if '-> Gsolv' in line:
                line_list = line.split()
                solvation_energy = float(line_list[3])
                break
                
            line = file.readline()  
        line = file.readline()        

    return solvation_energy

def run_calculations(polymer, scoring_prop, unit_list, run_name, gen_counter):
    run_label = run_name.split('_')[-1]
    # run GFN2 geometry optimization if not already done (needed for all properties)
    filename = utils.make_file_name(polymer)
    GFN2_file = '/ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/output_GFN2/%s.out.gz' % (filename)

    exists = os.path.isfile(GFN2_file)
    if not exists:
        # make polymer into SMILES string
        poly_smiles = utils.make_polymer_smi(polymer, unit_list)

        '''# make polymer string into pybel molecule object
        mol = pybel.readstring('smi', poly_smiles)
        utils.make3D(mol)
        # write polymer .mol file to containing folder
        mol.write('mol', '/ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/input_mol/%s.mol' % (filename), overwrite=True)
        '''
        # make directory to run xtb in for the polymer
        #mkdir_poly = subprocess.call('(mkdir %s_%s_xtb)' % (run_name, filename), shell=True)
        # run xTB geometry optimization
        #xtb = subprocess.call('(cd /ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/input_mol && sbatch -J /ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/input_mol/%s.mol --export=run_label=%s,gen_counter=%s,smiles=%s /ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/GA_cron/GA_run_GFN2.slurm)' %(filename, run_label, gen_counter, poly_smiles), shell=True)
        xtb = subprocess.call('sbatch -J %s --export=run_label=%s,gen_counter=%s /ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/GA_cron/GA_run_GFN2.slurm' %(filename, run_label, gen_counter), shell=True)

        #xtb = subprocess.call('(cd %s_%s_xtb && /ihome/ghutchison/oda6/xtb/xtb-641/bin/xtb ../../input_mol/%s.mol --opt > ../../output_GFN2/%s.out)' % (run_name, filename, filename, filename), shell=True)
        #gzip_xtb = subprocess.call('(gzip ../output_GFN2/%s.out)' % filename, shell=True)
        #save_opt_file = subprocess.call('(cp %s_%s_xtb/xtbopt.mol ../opt_mol/%s_opt.mol)' % (run_name, filename, filename), shell=True)
        # delete xtb run directory for the polymer
        #del_polydir = subprocess.call('(rm -r %s_%s_xtb)' % (run_name, filename), shell=True)
    
    else:
        with open('/ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/generations_%s_polar/%s/%s.txt' %(run_label, gen_counter, filename), 'w')  as fp:
            pass

    if scoring_prop == 'polar':
        # no further calculations needed
        return
    
    elif scoring_prop == 'opt_bg':
        std_file1 = '../output_std_dft_xtb/%s.out.gz' % filename
        std_file2 = '../output_std_dft_xtb/%s.stda.gz' % filename

        exists1 = os.path.isfile(std_file1)
        exists2 = os.path.isfile(std_file2)
        if (not exists1) or (not exists2):
            # create orca input file for sTD-DFT
            subprocess.call('(obabel ../opt_mol/%s_opt.mol -O ../opt_xyz/%s_opt.xyz)' %(filename, filename), shell=True)

            # make directory to run xtb in for the polymer
            mkdir_poly = subprocess.call('(mkdir %s_%s_std)' % (run_name, filename), shell=True)
            # run sTD-DFT-xtb
            stddftxtb = subprocess.call('(export XTB4STDAHOME=/ihome/ghutchison/blp62/xtb4stda && cd %s_%s_std && /ihome/ghutchison/blp62/xtb4stda/bin/xtb4stda ../../opt_xyz/%s_opt.xyz > ../../output_std_dft_xtb/%s.out && /ihome/ghutchison/blp62/xtb4stda/bin/stda -xtb -e 5 -rpa > ../../output_std_dft_xtb/%s.stda)' % (run_name, filename, filename, filename, filename), shell=True)
            gzip_out = subprocess.call('(gzip ../output_std_dft_xtb/%s.out)' % filename, shell=True)
            gzip_stda = subprocess.call('(gzip ../output_std_dft_xtb/%s.stda)' % filename, shell=True)

            # delete xtb run directory for the polymer
            del_polydir = subprocess.call('(rm -r %s_%s_std)' % (run_name, filename), shell=True)
    
    elif scoring_prop == 'solv_eng':
        water_file = '../output_water/%s.out.gz' % filename
        hexane_file = '../output_hexane/%s.out.gz' % filename
        
        exists_water = os.path.isfile(water_file)
        if not exists_water:        
            # make directory to run xtb in for the polymer
            mkdir_poly = subprocess.call('(mkdir %s_%s_water)' % (run_name, filename), shell=True)
            # run xtb for solvation in water
            xtb_water = subprocess.call('(cd %s_%s_water && /ihome/ghutchison/oda6/xtb/xtb-641/bin/xtb ../../opt_mol/%s_opt.mol --sp --alpb water > ../../output_water/%s.out)' % (run_name, filename, filename, filename), shell=True)
            gzip_water = subprocess.call('(gzip ../output_water/%s.out)' % filename, shell=True)
            # delete xtb run directory for the polymer
            del_polydir = subprocess.call('(rm -r %s_%s_water)' % (run_name, filename), shell=True)

        exists_hexane = os.path.isfile(hexane_file)
        if not exists_hexane:
            # make directory to run xtb in for the polymer
            mkdir_poly = subprocess.call('(mkdir %s_%s_hex)' % (run_name, filename), shell=True)
            # run xtb for solvation in hexane
            xtb_hexane = subprocess.call('(cd %s_%s_hex && /ihome/ghutchison/oda6/xtb/xtb-641/bin/xtb ../../opt_mol/%s_opt.mol --sp --alpb hexane > ../../output_hexane/%s.out)' % (run_name, filename, filename, filename), shell=True)
            gzip_hexane = subprocess.call('(gzip ../output_hexane/%s.out)' % filename, shell=True)
            # delete xtb run directory for the polymer
            del_polydir = subprocess.call('(rm -r %s_%s_hex)' % (run_name, filename), shell=True)    
    
    return

def fitness_function(population, scoring_prop, unit_list, run_name, gen_counter):
    """
    Calculates the score of a fitness property and ranks the population

    Parameters
    ----------
    population: list
        list of polymers that each have the format [(seq: #,#,#,#,#,#), monomer_index1, monomer_index2]
    scoring_prop: str
        can be 'polar', 'opt_bg', or 'solv_eng'
    run_name: str
        name of this GA run

    Return
    ------
    ranked_population: nested list
        lists of NFAs and their PCE ranked in order of highest PCE first. Also contains the best donor
        [ranked_NFA_names, ranked_PCE, ranked_best_donor]
    """

    score_list = []

    for x in range(len(population)):
        polymer = population[x]
        filename = utils.make_file_name(polymer)

        # run calculations - this method will only run calculations if output files do not already exist
        # run_calculations(polymer, scoring_prop, unit_list, run_name, gen_counter)

        if scoring_prop == 'polar':
            # parse out polarizability
            try:
                GFN2_file = '/ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/output_GFN2/%s.out.gz' % filename
                GFN2_props = parse_GFN2_gzip(GFN2_file) #dipole_moment, polarizability
                polarizability = GFN2_props[1]
            except:
                print('error with GFN2 file')
                print(filename)
                polarizability = 0
            score_list.append(polarizability)


        elif scoring_prop == 'opt_bg':
            # parse out optical bandgap
            try:
                stda_file = '../output_std_dft_xtb/%s.stda.gz' % filename
                opt_bg = parse_sTDA_gzip(stda_file)
                print(opt_bg)
            except:
                print('error with this sTDA file')
                print(filename)
                opt_bg = 10
            score_list.append(opt_bg)

        elif scoring_prop == 'solv_eng':
            try:
                solv_water_file = '../output_water/%s.out.gz' % filename
                solv_hexane_file = '../output_hexane/%s.out.gz' % filename

                # calculate solvation free energy of acceptor in water
                solv_water = solvation_gzip(solv_water_file)
                # calculate solvation free energy of acceptor in hexane
                solv_hexane = solvation_gzip(solv_hexane_file) 
                # ratio of water solvation energy to hexane solvation energy
                ratio_water_hexane = (solv_water - solv_hexane) / abs(solv_water)

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
        specific order [(#,#,...), A, B]
    scoring_prop: str
        can be 'polar', 'opt_bg', or 'solv_eng'

    Returns
    -------
    Returns the score depending on the property (polarizability, optical bandgap, or solvation ratio)
    '''

    filename = utils.make_file_name(polymer)
    if scoring_prop == 'polar':
        # parse out polarizability
        try:
            GFN2_file = 'output_GFN2/%s.out.gz' % filename
            GFN2_props = parse_GFN2_gzip(GFN2_file) #dipole_moment, polarizability
            polarizability = GFN2_props[1]
        except:
            print('error with GFN2 file')
            print(filename)
            polarizability = 0
        return polarizability

    elif scoring_prop == 'opt_bg':
        # parse out optical bandgap
        try:
            stda_file = 'output_std_dft_xtb/%s.stda.gz' % filename
            opt_bg = parse_sTDA_gzip(stda_file)
        except:
            print('error with this sTDA file')
            print(filename)
            opt_bg = 10
        return opt_bg

    elif scoring_prop == 'solv_eng':
        try:
            solv_water_file = 'output_water/%s.out.gz' % filename
            solv_hexane_file = 'output_hexane/%s.out.gz' % filename

            # calculate solvation free energy of acceptor in water
            solv_water = solvation_gzip(solv_water_file)
            # calculate solvation free energy of acceptor in hexane
            solv_hexane = solvation_gzip(solv_hexane_file) 
            # ratio of water solvation energy to hexane solvation energy
            ratio_water_hexane = (solv_water - solv_hexane) / abs(solv_water)

        except:
            print('error with solvation file')
            print(filename)
            ratio_water_hexane = 100000
        return ratio_water_hexane

    else:
        print('Not a valid scoring property')
        return None




