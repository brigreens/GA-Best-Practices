import pandas as pd
import pybel
from rdkit import Chem
import sys

def make_unit_list():
    '''
    Makes a dataframe containing the monomer SMILES
    Returns
    -------
    units: dataframe
        contains 1 column of monomer SMILES
    '''
    units = pd.read_csv('/ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/updated_monomer_list.csv', index_col=False)

    return units


def make_file_name(polymer):
    '''
    Makes file name for a given polymer

    Parameters
    ---------
    polymer: list (specific format)
        [(#,#,...), A, B]

    Returns
    -------
    file_name: str
        polymer file name (w/o extension) showing monomer indicies and full sequence
        e.g. 100_200_101010 for a certain hexamer
    '''

    # capture monomer indexes as strings for file naming
    mono1 = str(polymer[1])
    mono2 = str(polymer[2])

    # make string of actual length sequence
    seq = list(polymer[0])
    seq = ''.join(map(str, seq))

    # make file name string
    file_name = '%s_%s_%s' % (mono1, mono2, seq)

    return file_name


def make_polymer_smi(polymer, unit_list):
    '''
    Constructs polymer string from monomers

    Parameters
    ---------
    polymer: list (specific format)
        [(#,#,...), A, B]
    unit_list: list
        list of all possible monomer SMILES
    
    Returns
    -------
    poly_smiles: str
        polymer SMILES string
    '''
    poly_smiles = ''

    # cycle over monomer sequence until total number of monomers in polymer is reached
    for i in range(len(polymer[0])):
        # get monomer identifier from sequence
        seq_monomer_index = polymer[0][i]
        
        # find monomer index in smiles list and get monomer smiles
        monomer_index = polymer[seq_monomer_index + 1]
        poly_smiles = poly_smiles + unit_list.iloc[monomer_index][0]

    return poly_smiles

def make3D(mol):
    '''
    Makes the mol object from SMILES 3D

    Parameters
    ---------
    mol: object
        pybel molecule object
    '''
    # make mol object 3D and add hydrogens
    pybel._builder.Build(mol.OBMol)
    mol.addh()

    ff = pybel._forcefields["mmff94"]
    success = ff.Setup(mol.OBMol)
    if not success:
        ff = pybel._forcefields["uff"]
        success = ff.Setup(mol.OBMol)
        if not success:
            sys.exit("Cannot set up forcefield")

    ff.ConjugateGradients(100, 1.0e-3)
    ff.WeightedRotorSearch(100, 25)
    ff.ConjugateGradients(250, 1.0e-4)

    ff.GetCoordinates(mol.OBMol)


def binSearch(wheel, num):
    '''
    Finds what pie (or individual) in a wheel the number belongs in. Works with the SUS selection method

    Parameters
    ----------
    wheel: list
        contains list of lists of format [lower_limit, upper_limit, ranked_scores, ranked_population]
    num: float
        random number between 0 and 1
    
    Returns
    -------
    score: float
        fitness score of bin
    polymer: list
        [mon_1_index, mon_2_index]
    '''
    mid = len(wheel)//2
    low, high, score, polymer = wheel[mid]
    if low<=num<=high:
        return score, polymer
    elif high < num:
        return binSearch(wheel[mid+1:], num)
    else:
        return binSearch(wheel[:mid], num)

def rank_binSearch(wheel, num):
    '''
    Finds what pie in a wheel the number belongs in. Works with the rank selection method

    Parameters
    ----------
    wheel: list
        contains list of lists of format [lower_limit, upper_limit, ranked_scores, ranked_population]
    num: float
        random number between 0 and 1
    
    Returns
    -------
    score: float
        fitness score of bin
    polymer: list
        [mon_1_index, mon_2_index]
    '''
    mid = len(wheel)//2
    low, high, polymer = wheel[mid]
    if low<=num<=high:
        return polymer
    elif high < num:
        return rank_binSearch(wheel[mid+1:], num)
    else:
        return rank_binSearch(wheel[:mid], num)


# note: population is list of lists [[mono1,mono2], [mono1,mono2], ...]
def get_monomer_freq(population, mono_df, gen):
   
    # create new freq column of 0's
    new_col = 'freq_%d' % gen
    old_col = 'freq_%d' % (gen-1)
    mono_df[new_col] = mono_df[old_col].copy()
   
    # loop through population and count monomer frequency
    for polymer in population:
        for monomer in polymer:
            mono_df.loc[monomer,new_col] = mono_df.loc[monomer,new_col]+1
    
    return(mono_df)

def get_ranked_idx(mono_df, gen):
    
    col = 'freq_%d' % gen
    
    ranked_idx = mono_df[col].copy()
    ranked_idx = ranked_idx.sort_values(ascending=False)
    
    # get ranked indexes as ndarray
    ranked_idx = ranked_idx.index.values
    
    return(ranked_idx)

def not_valid(temp_child, unit_list):

    # make polymer into SMILES string
    poly_smiles = make_polymer_smi(temp_child, unit_list)

    # test smiles validity with RDKit
    try:
        m = Chem.MolFromSmiles(poly_smiles)
        Chem.MolToSmiles(m)
    except:
        print('SMILES to mol error')
        print(poly_smiles)
        print(temp_child)
        return True

    return False

def filename_to_poly(filename):
    split_filename = filename.split('_')
    mon1 = int(split_filename[0])
    mon2 = int(split_filename[1])
    seq = list(split_filename[2])

    # converts a string to a tuple
    for x in range(len(seq)):
        seq[x] = int(seq[x])

    seq_tuple = tuple(seq)

    poly = [seq_tuple, mon1, mon2]

    return poly