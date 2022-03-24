
from copy import deepcopy
from itertools import product
import pandas as pd
import numpy as np
from itertools import product


def make_unit_list():
    '''
    Makes a list of dataframe of the SMILES of different types of building block units
    Contains: smi_filename, smiles, sTDDFTxtb_HOMO, calibrated B3LYP_HOMO, A_or_D, fused_ring_count

    Returns
    -------
    units: list
        list of dataframes containing SMILES, filenames, HOMO, A_or_D, and fused ring count
        [left_terminals, fused_cores, right_terminals, spacers]
    '''
    # TODO: add correct csv containing monomer unit SMILES
    units = pd.read_csv('GA4_donor_units.csv', index_col=0)

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

    # make file name string
    file_name = '%s_%s_010101' % (mono1, mono2)

    return file_name



def make_polymer_smi(temp_poly, unit_list):
    '''
    Constructs polymer string from monomers with sequence ABABAB

    Parameters
    ---------
    polymer: list (specific format)
        [A, B]
    smiles_list: list
        list of all possible monomer SMILES

    Returns
    -------
    poly_string: str
        polymer SMILES string
    '''
    poly_smiles = ''

    # cycle over monomer sequence until total number of monomers in polymer is reached
    for i in range(3):
        poly_smiles = poly_smiles + unit_list.iloc[temp_poly[0]][0] + unit_list.iloc[temp_poly[1]][0]

    return poly_smiles
    
def binSearch(wheel, num):
    '''
    Finds what pie in a wheel the number belongs in. Works with the SUS selection method
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
    Finds what pie in a wheel the number belongs in. Works with the SUS selection method
    '''
    mid = len(wheel)//2
    low, high, polymer = wheel[mid]
    if low<=num<=high:
        return polymer
    elif high < num:
        return binSearch(wheel[mid+1:], num)
    else:
        return binSearch(wheel[:mid], num)