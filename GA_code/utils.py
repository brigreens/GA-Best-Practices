import pandas as pd

def make_unit_list():
    '''
    Makes a dataframe containing the monomer SMILES
    Returns
    -------
    units: dataframe
        contains 1 column of monomer SMILES
    '''
    units = pd.read_csv('../monomer_SMILES.csv', index_col=False)

    return units


def make_file_name(polymer):
    '''
    Makes file name for a given polymer

    Parameters
    ---------
    polymer: list (specific format)
        [A, B]

    Returns
    -------
    file_name: str
        polymer file name (w/o extension) showing monomer indicies and full sequence
        e.g. 100_200_101010 for a certain hexamer
    '''

    # capture monomer indexes as strings for file naming
    mono1 = str(polymer[0])
    mono2 = str(polymer[1])

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
    unit_list: dataframe
        contains all monomer SMILES

    Returns
    -------
    poly_smiles: str
        polymer SMILES string
    '''
    poly_smiles = ''

    # cycle over monomer sequence until total number of monomers in polymer is reached
    for i in range(3):
        poly_smiles = poly_smiles + unit_list.iloc[temp_poly[0]][0] + unit_list.iloc[temp_poly[1]][0]

    return poly_smiles
    
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
    ranked_idx = ranked_idx.index.to_numpy()
    
    return(ranked_idx)