import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


def morgan_fp(smiles, mon1, mon2):
    try:
        mol = Chem.MolFromSmiles(smiles)
        fp = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
    except:
        print('Error with polymer. Monomers are ' + str(mon1) + ' and ' + str(mon2))
        return False
    
    return fp

def make_hex_smiles(mon1_count, mon2_count, df_smiles):
    mon1_smiles = df_smiles.iloc[mon1_count][1]
    mon2_smiles = df_smiles.iloc[mon2_count][1]
    
    SMILES = mon1_smiles + mon2_smiles + mon1_smiles + mon2_smiles + mon1_smiles + mon2_smiles
    
    return SMILES

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
        [polarizability, HOMO, LUMO]
    '''
    
    with open(filename, 'r', encoding = 'utf-8') as file:
        line = file.readline()
        while line:
            if 'Mol. C8AA' in line:
                line = file.readline()
                line_list = line.split()
                
                polarizability = float(line_list[-1])
            
            elif 'Orbital Energies and Occupations' in line:
                line = file.readline()
                while 'HOMO' not in line:
                    line = file.readline()
                
                line_list = line.split()
                HOMO = float(line_list[-2])
                line = file.readline()
                line_list = line.split()
                LUMO = float(line_list[-2])

            line = file.readline()  
        line = file.readline()

        outputs = [polarizability, HOMO, LUMO]
        
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
            '''if 'ordered frontier orbitals' in line:
                for x in range(12):
                    line = file.readline()

                line_list = line.split()
                HOMO = float(line_list[1])
                
                line = file.readline()
                line = file.readline()
                line_list = line.split()
                LUMO = float(line_list[1])'''
            
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
            opt_bg = 0
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

# imports all monomer SMILES into DataFrame
df_smiles = pd.read_csv('monomer_SMILES_PCA.csv')
polar_df = pd.read_csv('/ihome/ghutchison/dch45/ga_methods/hexamer_pool/data_files/ranked_polar.csv')
top_polar_df = polar_df[:100]

opt_bg_df = pd.read_csv('/ihome/ghutchison/dch45/ga_methods/hexamer_pool/data_files/ranked_opt_bg.csv')
top_opt_bg_df = opt_bg_df[:100]

solv_eng_df = pd.read_csv('/ihome/ghutchison/dch45/ga_methods/hexamer_pool/data_files/ranked_solv_eng.csv')
top_solv_eng_df = solv_eng_df[:100]

# total number of monomers
num_monos = 447

# Start at monomer index 0
mon1 = 0

fps = []
bad_fps_count = 0

# checks to see if monomer is in top 100 
polar_checklist = []
opt_bg_checklist = []
solv_ratio_checklist = []

# iterate through all combinations of monomers
while mon1 < num_monos:
    mon2 = mon1
    while mon2 < num_monos:
        hex_smiles = make_hex_smiles(mon1, mon2, df_smiles)
        fp = morgan_fp(hex_smiles, mon1, mon2)
        if fp == False:
            bad_fps_count +=1
        else:
            monomers = [mon1, mon2]
            for mon in monomers:
                polar, HOMO, LUMO = parse_GFN2('/ihome/ghutchison/blp62/GA_best_practices/PCA/monomer_calcs/GFN2_output/%s.out' % mon)
                optbg = parse_sTDA('/ihome/ghutchison/blp62/GA_best_practices/PCA/monomer_calcs/sTDDFTxtb/%s.stda' % mon)
                solv_hex = solvation('/ihome/ghutchison/blp62/GA_best_practices/PCA/monomer_calcs/solvation_hexane/%s.out' %mon)
                solv_water = solvation('/ihome/ghutchison/blp62/GA_best_practices/PCA/monomer_calcs/solvation_water/%s.out' % mon)

                fp.extend([HOMO, LUMO, polar, optbg, solv_hex, solv_water])

            fps.append(fp)

            monomers = str([mon1, mon2])
            if monomers in top_polar_df.values:
                polar_checklist.append(1)
            else:
                polar_checklist.append(0)

            if monomers in top_opt_bg_df.values:
                opt_bg_checklist.append(1)
            else:
                opt_bg_checklist.append(0)

            if monomers in top_solv_eng_df.values:
                solv_ratio_checklist.append(1)
            else:
                solv_ratio_checklist.append(0)
        
        mon2 +=1
    mon1 +=1

print(len(fps))


# creates a PCA object to generate a lower dimensional projection of the data
pca = PCA(n_components=50)
# generates a set of two coordinates
crds = pca.fit_transform(fps)

# use TSNE
crds_embedded = TSNE(n_components=2).fit_transform(crds)

# create dataframe with coordinates
tsne_df = pd.DataFrame(crds_embedded,columns=["X","Y"])


# add whether each molecule is in the top 100 for each chemical property
tsne_df['top_polar'] = polar_checklist
tsne_df['top_optbg'] = opt_bg_checklist
tsne_df['top_solv'] = solv_ratio_checklist
tsne_df.head()

# polarizabilitiy
fig, ax1 = plt.subplots()
sns.scatterplot(data=tsne_df.query("top_polar == 0"),x="X",y="Y", color='lightgrey', ax=ax1)
sns.scatterplot(data=tsne_df.query("top_polar == 1"),x="X",y="Y", color = 'red', ax=ax1)
ax1.grid(False)
plt.tight_layout()

plt.savefig('tsne_polar_chem_props.pdf', dpi=600)
plt.savefig('tsne_polar_chem_props.png', dpi=600)

# Optical bandgap
fig, ax2 = plt.subplots()
sns.scatterplot(data=tsne_df.query("top_optbg == 0"),x="X",y="Y", color='lightgrey', ax=ax2)
sns.scatterplot(data=tsne_df.query("top_optbg == 1"),x="X",y="Y", color = 'green', ax=ax2)
ax2.grid(False)
plt.tight_layout()

plt.savefig('tsne_optbg_chem_props.pdf', dpi=600)
plt.savefig('tsne_optbg_chem_props.png', dpi=600)

# Solvation Energy
fig, ax3 = plt.subplots()
sns.scatterplot(data=tsne_df.query("top_solv == 0"),x="X",y="Y", color='lightgrey', ax=ax3)
sns.scatterplot(data=tsne_df.query("top_solv == 1"),x="X",y="Y", color = 'dodgerblue', ax=ax3)
ax3.grid(False)
plt.tight_layout()

plt.savefig('tsne_solv_eng_chem_props.pdf', dpi=600)
plt.savefig('tsne_solv_eng_chem_props.png', dpi=600)

    
tsne_df.to_csv('tnse_coords_chem_props.csv')