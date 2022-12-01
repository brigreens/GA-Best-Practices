import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import umap.umap_ as umap

fps_props_df = pd.read_csv('fps_chem_props.csv')
fps_df = fps_props_df.iloc[:, 0:-12]

fps = fps_df.values.tolist()
fps_props = fps_props_df.values.tolist()

top100_df = pd.read_csv('is_in_top100_polar_optbg_solv.csv')



# UMAP with 2 components and just ECFP
umap_model = umap.UMAP(metric = "jaccard",
                      n_neighbors = 25,
                      n_components = 2,
                      low_memory = False,
                      min_dist = 0.001)
X_umap = umap_model.fit_transform(fps)

# create dataframe with coordinates
umap_df = pd.DataFrame()

umap_df["X"], umap_df["Y"] = X_umap[:,0], X_umap[:,1]

# add whether each molecule is in the top 100 for each chemical property
umap_df['top_polar'] = top100_df['polar'].values.tolist()
umap_df['top_optbg'] = top100_df['optbg'].values.tolist()
umap_df['top_solv'] = top100_df['solv_eng'].values.tolist()
umap_df.head()

# polarizabilitiy
fig, ax1 = plt.subplots()
sns.scatterplot(data=umap_df.query("top_polar == 0"),x="X",y="Y", color='lightgrey', ax=ax1)
sns.scatterplot(data=umap_df.query("top_polar == 1"),x="X",y="Y", color = 'red', ax=ax1)
ax1.grid(False)
plt.tight_layout()

plt.savefig('umap_polar.pdf', dpi=600)
plt.savefig('umap_polar.png', dpi=600)

# Optical bandgap
fig, ax2 = plt.subplots()
sns.scatterplot(data=umap_df.query("top_optbg == 0"),x="X",y="Y", color='lightgrey', ax=ax2)
sns.scatterplot(data=umap_df.query("top_optbg == 1"),x="X",y="Y", color = 'green', ax=ax2)
ax2.grid(False)
plt.tight_layout()

plt.savefig('umap_optbg.pdf', dpi=600)
plt.savefig('umap_optbg.png', dpi=600)

# Solvation Energy
fig, ax3 = plt.subplots()
sns.scatterplot(data=umap_df.query("top_solv == 0"),x="X",y="Y", color='lightgrey', ax=ax3)
sns.scatterplot(data=umap_df.query("top_solv == 1"),x="X",y="Y", color = 'dodgerblue', ax=ax3)
ax3.grid(False)
plt.tight_layout()

plt.savefig('umap_solv_eng.pdf', dpi=600)
plt.savefig('umap_solv_eng.png', dpi=600)

    


# UMAP with 2 components and ECFP and chemical descriptors
umap_model = umap.UMAP(metric = "jaccard",
                      n_neighbors = 25,
                      n_components = 2,
                      low_memory = False,
                      min_dist = 0.001)
X_umap = umap_model.fit_transform(fps_props)

# create dataframe with coordinates
umap_df = pd.DataFrame()

umap_df["X"], umap_df["Y"] = X_umap[:,0], X_umap[:,1]

# add whether each molecule is in the top 100 for each chemical property
umap_df['top_polar'] = top100_df['polar'].values.tolist()
umap_df['top_optbg'] = top100_df['optbg'].values.tolist()
umap_df['top_solv'] = top100_df['solv_eng'].values.tolist()
umap_df.head()

# polarizabilitiy
fig, ax1 = plt.subplots()
sns.scatterplot(data=umap_df.query("top_polar == 0"),x="X",y="Y", color='lightgrey', ax=ax1)
sns.scatterplot(data=umap_df.query("top_polar == 1"),x="X",y="Y", color = 'red', ax=ax1)
ax1.grid(False)
plt.tight_layout()

plt.savefig('umap_polar_chem_props.pdf', dpi=600)
plt.savefig('umap_polar_chem_props.png', dpi=600)

# Optical bandgap
fig, ax2 = plt.subplots()
sns.scatterplot(data=umap_df.query("top_optbg == 0"),x="X",y="Y", color='lightgrey', ax=ax2)
sns.scatterplot(data=umap_df.query("top_optbg == 1"),x="X",y="Y", color = 'green', ax=ax2)
ax2.grid(False)
plt.tight_layout()

plt.savefig('umap_optbg_chem_props.pdf', dpi=600)
plt.savefig('umap_optbg_chem_props.png', dpi=600)

# Solvation Energy
fig, ax3 = plt.subplots()
sns.scatterplot(data=umap_df.query("top_solv == 0"),x="X",y="Y", color='lightgrey', ax=ax3)
sns.scatterplot(data=umap_df.query("top_solv == 1"),x="X",y="Y", color = 'dodgerblue', ax=ax3)
ax3.grid(False)
plt.tight_layout()

plt.savefig('umap_solv_eng_chem_props.pdf', dpi=600)
plt.savefig('umap_solv_eng_chem_props.png', dpi=600)