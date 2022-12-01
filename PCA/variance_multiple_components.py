import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

fps = []
polar_checklist = []
opt_bg_checklist = []
solv_ratio_checklist = []
with open('fps.txt', 'r') as f:
    for line in f:
        fp = []

        characters = line.strip().split()
        len_chars = len(characters)

        chars = list(characters[0])
        fp.append(int(chars[1]))

        for x in range(1, len_chars-4):
            chars = list(characters[x])
            fp.append(int(chars[0]))

        chars = list(characters[-4])
        fp.append(int(chars[0]))

        polar_checklist.append(int(characters[-3]))
        opt_bg_checklist.append(int(characters[-2]))
        solv_ratio_checklist.append(int(characters[-1]))

        fps.append(fp)


sns.set(rc={'figure.figsize': (10, 10)})
sns.set(font_scale=1.5)
sns.set_style('whitegrid')



def evaluate_components(fp_list):
    res = []
    for n_comp in tqdm(range(2,50)):
        pca = PCA(n_components=n_comp)
        crds = pca.fit_transform(fp_list) 
        var = np.sum(pca.explained_variance_ratio_)
        res.append([n_comp,var])
    return res


comp_res = evaluate_components(fps)

res_df = pd.DataFrame(comp_res,columns=["Components","Variance"])
ax = sns.lineplot(data=res_df,x="Components",y="Variance")
plt.savefig('components_vs_variance.png', dpi=600)
plt.savefig('components_vs_variance.pdf', dpi=600)
