import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# laod csv
df_faith = pd.read_csv('ds_sucks.csv')
df_rich  = pd.read_csv('richness_scores_all_models.csv')

# get the col for pcs
faith_cols = [
    'h_bleu', 'h_rouge', 'h_bert',
    'h_cosine', 'h_entail', 'h_novelty', 'h_lora'
]

rich_cols = [
    'ttr_diff', 'density_diff', 'adj_ratio_diff',
    'noun_ratio_diff', 'verb_ratio_diff', 'ner_diff'
]

# standardisation before pca (fauthfulness)
scaler_f = StandardScaler()
X_f = scaler_f.fit_transform(df_faith[faith_cols].dropna())
pca_f = PCA(n_components=1)
pca_f.fit(X_f)

loadings_f = pd.Series(
    pca_f.components_[0],
    index=faith_cols
).abs().sort_values(ascending=False)

print("Faithfulness PC1 loadings (abs):")
print(loadings_f)

# standardisation before pca (richness)
scaler_r = StandardScaler()
X_r = scaler_r.fit_transform(df_rich[rich_cols].dropna())
pca_r = PCA(n_components=1)
pca_r.fit(X_r)

loadings_r = pd.Series(
    pca_r.components_[0],
    index=rich_cols
).abs().sort_values(ascending=False)

print("\nRichness PC1 loadings (abs):")
print(loadings_r)