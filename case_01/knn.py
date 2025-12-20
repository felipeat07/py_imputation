import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
from collections import Counter
import random

# =========================================================
# Configuração das bases
# =========================================================

DATASETS = {
    'glass': {
        'path': 'data/glass_identification/glass.data',
        'columns': ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type_of_glass'],
        'target': 'Type_of_glass',
        'drop': ['Id']
    },
    'wine': {
        'path': 'data/wine/wine.data',
        'columns': ['Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity',
                    'Magnesium', 'Total_phenols', 'Flavanoids',
                    'Nonflavanoid_phenols', 'Proanthocyanins',
                    'Color_intensity', 'Hue', 'OD280_OD315', 'Proline'],
        'target': 'Class',
        'drop': []
    },
    'letter': {
        'path': 'data/letter_recognition/letter-recognition.data',
        'columns': ['Letter', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7',
                    'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16'],
        'target': 'Letter',
        'drop': []
    }
}

# =========================================================
# Funções auxiliares
# =========================================================

def eta_squared(x, y):
    classes = y.dropna().unique()
    grand_mean = x[y.notna()].mean()
    ss_between = sum(
        len(x[y == cl]) * ((x[y == cl].mean() - grand_mean) ** 2)
        for cl in classes
    )
    ss_total = sum((x[y.notna()] - grand_mean) ** 2)
    return ss_between / ss_total if ss_total != 0 else 0


def gower_distance_matrix(X):
    ranges = X.max() - X.min()
    X_norm = (X - X.min()) / ranges
    n = X.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        dist_matrix[i] = np.abs(X_norm.values[i] - X_norm.values).mean(axis=1)
    return dist_matrix


def knn_impute(row_idx, df_weighted, df_target, k):
    row = df_weighted.iloc[row_idx].values.reshape(1, -1)
    df_known = df_weighted[df_target.notna()]
    target_known = df_target[df_target.notna()]
    distances = cdist(row, df_known.values, metric='euclidean').flatten()
    nearest_idx = df_known.index[np.argsort(distances)[:k]]
    labels = target_known.loc[nearest_idx].dropna()
    return labels.mode().iloc[0]


def knn_impute_simple(row_idx, df_scaled, df_target, k):
    row = df_scaled.iloc[row_idx].values.reshape(1, -1)
    df_known = df_scaled[df_target.notna()]
    target_known = df_target[df_target.notna()]
    distances = cdist(row, df_known.values, metric='euclidean').flatten()
    nearest_idx = df_known.index[np.argsort(distances)[:k]]
    labels = target_known.loc[nearest_idx].dropna()
    return labels.mode().iloc[0]


def knn_impute_gower(row_idx, gower_matrix, df_target, k):
    distances = gower_matrix[row_idx]
    known_mask = df_target.notna().values
    distances_known = distances[known_mask]
    indices_known = np.where(known_mask)[0]
    nearest = indices_known[np.argsort(distances_known)[:k]]
    labels = df_target.iloc[nearest].dropna()
    return labels.mode().iloc[0]

# =========================================================
# Pipeline de experimento (genérica)
# =========================================================

def run_experiment(cfg, seed, missing_frac=0.2, k=5):

    # Carregar dados
    df = pd.read_csv(cfg['path'], header=None, names=cfg['columns'])
    if cfg['drop']:
        df.drop(columns=cfg['drop'], inplace=True)

    target = cfg['target']
    df_original = df.copy()

    random.seed(seed)

    # Inserir MCAR
    n_missing = int(missing_frac * len(df))
    missing_indices = random.sample(list(df.index), n_missing)
    df.loc[missing_indices, target] = np.nan

    numerical_cols = df.columns.drop(target)

    # ---------- KNN ponderado (Eta² + Euclidiana)
    etas = {col: eta_squared(df[col], df[target]) for col in numerical_cols}
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[numerical_cols]),
                             columns=numerical_cols)
    df_weighted = df_scaled * pd.Series(etas)

    imputed_weighted = df[target].copy()
    for idx in imputed_weighted[imputed_weighted.isna()].index:
        imputed_weighted[idx] = knn_impute(idx, df_weighted, df[target], k)

    acc_weighted = accuracy_score(
        df_original.loc[missing_indices, target],
        imputed_weighted.loc[missing_indices]
    )

    # ---------- KNN simples
    df_scaled_simple = pd.DataFrame(
        scaler.fit_transform(df[numerical_cols]),
        columns=numerical_cols
    )

    imputed_simple = df[target].copy()
    for idx in imputed_simple[imputed_simple.isna()].index:
        imputed_simple[idx] = knn_impute_simple(idx, df_scaled_simple, df[target], k)

    acc_simple = accuracy_score(
        df_original.loc[missing_indices, target],
        imputed_simple.loc[missing_indices]
    )

    # ---------- KNN Gower
    gower_matrix = gower_distance_matrix(df[numerical_cols])
    imputed_gower = df[target].copy()
    for idx in imputed_gower[imputed_gower.isna()].index:
        imputed_gower[idx] = knn_impute_gower(idx, gower_matrix, df[target], k)

    acc_gower = accuracy_score(
        df_original.loc[missing_indices, target],
        imputed_gower.loc[missing_indices]
    )

    return acc_weighted, acc_simple, acc_gower

# =========================================================
# Execução global (todas as bases)
# =========================================================

seeds = range(10, 40)
all_results = []

for dataset_name, cfg in DATASETS.items():
    print(f"\n=== Executando base: {dataset_name} ===")
    for s in seeds:
        acc_w, acc_s, acc_g = run_experiment(cfg, s)
        all_results.append({
            'dataset': dataset_name,
            'seed': s,
            'knn_weighted': acc_w,
            'knn_simple': acc_s,
            'knn_gower': acc_g
        })

df_results = pd.DataFrame(all_results)

summary = (
    df_results
    .groupby('dataset')
    .agg(['mean', 'std'])
)

print("\nResumo final (média e desvio padrão):")
print(summary)
