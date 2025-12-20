import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import random
import matplotlib.pyplot as plt

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
        'columns': [
            'Class', 
            'x-box', 'y-box', 'width', 'height', 
            'onpix', 'x-bar', 'y-bar', 'x2bar', 'y2bar',
            'xybar', 'x2ybr', 'xy2br', 'x-ege', 'xegvy',
            'y-ege', 'yegvx'
        ],
        'target': 'Class',
        'drop': [],
        'n_rows': 2000  # limite linhas para testes rápidos
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
    """Matriz de distância Gower vetorizada"""
    X_norm = (X - X.min()) / (X.max() - X.min())
    return np.abs(X_norm.values[:, None, :] - X_norm.values[None, :, :]).mean(axis=2)

def knn_gower_impute(df, target, gower_matrix, k=5):
    imputed = df[target].copy()
    missing_idx = imputed[imputed.isna()].index
    known_idx = imputed[imputed.notna()].index
    for idx in missing_idx:
        distances = gower_matrix[idx, known_idx]
        nearest = known_idx[np.argsort(distances)[:k]]
        labels = imputed.loc[nearest]
        imputed[idx] = labels.mode().iloc[0]
    return imputed

def knn_impute_vectorized(df_scaled, df_target, k=5):
    imputed = df_target.copy()
    missing_idx = imputed[imputed.isna()].index
    known_idx = imputed[imputed.notna()].index

    known_values = df_scaled.loc[known_idx].values
    known_targets = df_target.loc[known_idx]

    for idx in missing_idx:
        row = df_scaled.loc[idx].values
        distances = np.linalg.norm(known_values - row, axis=1)
        nearest = known_idx[np.argsort(distances)[:k]]
        labels = known_targets.loc[nearest]
        imputed[idx] = labels.mode().iloc[0]

    return imputed

# =========================================================
# Pipeline de experimento otimizada
# =========================================================
def run_experiment(cfg, seed, missing_frac=0.2, k=5):
    df = pd.read_csv(cfg['path'], header=None, names=cfg['columns'])
    if cfg.get('n_rows'):
        df = df.iloc[:cfg['n_rows']]
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

    # ---------- KNN simples
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[numerical_cols]), columns=numerical_cols)
    imputed_simple = knn_impute_vectorized(df_scaled, df[target], k=k)
    acc_simple = accuracy_score(df_original.loc[missing_indices, target],
                                imputed_simple.loc[missing_indices])

    # ---------- KNN ponderado (Eta² + Euclidiana)
    etas = np.array([eta_squared(df[col], df[target]) for col in numerical_cols])
    df_weighted = df_scaled * etas
    imputed_weighted = knn_impute_vectorized(df_weighted, df[target], k=k)
    acc_weighted = accuracy_score(df_original.loc[missing_indices, target],
                                  imputed_weighted.loc[missing_indices])

    # ---------- KNN Gower
    gower_matrix = gower_distance_matrix(df[numerical_cols])
    imputed_gower = knn_gower_impute(df, target, gower_matrix, k=k)
    acc_gower = accuracy_score(df_original.loc[missing_indices, target],
                               imputed_gower.loc[missing_indices])

    return acc_weighted, acc_simple, acc_gower

# =========================================================
# Execução global
# =========================================================
seeds = range(10, 40)
missing_fracs = [0.1, 0.2, 0.3]

all_results = []

for dataset_name, cfg in DATASETS.items():
    print(f"\n=== Executando base: {dataset_name} ===")
    for s in seeds:
        for frac in missing_fracs:
            acc_w, acc_s, acc_g = run_experiment(cfg, s, missing_frac=frac)
            all_results.append({
                'dataset': dataset_name,
                'seed': s,
                'missing_frac': frac,
                'knn_weighted': acc_w,
                'knn_simple': acc_s,
                'knn_gower': acc_g
            })

df_results = pd.DataFrame(all_results)

# =========================================================
# Resumo final
# =========================================================
summary = df_results.groupby(['dataset', 'missing_frac']).agg(['mean', 'std'])
print("\nResumo final (média e desvio padrão):")
print(summary)

# =========================================================
# Gráficos de barras (mean ± std)
# =========================================================
methods = ['knn_weighted', 'knn_simple', 'knn_gower']

for dataset in df_results['dataset'].unique():
    df_plot = summary.loc[dataset]
    x = np.arange(len(missing_fracs))
    width = 0.2

    plt.figure(figsize=(8,5))
    for i, method in enumerate(methods):
        means = df_plot[method]['mean'].values
        stds = df_plot[method]['std'].values
        plt.bar(x + i*width, means, width, yerr=stds, capsize=4, label=method)

    plt.xticks(x + width, [f"{int(f*100)}%" for f in missing_fracs])
    plt.ylim(0, 1.05)
    plt.xlabel("Fração de missing (MCAR)")
    plt.ylabel("Acurácia")
    plt.title(f"Base: {dataset}")
    plt.legend()
    plt.tight_layout()
    plt.show()
