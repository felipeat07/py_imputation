import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# =========================================================
# Configuração das bases
# =========================================================
DATASETS = {
    'glass': {
        'path': 'data/glass_identification/glass.data',
        'columns': ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type'],
        'target': 'Type',
        'drop': ['Id']
    },
    'wine': {
        'path': 'data/wine/wine.data',
        'columns': ['Class', 'Alcohol', 'Malic', 'Ash', 'Alcalinity',
                    'Magnesium', 'Phenols', 'Flavanoids',
                    'Nonflav', 'Proanth', 'Color', 'Hue',
                    'OD', 'Proline'],
        'target': 'Class',
        'drop': []
    },
    'letter': {
        'path': 'data/letter_recognition/letter-recognition.data',
        'columns': [
            'Class', 'x-box', 'y-box', 'width', 'height',
            'onpix', 'x-bar', 'y-bar', 'x2bar', 'y2bar',
            'xybar', 'x2ybr', 'xy2br', 'x-ege', 'xegvy',
            'y-ege', 'yegvx'
        ],
        'target': 'Class',
        'drop': [],
        'n_rows': 1000
    }
}

# =========================================================
# Funções auxiliares
# =========================================================
def eta_squared(x, y):
    classes = y.dropna().unique()
    grand_mean = x[y.notna()].mean()
    ss_between = sum(
        len(x[y == c]) * (x[y == c].mean() - grand_mean) ** 2
        for c in classes
    )
    ss_total = sum((x[y.notna()] - grand_mean) ** 2)
    return ss_between / ss_total if ss_total != 0 else 0

def knn_impute_vectorized(X, y, k=5):
    y_imp = y.copy()
    missing = y_imp[y_imp.isna()].index
    known = y_imp[y_imp.notna()].index

    X_known = X.loc[known].values
    y_known = y_imp.loc[known]

    for idx in missing:
        row = X.loc[idx].values
        dists = np.linalg.norm(X_known - row, axis=1)
        nearest = known[np.argsort(dists)[:k]]
        y_imp.loc[idx] = y_known.loc[nearest].mode().iloc[0]

    return y_imp

def gower_distance_matrix(X):
    Xn = (X - X.min()) / (X.max() - X.min())
    return np.abs(
        Xn.values[:, None, :] - Xn.values[None, :, :]
    ).mean(axis=2)

def knn_gower_impute(df, target, gower_matrix, k=5):
    y_imp = df[target].copy()
    missing = y_imp[y_imp.isna()].index
    known = y_imp[y_imp.notna()].index

    for idx in missing:
        dists = gower_matrix[idx, known]
        nearest = known[np.argsort(dists)[:k]]
        y_imp.loc[idx] = y_imp.loc[nearest].mode().iloc[0]

    return y_imp

def evaluate_knn_classifier(Xtr, ytr, Xte, yte, k=5):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(Xtr, ytr)
    preds = clf.predict(Xte)
    return accuracy_score(yte, preds)

# =========================================================
# Pipeline principal
# =========================================================
def run_experiment(cfg, seed, missing_frac, k=5):
    random.seed(seed)
    np.random.seed(seed)

    # leitura e preparação
    df = pd.read_csv(cfg['path'], header=None, names=cfg['columns'])
    if cfg.get('n_rows'):
        df = df.iloc[:cfg['n_rows']]
    if cfg['drop']:
        df = df.drop(columns=cfg['drop'])

    target = cfg['target']
    X = df.drop(columns=target)
    y = df[target]

    # split treino/teste fixo
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )

    # reset de índices
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    # escalonamento
    scaler = MinMaxScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_s  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # baseline classificador na base original
    acc_clf_original = evaluate_knn_classifier(X_train_s, y_train, X_test_s, y_test, k)

    # MCAR
    y_train_miss = y_train.copy()
    n_missing = int(missing_frac * len(y_train))
    miss_idx = random.sample(list(y_train.index), n_missing)
    y_train_miss.loc[miss_idx] = np.nan

    # imputação simples
    y_simple = knn_impute_vectorized(X_train_s, y_train_miss, k)
    acc_imp_simple = accuracy_score(y_train.loc[miss_idx], y_simple.loc[miss_idx])
    acc_clf_simple = evaluate_knn_classifier(X_train_s, y_simple, X_test_s, y_test, k)

    # imputação ponderada (Eta²)
    etas = np.array([eta_squared(X_train[col], y_train) for col in X_train.columns])
    X_weighted = X_train_s * etas
    y_weighted = knn_impute_vectorized(X_weighted, y_train_miss, k)
    acc_imp_weighted = accuracy_score(y_train.loc[miss_idx], y_weighted.loc[miss_idx])
    acc_clf_weighted = evaluate_knn_classifier(X_train_s, y_weighted, X_test_s, y_test, k)

    # imputação Gower
    df_gower = X_train.copy()
    df_gower[target] = y_train_miss
    gower = gower_distance_matrix(X_train)
    y_gower = knn_gower_impute(df_gower, target, gower, k)
    acc_imp_gower = accuracy_score(y_train.loc[miss_idx], y_gower.loc[miss_idx])
    acc_clf_gower = evaluate_knn_classifier(X_train_s, y_gower, X_test_s, y_test, k)

    return {
        'imp_simple': acc_imp_simple,
        'imp_weighted': acc_imp_weighted,
        'imp_gower': acc_imp_gower,
        'clf_original': acc_clf_original,
        'clf_simple': acc_clf_simple,
        'clf_weighted': acc_clf_weighted,
        'clf_gower': acc_clf_gower
    }

# =========================================================
# Execução global
# =========================================================
seeds = range(10, 40)
missing_fracs = [0.1, 0.2, 0.3]

all_results = []

for name, cfg in DATASETS.items():
    print(f"\n=== Base: {name} ===")
    for seed in seeds:
        for frac in missing_fracs:
            res = run_experiment(cfg, seed, frac)
            res.update({'dataset': name, 'seed': seed, 'missing_frac': frac})
            all_results.append(res)

df_results = pd.DataFrame(all_results)

# =========================================================
# Resumo final: médias e desvios
# =========================================================
summary = df_results.groupby(['dataset', 'missing_frac']).agg(['mean', 'std'])
print("\n=== Resumo final (acurácia de imputação e classificadores) ===")
pd.set_option('display.max_columns', None)  # mostra todas as colunas
pd.set_option('display.width', 200)         # aumenta a largura do display
print(summary)