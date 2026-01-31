import pandas as pd
import numpy as np
import random
from numba import njit, prange

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# =========================================================
# Configuração das bases
# =========================================================
DATASETS = {    
    'wine': {
        'path': 'data/wine/wine.data',
        'columns': [
            'Class',
            'Alcohol', 'Malic', 'Ash', 'Alcalinity',
            'Magnesium', 'Phenols', 'Flavanoids',
            'Nonflav', 'Proanth', 'Color', 'Hue',
            'OD', 'Proline'
        ],
        'target': 'Class',
        'drop': []
    },
    'image_segmentation': {
        'path': 'data/image_segmentation/segmentation_full_clean.csv',
        'sep': ',',
        'header': 0,
        'columns': [
            'Class',
            'Region-centroid-col', 'Region-centroid-row',
            'Region-pixel-count', 'Short-line-density-5',
            'Short-line-density-2', 'Vedge-mean',
            'Vedge-sd', 'Hedge-mean', 'Hedge-sd',
            'Intensity-mean', 'Rawred-mean',
            'Rawblue-mean', 'Rawgreen-mean',
            'Exred-mean', 'Exblue-mean', 'Exgreen-mean',
            'Value-mean', 'Saturation-mean', 'Hue-mean'
        ],
        'target': 'Class',
        'drop': [],
        'n_rows': 100
    },
    'pen_based_recognition': {
        'path': 'data/pen_based_recognition/pendigits_full.csv',
        'sep': ',',
        'header': 0,
        'columns': None,   
        'target': 'Class',
        'drop': [],
        'n_rows': 100
},
    'student': {
        'path': 'data/student/student-por.csv', 
        'sep': ';',          # IMPORTANTE: essa base usa ';'
        'header': 0,         # primeira linha é cabeçalho
        'columns': None,     # já vem com nomes
        'target': 'G3',      # nota final
        'drop': []
    },
    'obesity': {
        'path': 'data/obesity/ObesityDataSet_raw_and_data_sinthetic.csv',
        'sep': ',',
        'header': 0,
        'columns': None,
        'target': 'NObeyesdad',
        'drop': [],
        'n_rows': 100
    },
        'bank_marketing': {
        'path': 'data/bank/bank-full.csv',
        'sep': ';',
        'header': 0,
        'columns': None,         # usa os nomes do CSV
        'target': 'job',           # coluna alvo (yes/no)
        'drop': [],
        'n_rows': 100
    },
    'balance_scale': {
        'path': 'data/balance_scale/balance-scale.data',
        'sep': ',',
        'header': None,     # NÃO há cabeçalho
        'columns': [
            'Class',
            'Left-Weight',
            'Left-Distance',
            'Right-Weight',
            'Right-Distance'
        ],
        'target': 'Class',
        'drop': []
    },
    'nursery': {
        'path': 'data/nursery/nursery.data',
        'sep': ',',
        'header': None,   # NÃO há cabeçalho
        'columns': [
            'Parents',
            'Has_nurs',
            'Form',
            'Children',
            'Housing',
            'Finance',
            'Social',
            'Health',
            'Class'
        ],
        'target': 'Class',
        'drop': [],
        'n_rows': 100
},
    'car_evaluation': {
        'path': 'data/car_evaluation/car.data',
        'sep': ',',
        'header': None,   # não tem cabeçalho
        'columns': [
            'buying',
            'maint',
            'doors',
            'persons',
            'lug_boot',
            'safety',
            'class'
        ],
        'target': 'class',
        'drop': [],
        'n_rows': 100
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


@njit(parallel=True, fastmath=True)
def gower_num_cat(X_num, X_cat, min_vals, max_vals):
    n, p_num = X_num.shape
    _, p_cat = X_cat.shape

    dist = np.zeros((n, n), dtype=np.float32)

    for i in prange(n):
        for j in range(i + 1, n):
            # num part
            s_num = 0.0
            for k in range(p_num):
                denom = max_vals[k] - min_vals[k]
                if denom == 0:
                    continue
                s_num += abs(X_num[i, k] - X_num[j, k]) / denom

            # cat part
            s_cat = 0.0
            for k in range(p_cat):
                s_cat += 0 if X_cat[i, k] == X_cat[j, k] else 1

            # normalize
            denom_total = p_num + p_cat
            dist_val = (s_num + s_cat) / denom_total

            dist[i, j] = dist_val
            dist[j, i] = dist_val

    return dist


def gower_distance_matrix(X):
    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns

    X_num = X[num_cols].to_numpy(dtype=np.float32)
    X_cat = X[cat_cols].apply(lambda c: c.astype('category').cat.codes).to_numpy(dtype=np.int32)

    min_vals = np.nanmin(X_num, axis=0)
    max_vals = np.nanmax(X_num, axis=0)

    return gower_num_cat(X_num, X_cat, min_vals, max_vals)




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


def encode_features(X_train, X_test):
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns
    num_cols = X_train.select_dtypes(include=[np.number]).columns

    if len(cat_cols) == 0:
        return X_train.copy(), X_test.copy()

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    Xtr_cat = pd.DataFrame(
        ohe.fit_transform(X_train[cat_cols]),
        columns=ohe.get_feature_names_out(cat_cols)
    )

    Xte_cat = pd.DataFrame(
        ohe.transform(X_test[cat_cols]),
        columns=ohe.get_feature_names_out(cat_cols)
    )

    Xtr = pd.concat(
        [X_train[num_cols].reset_index(drop=True), Xtr_cat],
        axis=1
    )

    Xte = pd.concat(
        [X_test[num_cols].reset_index(drop=True), Xte_cat],
        axis=1
    )

    return Xtr, Xte


# =========================================================
# Pipeline principal (KNN invariável)
# =========================================================
def run_experiment(cfg, seed, missing_frac, k=5):
    random.seed(seed)
    np.random.seed(seed)

  #  # leitura antiga
  #  df = pd.read_csv(cfg['path'], header=None, names=cfg['columns'])

    # leitura (robusta para bases com e sem header)
    if cfg.get('columns') is not None:
        df = pd.read_csv(
            cfg['path'],
            header=None,
            names=cfg['columns'],
            sep=cfg.get('sep', ',')
        )
    else:
        df = pd.read_csv(
            cfg['path'],
            header=cfg.get('header', 0),
            sep=cfg.get('sep', ',')
        )

    if cfg.get('n_rows'):
        df = df.iloc[:cfg['n_rows']]
    if cfg['drop']:
        df = df.drop(columns=cfg['drop'])

    target = cfg['target']
    X = df.drop(columns=target)
    y = df[target]

    # REMOVE CLASSES COM MENOS DE 2 INSTÂNCIAS (necessário para stratify)
    vc = y.value_counts()
    valid_classes = vc[vc >= 2].index
    mask = y.isin(valid_classes)

    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)


    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )

    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    # encoding (única adaptação estrutural)
    X_train_enc, X_test_enc = encode_features(X_train, X_test)

    # escalonamento
    scaler = MinMaxScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train_enc),
        columns=X_train_enc.columns
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test_enc),
        columns=X_test_enc.columns
    )

    # baseline classificador
    acc_clf_original = evaluate_knn_classifier(
        X_train_s, y_train, X_test_s, y_test, k
    )

    # MCAR no alvo
    y_train_miss = y_train.copy()
    n_missing = int(missing_frac * len(y_train))
    miss_idx = random.sample(list(y_train.index), n_missing)
    y_train_miss.loc[miss_idx] = np.nan

    # KNN simples
    y_simple = knn_impute_vectorized(X_train_s, y_train_miss, k)
    acc_imp_simple = accuracy_score(y_train.loc[miss_idx], y_simple.loc[miss_idx])
    acc_clf_simple = evaluate_knn_classifier(
        X_train_s, y_simple, X_test_s, y_test, k
    )

    # KNN ponderado por Eta²
    etas = np.array([
        eta_squared(X_train_s[col], y_train)
        for col in X_train_s.columns
    ])

    X_weighted = X_train_s * etas
    y_weighted = knn_impute_vectorized(X_weighted, y_train_miss, k)
    acc_imp_weighted = accuracy_score(y_train.loc[miss_idx], y_weighted.loc[miss_idx])
    acc_clf_weighted = evaluate_knn_classifier(
        X_train_s, y_weighted, X_test_s, y_test, k
    )

    # KNN Gower
    df_gower = X_train.copy()
    df_gower[target] = y_train_miss

    gower = gower_distance_matrix(X_train)   # <-- aqui!
    y_gower = knn_gower_impute(df_gower, target, gower, k)
    acc_imp_gower = accuracy_score(y_train.loc[miss_idx], y_gower.loc[miss_idx])
    acc_clf_gower = evaluate_knn_classifier(
        X_train_s, y_gower, X_test_s, y_test, k
    )

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
# Resumo final
# =========================================================
summary = df_results.groupby(['dataset', 'missing_frac']).agg(['mean', 'std'])
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("\n=== Resumo final (acurácia de imputação e classificadores) ===")
print(summary)
