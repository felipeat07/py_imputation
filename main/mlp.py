import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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
# ENCODING DE FEATURES
# =========================================================

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
# CLASSIFICADOR PARA AVALIAÇÃO
# =========================================================

def evaluate_knn_classifier(Xtr, ytr, Xte, yte, k=5):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(Xtr, ytr)
    preds = clf.predict(Xte)
    return accuracy_score(yte, preds)


# =========================================================
# IMPUTAÇÃO COM MLP (CORRIGIDA)
# =========================================================

def mlp_impute(X, y_miss, seed):
    y_imp = y_miss.copy()

    known_idx = y_imp[y_imp.notna()].index
    miss_idx  = y_imp[y_imp.isna()].index

    if len(miss_idx) == 0:
        return y_imp

    X_known = X.loc[known_idx]
    X_miss  = X.loc[miss_idx]

    # Encode do alvo (ESSENCIAL)
    le = LabelEncoder()
    y_known_enc = le.fit_transform(y_imp.loc[known_idx])

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=seed,
        early_stopping=True,
        n_iter_no_change=20
    )

    mlp.fit(X_known, y_known_enc)

    y_pred_enc = mlp.predict(X_miss)
    y_pred = le.inverse_transform(y_pred_enc)

    y_imp.loc[miss_idx] = y_pred
    return y_imp


# =========================================================
# EXPERIMENTO
# =========================================================

def run_experiment_mlp(cfg, seed, missing_frac, k=5):
    random.seed(seed)
    np.random.seed(seed)

    df = pd.read_csv(
        cfg['path'],
        sep=cfg.get('sep', ','),
        header=cfg.get('header', 0),
        skipinitialspace=True
    )

    if cfg.get('columns') is not None:
        df.columns = cfg['columns']

    # converte possíveis símbolos de missing
    df = df.replace('?', np.nan)

    # remove espaços extras em colunas string
    df = df.apply(lambda c: c.str.strip() if c.dtype == "object" else c)

    # embaralha antes de cortar linhas (evita classes raras)
    if cfg.get('n_rows'):
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        df = df.iloc[:cfg['n_rows']]

    if cfg['drop']:
        df = df.drop(columns=cfg['drop'])

    # remove classes com apenas 1 amostra (necessário para stratify)
    target = cfg['target']
    counts = df[target].value_counts()
    df = df[df[target].isin(counts[counts > 1].index)]

    X = df.drop(columns=target)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        stratify=y,
        random_state=seed
    )

    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    X_train_enc, X_test_enc = encode_features(X_train, X_test)

    scaler = MinMaxScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train_enc),
        columns=X_train_enc.columns
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test_enc),
        columns=X_test_enc.columns
    )

    acc_clf_original = evaluate_knn_classifier(
        X_train_s, y_train, X_test_s, y_test, k
    )

    y_train_miss = y_train.copy()
    n_missing = int(missing_frac * len(y_train))
    miss_idx = random.sample(list(y_train.index), n_missing)
    y_train_miss.loc[miss_idx] = np.nan

    y_mlp = mlp_impute(X_train_s, y_train_miss, seed)

    acc_imp_mlp = accuracy_score(
        y_train.loc[miss_idx],
        y_mlp.loc[miss_idx]
    )

    acc_clf_mlp = evaluate_knn_classifier(
        X_train_s, y_mlp, X_test_s, y_test, k
    )

    return {
        'imp_mlp': acc_imp_mlp,
        'clf_original': acc_clf_original,
        'clf_mlp': acc_clf_mlp
    }


# =========================================================
# EXECUÇÃO
# =========================================================

seeds = range(10, 40)
missing_fracs = [0.1, 0.2, 0.3]

all_results = []

for name, cfg in DATASETS.items():
    print(f"\n=== Base: {name} ===")
    for seed in seeds:
        for frac in missing_fracs:
            res = run_experiment_mlp(cfg, seed, frac)
            res.update({
                'dataset': name,
                'seed': seed,
                'missing_frac': frac
            })
            all_results.append(res)

df_results = pd.DataFrame(all_results)

summary = df_results.groupby(
    ['dataset', 'missing_frac']
).agg(['mean', 'std'])

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("\n=== Resumo final — MLP Imputation ===")
print(summary)
