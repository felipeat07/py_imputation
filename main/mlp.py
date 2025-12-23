import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


# =========================================================
# CONFIGURAÇÃO DAS BASES
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
        'n_rows': 2000
    },
    'car': {
        'path': 'data/car/car.data',
        'columns': [
            'buying', 'maint', 'doors',
            'persons', 'lug_boot', 'safety',
            'Class'
        ],
        'target': 'Class',
        'drop': [],
        'n_rows': 2000
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

    df = pd.read_csv(cfg['path'], header=None, names=cfg['columns'])

    if cfg.get('n_rows'):
        df = df.iloc[:cfg['n_rows']]

    if cfg['drop']:
        df = df.drop(columns=cfg['drop'])

    target = cfg['target']
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
