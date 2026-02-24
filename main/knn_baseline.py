import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# =========================================================
# CONFIGURAÇÃO DAS 4 BASES DO ARTIGO
# =========================================================
DATASETS = {
    'obesity': {
        'path': 'data/obesity/ObesityDataSet_raw_and_data_sinthetic.csv',
        'sep': ',',
        'header': 0,
        'columns': None,
        'target': 'NObeyesdad'
    },
    'bank_marketing': {
        'path': 'data/bank/bank-full.csv',
        'sep': ';',
        'header': 0,
        'columns': None,
        'target': 'job'
    },
    'nursery': {
        'path': 'data/nursery/nursery.data',
        'sep': ',',
        'header': None,
        'columns': [
            'Parents','Has_nurs','Form','Children',
            'Housing','Finance','Social','Health','Class'
        ],
        'target': 'Class'
    },
    'car_evaluation': {
        'path': 'data/car_evaluation/car.data',
        'sep': ',',
        'header': None,
        'columns': [
            'buying','maint','doors','persons',
            'lug_boot','safety','class'
        ],
        'target': 'class'
    }
}


# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================

def knn_impute(X, y, k=1):
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

    Xtr = pd.concat([X_train[num_cols].reset_index(drop=True), Xtr_cat], axis=1)
    Xte = pd.concat([X_test[num_cols].reset_index(drop=True), Xte_cat], axis=1)

    return Xtr, Xte


# =========================================================
# PIPELINE PRINCIPAL
# =========================================================

def run_experiment(cfg, seed, missing_frac):

    random.seed(seed)
    np.random.seed(seed)

    # leitura
    if cfg['columns'] is not None:
        df = pd.read_csv(cfg['path'], header=None,
                         names=cfg['columns'], sep=cfg['sep'])
    else:
        df = pd.read_csv(cfg['path'], header=cfg['header'],
                         sep=cfg['sep'])

    target = cfg['target']
    X = df.drop(columns=target)
    y = df[target]

    # remove classes com menos de 2 instâncias
    vc = y.value_counts()
    valid_classes = vc[vc >= 2].index
    mask = y.isin(valid_classes)

    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    # split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )

    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    # encoding + scaling
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

    # =====================================================
    # 1️⃣ Classificador na base original
    # =====================================================
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train_s, y_train)
    acc_original = accuracy_score(y_test, clf.predict(X_test_s))

    # =====================================================
    # 2️⃣ MCAR no alvo
    # =====================================================
    y_train_miss = y_train.copy()
    n_missing = int(missing_frac * len(y_train))
    miss_idx = random.sample(list(y_train.index), n_missing)
    y_train_miss.loc[miss_idx] = np.nan

    # =====================================================
    # 3️⃣ Imputação KNN k=1
    # =====================================================
    y_imp = knn_impute(X_train_s, y_train_miss, k=1)

    acc_imput = accuracy_score(
        y_train.loc[miss_idx],
        y_imp.loc[miss_idx]
    )

    # =====================================================
    # 4️⃣ Classificador após imputação
    # =====================================================
    clf.fit(X_train_s, y_imp)
    acc_imputed = accuracy_score(y_test, clf.predict(X_test_s))

    return acc_imput, acc_original, acc_imputed


# =========================================================
# EXECUÇÃO GLOBAL
# =========================================================

seeds = range(10, 30)
missing_fracs = [0.1, 0.2, 0.3]

for name, cfg in DATASETS.items():

    print("\n===================================================")
    print(f"BASE: {name}")
    print("===================================================")

    for frac in missing_fracs:

        imp_scores = []
        clf_original_scores = []
        clf_imputed_scores = []

        for seed in seeds:
            acc_imp, acc_original, acc_imputed = run_experiment(cfg, seed, frac)

            imp_scores.append(acc_imp)
            clf_original_scores.append(acc_original)
            clf_imputed_scores.append(acc_imputed)

        print(f"\nMissing: {int(frac*100)}%")

        print(f"Imputação              -> Média: {np.mean(imp_scores):.4f} | Std: {np.std(imp_scores):.4f}")
        print(f"Classificador Original -> Média: {np.mean(clf_original_scores):.4f} | Std: {np.std(clf_original_scores):.4f}")
        print(f"Classificador Imputado -> Média: {np.mean(clf_imputed_scores):.4f} | Std: {np.std(clf_imputed_scores):.4f}")