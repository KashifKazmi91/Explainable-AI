#!/usr/bin/env python3
"""
HXAI-LSTM Colab-ready script
Adds: data loading (NSL-KDD), optional Kaggle download hooks (UNSW-NB15, TON-IoT), preprocessing,
LSTM training, explainability (SHAP, PFI, PDP/ICE, LIME, ELI5), Feature Consensus Score (FCS),
explainability-guided feature selection and retraining.

Usage: Upload to Google Colab or run locally after installing required packages.
Requirements (Colab):
  !pip install numpy pandas scikit-learn imbalanced-learn tensorflow shap lime eli5 matplotlib seaborn tqdm pdpbox kaggle

Author: generated for repository Explainable-AI
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import shap
from lime.lime_tabular import LimeTabularExplainer
import eli5
from eli5.sklearn import PermutationImportance
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Optional: pdpbox used for PDP plotting
try:
    from pdpbox import pdp
except Exception:
    pdp = None

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
LSTM_EPOCHS = 10
BATCH_SIZE = 32
LSTM_UNITS = 50
DROPOUT_RATE = 0.2
EXPLAINER_BACKGROUND_SIZE = 200
EXPLAINER_SAMPLE_SIZE = 200
TOP_K_FEATURES = 15

# -------------------------
# Kaggle helpers (optional)
# -------------------------

def setup_kaggle(kaggle_json_path):
    """Place kaggle.json in ~/.kaggle and set permissions (for Colab)."""
    dest = os.path.expanduser('~/.kaggle')
    os.makedirs(dest, exist_ok=True)
    dest_file = os.path.join(dest, 'kaggle.json')
    with open(kaggle_json_path, 'rb') as src:
        open(dest_file, 'wb').write(src.read())
    os.chmod(dest_file, 0o600)
    print('kaggle.json installed to', dest_file)


def download_kaggle_dataset(slug, dest_folder='/content/datasets'):
    """Download and unzip a Kaggle dataset using the kaggle CLI. Requires kaggle.json set up."""
    os.makedirs(dest_folder, exist_ok=True)
    cmd = f"kaggle datasets download -d {slug} -p {dest_folder} --unzip"
    print('Running:', cmd)
    rc = os.system(cmd)
    if rc != 0:
        raise RuntimeError('kaggle download failed; ensure kaggle.json is configured and dataset slug is correct')
    print('Downloaded dataset', slug, 'to', dest_folder)

# -------------------------
# Data loading
# -------------------------

def load_nsl_kdd():
    train_url = 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt'
    test_url = 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt'
    col_names = [
        'duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent',
        'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root',
        'num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
        'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
        'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
        'dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
        'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label'
    ]
    df_train = pd.read_csv(train_url, names=col_names)
    df_test = pd.read_csv(test_url, names=col_names)
    df = pd.concat([df_train, df_test], ignore_index=True)
    return df

# -------------------------
# Preprocessing
# -------------------------

def preprocess_df(df, label_col='label'):
    df = df.copy()
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df['is_attack'] = df[label_col].apply(lambda x: 0 if str(x).lower() == 'normal' else 1)
    y = df['is_attack'].values
    df_feat = df.drop(columns=[label_col, 'is_attack'])
    cat_cols = df_feat.select_dtypes(include=['object']).columns.tolist()
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        df_feat[c] = le.fit_transform(df_feat[c].astype(str))
        encoders[c] = le
    numeric_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
    scaler = MinMaxScaler()
    df_feat[numeric_cols] = scaler.fit_transform(df_feat[numeric_cols])
    X = df_feat.values
    return X, y, df_feat.columns.tolist(), encoders, scaler

def reshape_for_lstm(X, timesteps=1):
    if timesteps == 1:
        return X.reshape((X.shape[0], 1, X.shape[1]))
    # sliding windows could be implemented here
    return X.reshape((X.shape[0], 1, X.shape[1]))

# -------------------------
# LSTM model
# -------------------------

def build_lstm_model(input_shape, units=LSTM_UNITS, dropout=DROPOUT_RATE):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.LSTM(units))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# -------------------------
# Explainability helpers
# -------------------------

def normalize_series(arr):
    arr = np.array(arr, dtype=float)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if mx - mn == 0:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def get_shap_values_keras(model, X_background, X_sample):
    def f(x):
        if x.ndim == 3:
            preds = model.predict(x, verbose=0).ravel()
        else:
            preds = model.predict(x.reshape((x.shape[0], 1, x.shape[1])), verbose=0).ravel()
        return np.vstack([1 - preds, preds]).T
    if X_background.ndim == 3 and X_background.shape[1] == 1:
        Xb = X_background.reshape((X_background.shape[0], X_background.shape[2]))
        Xs = X_sample.reshape((X_sample.shape[0], X_sample.shape[2]))
    else:
        Xb, Xs = X_background, X_sample
    explainer = shap.KernelExplainer(f, Xb[:min(len(Xb), EXPLAINER_BACKGROUND_SIZE)])
    shap_vals = explainer.shap_values(Xs[:min(len(Xs), EXPLAINER_SAMPLE_SIZE)], nsamples=100)
    shap_arr = np.array(shap_vals[1])
    return shap_arr

def compute_permutation_importance_sklearn(model, X_val, y_val, n_repeats=10):
    class KerasWrapper:
        def __init__(self, m):
            self.model = m
        def predict(self, X):
            if X.ndim == 3:
                preds = self.model.predict(X, verbose=0).ravel()
            else:
                preds = self.model.predict(X.reshape((X.shape[0], 1, X.shape[1])), verbose=0).ravel()
            return (preds > 0.5).astype(int)
        def predict_proba(self, X):
            if X.ndim == 3:
                preds = self.model.predict(X, verbose=0).ravel()
            else:
                preds = self.model.predict(X.reshape((X.shape[0], 1, X.shape[1])), verbose=0).ravel()
            return np.vstack([1-preds, preds]).T
    wrapper = KerasWrapper(model)
    if X_val.ndim == 3 and X_val.shape[1] == 1:
        Xflat = X_val.reshape((X_val.shape[0], X_val.shape[2]))
    else:
        Xflat = X_val
    res = permutation_importance(wrapper, Xflat, y_val, n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=1, scoring='f1')
    importances = res.importances_mean
    return importances

def eli5_permutation_importance(model, X_val, y_val):
    class KerasWrap:
        def __init__(self, model):
            self.model = model
        def predict(self, X):
            if X.ndim == 3:
                preds = self.model.predict(X, verbose=0).ravel()
            else:
                preds = self.model.predict(X.reshape((X.shape[0], 1, X.shape[1])), verbose=0).ravel()
            return (preds>0.5).astype(int)
        def predict_proba(self, X):
            if X.ndim == 3:
                preds = self.model.predict(X, verbose=0).ravel()
            else:
                preds = self.model.predict(X.reshape((X.shape[0], 1, X.shape[1])), verbose=0).ravel()
            return np.vstack([1-preds, preds]).T
    wrap = KerasWrap(model)
    if X_val.ndim == 3 and X_val.shape[1] == 1:
        Xflat = X_val.reshape((X_val.shape[0], X_val.shape[2]))
    else:
        Xflat = X_val
    perm = PermutationImportance(wrap, scoring='f1', n_iter=10, random_state=RANDOM_STATE)
    perm.fit(Xflat, y_val)
    return perm.feature_importances_

def compute_pdp_ice(model, X, feature_index, grid_size=20):
    if X.ndim == 3 and X.shape[1] == 1:
        Xflat = X.reshape((X.shape[0], X.shape[2]))
        reshape_needed = True
    else:
        Xflat = X
        reshape_needed = False
    col_vals = Xflat[:, feature_index]
    grid = np.linspace(col_vals.min(), col_vals.max(), grid_size)
    preds = []
    n_instances = Xflat.shape[0]
    max_ice_instances = min(200, n_instances)
    idxs = np.random.choice(n_instances, size=max_ice_instances, replace=False)
    for g in grid:
        Xtmp = Xflat.copy()
        Xtmp[:, feature_index] = g
        if reshape_needed:
            Xtmp_in = Xtmp.reshape((Xtmp.shape[0], 1, Xtmp.shape[1]))
        else:
            Xtmp_in = Xtmp
        pred = model.predict(Xtmp_in, verbose=0).ravel()
        preds.append(pred)
    preds = np.stack(preds, axis=1)
    pdp_vals = preds.mean(axis=0)
    ice_vals = preds[idxs, :]
    return grid, pdp_vals, ice_vals

# -------------------------
# FCS aggregation
# -------------------------

def compute_fcs(feature_names, shap_importance, pfi_importance, pdp_importance, ice_importance=None, lime_importance=None, eli5_importance=None):
    n = len(feature_names)
    scores = pd.DataFrame(index=feature_names)
    scores['shap'] = normalize_series(shap_importance)
    scores['pfi'] = normalize_series(pfi_importance)
    scores['pdp'] = normalize_series(pdp_importance)
    scores['ice'] = normalize_series(ice_importance) if ice_importance is not None else 0
    scores['lime'] = normalize_series(lime_importance) if lime_importance is not None else 0
    scores['eli5'] = normalize_series(eli5_importance) if eli5_importance is not None else 0
    scores.fillna(0, inplace=True)
    scores['fcs'] = scores.mean(axis=1)
    return scores.sort_values('fcs', ascending=False)

# -------------------------
# Full pipeline
# -------------------------

def run_pipeline(df, dataset_name='NSL-KDD'):
    print(f"Running pipeline for {dataset_name}")
    X, y, feat_names, encoders, scaler = preprocess_df(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    X_train_seq = reshape_for_lstm(X_train_res, timesteps=1)
    X_test_seq = reshape_for_lstm(X_test, timesteps=1)
    model = build_lstm_model(input_shape=(1, X_train_seq.shape[2]))
    early = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X_train_seq, y_train_res, validation_split=0.2, epochs=LSTM_EPOCHS, batch_size=BATCH_SIZE, callbacks=[early], verbose=2)
    y_pred_prob = model.predict(X_test_seq, verbose=0).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)
    print('Initial classification report:')
    print(classification_report(y_test, y_pred, digits=4))
    # Explainability (sampled to reduce runtime)
    print('Computing SHAP (sampled)...')
    try:
        idx_bg = np.random.choice(X_train_seq.shape[0], size=min(EXPLAINER_BACKGROUND_SIZE, X_train_seq.shape[0]), replace=False)
        idx_samp = np.random.choice(X_test_seq.shape[0], size=min(EXPLAINER_SAMPLE_SIZE, X_test_seq.shape[0]), replace=False)
        shap_arr = get_shap_values_keras(model, X_train_seq[idx_bg], X_test_seq[idx_samp])
        shap_mean_abs = np.mean(np.abs(shap_arr), axis=0)
    except Exception as e:
        print('SHAP error:', e)
        shap_mean_abs = np.zeros(len(feat_names))
    print('Computing sklearn permutation importance (PFI)...')
    try:
        pfi_vals = compute_permutation_importance_sklearn(model, X_test_seq, y_test, n_repeats=5)
    except Exception as e:
        print('PFI error:', e)
        pfi_vals = np.zeros(len(feat_names))
    print('Computing ELI5 permutation importance...')
    try:
        eli5_vals = eli5_permutation_importance(model, X_test_seq, y_test)
    except Exception as e:
        print('ELI5 error:', e)
        eli5_vals = np.zeros(len(feat_names))
    # PDP/ICE for top features by PFI
    pdp_importance = np.zeros(len(feat_names))
    ice_importance = np.zeros(len(feat_names))
    try:
        top_pfi_idx = np.argsort(pfi_vals)[::-1][:min(5, len(feat_names))]
        for idx in top_pfi_idx:
            grid, pdp_vals, ice_vals = compute_pdp_ice(model, X_test_seq, idx, grid_size=20)
            pdp_importance[idx] = np.max(pdp_vals) - np.min(pdp_vals)
            ice_importance[idx] = np.mean(np.std(ice_vals, axis=1))
    except Exception as e:
        print('PDP/ICE error:', e)
    print('Computing LIME aggregated weights (sampled)...')
    lime_agg = np.zeros(len(feat_names))
    try:
        if X_train_seq.ndim == 3 and X_train_seq.shape[1] == 1:
            Xtrain_flat = X_train_seq.reshape((X_train_seq.shape[0], X_train_seq.shape[2]))
            Xtest_flat = X_test_seq.reshape((X_test_seq.shape[0], X_test_seq.shape[2]))
        else:
            Xtrain_flat = X_train_seq
            Xtest_flat = X_test_seq
        expl = LimeTabularExplainer(Xtrain_flat[:1000], feature_names=feat_names, class_names=['normal','attack'], discretize_continuous=True)
        n_lime_samples = min(20, Xtest_flat.shape[0])
        lime_idxs = np.random.choice(Xtest_flat.shape[0], size=n_lime_samples, replace=False)
        for i in lime_idxs:
            exp = expl.explain_instance(Xtest_flat[i], lambda x: np.vstack([1-model.predict(x.reshape((x.shape[0],1,x.shape[1])),verbose=0).ravel(), model.predict(x.reshape((x.shape[0],1,x.shape[1])),verbose=0).ravel()]).T, num_features=10)
            for fname, weight in exp.as_list():
                for j, fn in enumerate(feat_names):
                    if fn in fname:
                        lime_agg[j] += abs(weight)
        lime_agg = lime_agg / max(1, n_lime_samples)
    except Exception as e:
        print('LIME error:', e)
        lime_agg = np.zeros(len(feat_names))
    # Compute FCS
    fcs_df = compute_fcs(feat_names, shap_mean_abs, pfi_vals, pdp_importance, ice_importance, lime_agg, eli5_vals)
    print('Top features by FCS:')
    print(fcs_df.head(20))
    # Retrain on top-K features
    top_features = fcs_df.index.tolist()[:TOP_K_FEATURES]
    selected_idxs = [feat_names.index(f) for f in top_features]
    X_train_red = X_train_res[:, selected_idxs]
    X_test_red = X_test[:, selected_idxs]
    X_train_red_seq = reshape_for_lstm(X_train_red, timesteps=1)
    X_test_red_seq = reshape_for_lstm(X_test_red, timesteps=1)
    model2 = build_lstm_model(input_shape=(1, X_train_red_seq.shape[2]))
    early2 = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model2.fit(X_train_red_seq, y_train_res, validation_split=0.2, epochs=LSTM_EPOCHS, batch_size=BATCH_SIZE, callbacks=[early2], verbose=2)
    y_pred2 = (model2.predict(X_test_red_seq, verbose=0).ravel() > 0.5).astype(int)
    print('Retrained model report:')
    print(classification_report(y_test, y_pred2, digits=4))
    # Save results
    fcs_csv = f'fcs_{dataset_name}.csv'
    fcs_df.to_csv(fcs_csv)
    print('Saved', fcs_csv)
    model.save(f'hxai_lstm_{dataset_name}.h5')
    print('Saved model hxai_lstm_{dataset_name}.h5')
    return {
        'model': model,
        'model_selected': model2,
        'fcs': fcs_df,
        'feat_names': feat_names,
        'history': history
    }

# -------------------------
# Entry point
# -------------------------

def main():
    parser = argparse.ArgumentParser(description='HXAI-LSTM runnable script')
    parser.add_argument('--dataset', choices=['nsl-kdd','unsw-nb15','toniot'], default='nsl-kdd')
    parser.add_argument('--kaggle-json', default=None, help='Path to kaggle.json to enable Kaggle downloads (optional)')
    parser.add_argument('--kaggle-slug', default=None, help='Kaggle dataset slug to download when dataset is unsw-nb15 or toniot')
    args = parser.parse_args()
    if args.dataset == 'nsl-kdd':
        df = load_nsl_kdd()
        run_pipeline(df, dataset_name='NSL-KDD')
    else:
        if args.kaggle_json:
            setup_kaggle(args.kaggle_json)
        if args.kaggle_slug is None:
            print('Please provide --kaggle-slug for UNSW-NB15 or TON-IoT download')
            return
        dest = '/content/datasets'
        download_kaggle_dataset(args.kaggle_slug, dest_folder=dest)
        # Attempt to auto-find CSV in downloaded folder
        files = []
        for root, _, filenames in os.walk(dest):
            for fn in filenames:
                if fn.lower().endswith('.csv'):
                    files.append(os.path.join(root, fn))
        if len(files) == 0:
            print('No CSV files found in downloaded dataset. Please inspect', dest)
            return
        # Naive choice: use the largest CSV file
        files_sorted = sorted(files, key=lambda p: os.path.getsize(p), reverse=True)
        csv_path = files_sorted[0]
        print('Using CSV:', csv_path)
        df = pd.read_csv(csv_path)
        # User may need to adapt preprocessing for these datasets
        X, y, feat_names, encoders, scaler = preprocess_df(df, label_col=df.columns[-1] if df.shape[1] > 0 else 'label')
        run_pipeline(df, dataset_name=args.dataset.upper())

if __name__ == '__main__':
    main()
