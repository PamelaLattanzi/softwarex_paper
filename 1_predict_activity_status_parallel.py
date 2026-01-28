#!/usr/bin/env python3

"""
MODELS FOR PREDICTING ACTIVITY STATUS (fishing/not_fishing) OF EACH POSITION WITHIN A SSF FISHING TRIP

This script trains and evaluates statistical (Logistic Regression) and Machine Learning models
(Decision Tree, Random Forest, Extreme Gradient Boosting) to predict the activity status
(in each position - GPS point - within SSF fishing trips) based on a set of variables selected as predictors.
It utilizes a nested cross-validation approach for robust model selection
and hyperparameter tuning, followed by a final evaluation on a held-out test set.
Feature importance (through SHapley Additive exPlanations, SHAP by Lundberg and Lee (2017)) 
is also calculated.

Usage (set basedir as the 'path/to/your_folder/'):
    cd basedir
    python 1_predict_activity_status_parallel.py

Arguments:
    --approach: Defines the splitting strategy for cross-validation and test set (line 113).
                'p' for point-based (individual point) splitting with stratification.
                'b' for trip-based grouping, ensuring points from the same trip stay together
                    (preferred choice, based on Samarao et al.)
    
Outputs:
    - model_performances_on_data_splitting_with_shap.csv: 
        Detailed results from nested cross-validation and final test.

"""

#### LIBRARIES ####
import os
import numpy as np
import pandas as pd
import warnings
import pyreadr
import time
import matplotlib.pyplot as plt
import shap
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GroupKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


warnings.filterwarnings('ignore')

#### DIRECTORIES ####
basedir = "path/to/your_folder/"  # must contain this script and the input dataset
outdir = "results"

if not os.path.exists(os.path.join(basedir, outdir)):
    os.makedirs(os.path.join(basedir, outdir))

#### DATA LOADING ####
# Load the dataset
dat_ini = pyreadr.read_r(os.path.join(
    basedir, "anonymized_dataset.rds"))
dat = dat_ini[None]  # Extract the data frame from the dictionary

# Set time resolution - 30 sec or 1 min
# dat = dat[dat['time_stamp'].dt.second == 0] #1 minute

own_data = dat.copy()

# Select specific columns to keep
columns_to_keep = ['DATE_TIME', 'seq', 'longitude', 'latitude', 'TRIP_ID', 'depth', 'course_diff',
                   'SPEED', 'STATUS', 'distance_from_coast', 'time_seconds', 'trip_duration', 'months', 'hours']
own_data = own_data[columns_to_keep]
# Rename columns as needed
own_data.rename(columns={"TRIP_ID": "boat_trip_id",
                         "STATUS": "target"}, inplace=True)

# Define columns to train the model - in this case, 7 predictors (use distance_from_coast OR depth)
# Note that these variables have already been transformed as needed (e.g., scaling into [0-1])
columns_to_train = ['SPEED',
                    'course_diff',
                    'distance_from_coast',
                    # 'depth',
                    'time_seconds',
                    'trip_duration',
                    'months',
                    'hours']

# Check for columns with string or categorical data
own_data["target"] = own_data["target"].astype(
    "category").cat.codes  # Convert to numeric codes
# print(">>> Data loading and pre-processing done \n")

#### COMBINATIONS ####
# Get the number of variables in the list
num_var = len(columns_to_train)
combo = columns_to_train.copy()

#### FISHING AS POSITIVES ####
# Define the positive label for 'fishing'
fishing_label = 0

#### CORE PART ####

# --- USER SETTINGS ---------------------------------------------------

# Choose 'p' = point-based CV, or 'b' = boat(trip)-based CV
approach = 'b'

# The raw data, with columns 'boat_trip_id' and 'target'
unique_trip_ids = own_data['boat_trip_id'].unique()

# Split unique_trip_ids into (train+val) and test sets based on boat_trip_id
train_val_trip_ids, test_trip_ids = train_test_split(
    unique_trip_ids, test_size=0.1, random_state=42)

# Initialize 'set' column - track if a trip was in the 'train_val' or 'test' set
own_data['set'] = 'train_val'
own_data.loc[own_data['boat_trip_id'].isin(test_trip_ids), 'set'] = 'test'

# Create the fixed DataFrames
df_test = own_data[own_data['set'] == 'test'].copy()
df_train_val = own_data[own_data['set'] == 'train_val'].copy()

print(f"Training+Validation data size: {len(df_train_val)}")
print(f"Test data size: {len(df_test)}")

# ----------------------------------------------------------------------

# Encode the target once on the combined training and validation set
# Use df_train_val for encoding the target
label_encoder_overall = LabelEncoder().fit(df_train_val['target'])
y_train_val = label_encoder_overall.transform(df_train_val['target'])

# For GroupKFold, extract groups as unique boat_trip_id values for training and validation
groups_train_val = df_train_val['boat_trip_id'].values
# Number of unique groups (= trips)
print(f"N. groups (train+val): {len(np.unique(groups_train_val))}")

# Outer CV: stratify on points or group on trips
if approach == 'p':
    outer_cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42)
else:
    outer_cv = GroupKFold(n_splits=5)

# Inner CV for hyperparameter tuning
inner_cv = StratifiedKFold(n_splits=3, shuffle=True,
                           random_state=42)

# Define model pipelines
model_pipelines = {
    'LoRe': Pipeline([
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ]),
    'Dtree': Pipeline([
        ('clf', DecisionTreeClassifier(random_state=42))
    ]),
    'RaFo': Pipeline([
        ('clf', RandomForestClassifier(random_state=42, n_jobs=1))
    ]),
    'XGBo': Pipeline([
        ('clf', xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        ))
    ]),
}

# Define parameter grids - modify tuning parameters as needed to control overfitting
param_grids = {
    'LoRe': {
        'clf__C': [0.01, 0.1, 1, 10, 100],
        'clf__solver': ['liblinear', 'lbfgs']
    },
    'Dtree': {
        'clf__max_depth': [3, 5, 10, 15],
        'clf__min_samples_leaf': [10, 20, 50],
        'clf__min_samples_split': [20, 50, 100],
        'clf__criterion': ['gini', 'entropy']
    },
    'RaFo': {
        'clf__n_estimators': [100, 200, 500],
        'clf__max_depth': [5, 10],
        'clf__max_features': ['sqrt', 'log2']
    },
    'XGBo': {
        'clf__n_estimators': [500, 800, 1000],
        'clf__max_depth': [3, 4, 5],
        'clf__learning_rate': [0.01, 0.03, 0.05],

        'clf__min_child_weight': [1, 5, 10],
        'clf__gamma': [0.5, 1, 5],
        'clf__subsample': [0.6, 0.8],
        'clf__colsample_bytree': [0.6, 0.8],

        'clf__reg_alpha': [0, 0.1, 1],
        'clf__reg_lambda': [1, 5, 10],
    }
}

# print(">>> User settings defined \n")


def run_combo(args):
    combo, df, y_all, groups_all, approach, outer_cv, inner_cv, model_pipelines, param_grids = args
    results = []

    # print(f"\n ▶ Starting combo: {combo} \n")
    t_start_combo = time.time()

    X_full = df[list(combo)].reset_index(drop=True)
    # Ensure groups_all is aligned with X_full
    # If df_train_val was already filtered by boat_trip_id, this is just to reset index
    groups_all_aligned = df['boat_trip_id'].reset_index(drop=True)
    y_full = y_all  # y_all is already derived from df_train_val target

    split_args = (X_full, y_full, groups_all_aligned) if approach == 'b' else (
        X_full, y_full)

    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(*split_args), start=1):
        X_tr = X_full.loc[train_idx].copy()
        X_va = X_full.loc[val_idx].copy()
        y_tr = y_full[train_idx]
        y_va = y_full[val_idx]
        groups_inner = groups_all_aligned.loc[train_idx] if approach == 'b' else None

        for var in ['months', 'hours']:
            if var in X_tr.columns:  # Check if the variable is in the current fold's X_tr before dummifying
                dt = pd.get_dummies(X_tr[var], prefix=var)
                dv = pd.get_dummies(X_va[var], prefix=var)
                X_tr = X_tr.drop(columns=var).join(dt)
                X_va = X_va.drop(columns=var).join(dv)
        X_tr, X_va = X_tr.align(X_va, join='outer', axis=1, fill_value=0)
        cls, cnt = np.unique(y_tr, return_counts=True)
        # print(f"Combo={combo}, Fold={fold}, class dist={dict(zip(cls, cnt))} \n")

        for name, pipe in model_pipelines.items():
            print(f"\n ▶ Combo={combo} --- Fold= {fold} --- Model={name}")
            clf = pipe.named_steps['clf']
            if hasattr(clf, 'class_weight'):
                cw = compute_class_weight('balanced', classes=cls, y=y_tr)
                pipe.named_steps['clf'].set_params(
                    class_weight=dict(zip(cls, cw)))

            scorer = make_scorer(
                precision_score, pos_label=fishing_label, average='binary')
            # RandomizedSearchCV
            gs = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=param_grids[name],
                n_iter=30,  # Number of random combinations to try
                cv=inner_cv if approach == 'p' else GroupKFold(
                    n_splits=inner_cv.get_n_splits()).split(X_tr, y_tr, groups=groups_inner),
                scoring=scorer,
                n_jobs=1,
                refit=True,
                random_state=42
            )

            t0 = time.time()
            gs.fit(X_tr, y_tr)
            fit_time = time.time() - t0

            best_model = gs.best_estimator_
            y_tr_pred = best_model.predict(X_tr)
            y_va_pred = best_model.predict(X_va)
            # Probabilities for the positive class
            y_va_proba = best_model.predict_proba(X_va)[:, 1]

            results.append({
                'combo': ','.join(combo),
                'fold': fold,
                'model': name,
                'best_params': gs.best_params_,
                'fit_time_s': round(fit_time, 2),
                'accuracy_train': accuracy_score(y_tr, y_tr_pred),
                'precision_train': precision_score(y_tr, y_tr_pred, average='binary', zero_division=0, pos_label=fishing_label),
                'recall_train': recall_score(y_tr, y_tr_pred, average='binary', zero_division=0, pos_label=fishing_label),
                'f1_train': f1_score(y_tr, y_tr_pred, average='binary', zero_division=0, pos_label=fishing_label),
                'specificity_train': recall_score(y_tr, y_tr_pred, average='binary', zero_division=0, pos_label=1 - fishing_label),
                'auc_train': roc_auc_score(y_tr, best_model.predict_proba(X_tr)[:, 1]) if len(np.unique(y_tr)) > 1 else np.nan,
                'accuracy_val': accuracy_score(y_va, y_va_pred),
                'precision_val': precision_score(y_va, y_va_pred, average='binary', zero_division=0, pos_label=fishing_label),
                'recall_val': recall_score(y_va, y_va_pred, average='binary', zero_division=0, pos_label=fishing_label),
                'f1_val': f1_score(y_va, y_va_pred, average='binary', zero_division=0, pos_label=fishing_label),
                'specificity_val': recall_score(y_va, y_va_pred, average='binary', zero_division=0, pos_label=1 - fishing_label),
                'auc_val': roc_auc_score(y_va, y_va_proba) if len(np.unique(y_va)) > 1 else np.nan
            })

    t_end_combo = time.time()
    print(
        f"\n >>> ⏱ Combo {combo} completed in {round(t_end_combo - t_start_combo, 2)}s \n")
    return results


if __name__ == '__main__':
    print("\n ----- Starting parallelization ----- \n")

    all_args = []

    # Select the chosen variables combination
    combo = columns_to_train.copy()
    print(f" \n Combo: {combo}")

    all_args.append((
        combo,
        df_train_val,
        y_train_val,
        groups_train_val,
        approach,
        outer_cv,
        inner_cv,
        model_pipelines,
        param_grids
    ))
    print(f"\n Total combinations to run: {len(all_args)}\n")

    # Run in parallel using available cores (leave 1 core free)
    # or set the numbers of core you want to use directly (e.g., 4)
    num_workers = max(cpu_count() - 1, 1)
    results_flat = []

    with Pool(processes=min(num_workers, cpu_count())) as pool:
        with tqdm(total=len(all_args)) as pbar:
            for result_group in pool.imap_unordered(run_combo, all_args):
                results_flat.extend(result_group)
                pbar.update(1)

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results_flat)

    # ---------------- Final evaluation for each best (model, combo) on test set ----------------

    test_results = []

    # Encode target for the test set using the fitted encoder
    y_test = label_encoder_overall.transform(df_test['target'])

    # Loop over each unique (model, combo) pair
    # In this case, only 4 loop corresponding to the 4 models (given that we have a unique combo)
    for (model_name, combo), group_df in results_df.groupby(['model', 'combo']):
        print(f"Evaluating final test for model={model_name}, combo={combo}")

        # Select the best fold (based on precision_val) for this model-combo pair
        best_row = group_df.sort_values(
            'precision_val', ascending=False).iloc[0]
        best_params = best_row['best_params']
        combo_list = combo.split(',')

        # Prepare X_train (using df_train_val for the final re-training) and X_test
        X_train = df_train_val[combo_list].copy()
        X_test = df_test[combo_list].copy()

        # Handle categorical features (months, hours)
        if 'months' in combo_list:
            train_months_dummies = pd.get_dummies(
                df_train_val['months'], prefix='months')
            test_months_dummies = pd.get_dummies(
                df_test['months'], prefix='months')
            X_train = X_train.drop(columns='months').join(train_months_dummies)
            X_test = X_test.drop(columns='months').join(test_months_dummies)

        if 'hours' in combo_list:
            train_hourss_dummies = pd.get_dummies(
                df_train_val['hours'], prefix='hours')
            test_hourss_dummies = pd.get_dummies(
                df_test['hours'], prefix='hours')
            X_train = X_train.drop(columns='hours').join(train_hourss_dummies)
            X_test = X_test.drop(columns='hours').join(test_hourss_dummies)

        # After one-hot encoding, feature_names should be updated for correct feature importance mapping
        feature_names = list(X_train.columns)

        # Align columns between train and test
        X_train, X_test = X_train.align(
            X_test, join='outer', axis=1, fill_value=0)

        # Clone model
        model = clone(model_pipelines[model_name])
        # Set best params found during CV and fit
        try:
            if isinstance(model, Pipeline):
                model.set_params(**best_params)
            else:
                model.set_params(
                    **{k.replace('clf__', ''): v for k, v in best_params.items()})

        except Exception as e:
            print(
                f"Error during setting model parameters for {model_name}: {e}")
            continue

        # Calculate Weights for the full train_val set
        cls_final, counts_final = np.unique(y_train_val, return_counts=True)

        # Apply weights based on model type
        clf_final = model.named_steps['clf'] if isinstance(
            model, Pipeline) else model

        if model_name == 'XGBo':
            # Counts for classes
            pos_count = counts_final[np.where(cls_final == 1)[
                0][0]]  # Not Fishing (label = 1)
            # Fishing (label = 0, the Target)
            neg_count = counts_final[np.where(cls_final == 0)[0][0]]

            # To boost Recall for Class 0 (Fishing), we need to give it more weight.
            # Since scale_pos_weight scales Class 1, we provide a ratio SMALLER than 1
            # to down-weight the majority class, effectively making the model more sensitive to Class 0.
            ratio = neg_count / pos_count
            clf_final.set_params(scale_pos_weight=ratio)

        elif hasattr(clf_final, 'class_weight'):
            # Standard sklearn "balanced" weights
            cw = compute_class_weight(
                'balanced', classes=cls_final, y=y_train_val)
            clf_final.set_params(class_weight=dict(zip(cls_final, cw)))

        # Measure training time
        start_time = time.time()
        model.fit(X_train, y_train_val)
        fit_time_s = time.time() - start_time

        # Predict labels
        y_pred_test = model.predict(X_test)
        y_pred_proba_test = model.predict_proba(X_test)[:, 1]

        ### Save predictions and probabilities to original dataset ###
        # Storing results for the test set
        own_data.loc[df_test.index, f'target_pred_{model_name}'] = y_pred_test
        own_data.loc[df_test.index,
                     f'target_prob_{model_name}'] = y_pred_proba_test

        # Storing results for the train_val set for the final best model
        y_pred_train = model.predict(X_train)
        y_pred_proba_train = model.predict_proba(X_train)[:, 1]
        own_data.loc[df_train_val.index,
                     f'target_pred_{model_name}'] = y_pred_train
        own_data.loc[df_train_val.index,
                     f'target_prob_{model_name}'] = y_pred_proba_train

        # Test metrics
        acc_test = accuracy_score(y_test, y_pred_test)
        prec_test = precision_score(
            y_test, y_pred_test, average='binary', zero_division=0, pos_label=fishing_label)
        rec_test = recall_score(y_test, y_pred_test,
                                average='binary', zero_division=0, pos_label=fishing_label)
        f1_test = f1_score(y_test, y_pred_test, average='binary',
                           zero_division=0, pos_label=fishing_label)
        specificity_test = recall_score(
            y_test, y_pred_test, average='binary', zero_division=0, pos_label=1 - fishing_label)
        auc_test = roc_auc_score(y_test, y_pred_proba_test) if len(
            np.unique(y_test)) > 1 else np.nan

        # Recording SHAP feature importance for the test set
        shap_feature_importance = {}
        clf_model = model.named_steps['clf'] if isinstance(
            model, Pipeline) else model

        try:
            # --- 1. CALCULATE RAW SHAP VALUES ---
            if isinstance(clf_model, (DecisionTreeClassifier, RandomForestClassifier, xgb.XGBClassifier)):
                explainer = shap.TreeExplainer(clf_model)
                shap_values_raw = explainer.shap_values(X_test)

            elif isinstance(clf_model, LogisticRegression):
                X_test_numeric = np.array(X_test).astype(float)
                explainer = shap.LinearExplainer(clf_model, X_test_numeric)
                shap_values_raw = explainer.shap_values(X_test_numeric)

            else:
                print(f"Warning: No explainer for {model_name}")
                shap_values_raw = None

            # --- 2. PROCESS & AGGREGATE ALL THE LEVELS OF CATEGORICAL VARIABLES FOR BEESWARM ---
            if shap_values_raw is not None:
                # Case A: Model returns a list (usually Dtree, RaFo)
                if isinstance(shap_values_raw, list):
                    shap_values_positive_class = shap_values_raw[fishing_label]

                # Case B: Model returns a 3D array (some versions of TreeExplainer)
                elif len(shap_values_raw.shape) == 3:
                    shap_values_positive_class = shap_values_raw[:,
                                                                 :, fishing_label]

                # Case C: Model returns a single 2D array (XGBo, LoRe)
                # These typically default to Class 1. If our target is Class 0, we must negate.
                else:
                    if fishing_label == 0:
                        # Negating flips the plot: what was "Not Fishing" on the right
                        # now becomes "Fishing" on the right.
                        shap_values_positive_class = -1 * shap_values_raw
                    else:
                        shap_values_positive_class = shap_values_raw

                X_df = pd.DataFrame(X_test, columns=feature_names)

                # We will build these lists to contain EVERY variable for the plot
                final_shap_list = []
                final_val_list = []
                final_names = []

                # A. Handle Categorical/Aggregated Variables (hours and months)
                for agg_var in ['hours', 'months']:
                    cols = [c for c in feature_names if c.startswith(
                        f'{agg_var}_')]
                    if cols:
                        idx_cols = [feature_names.index(c) for c in cols]
                        # Sum SHAP values across all dummy columns for those variable
                        final_shap_list.append(
                            shap_values_positive_class[:, idx_cols].sum(axis=1))
                        # Get the original value (the one that was 1) to represent the feature value
                        label = agg_var
                        final_val_list.append(X_df[cols].idxmax(
                            axis=1).str.replace(f'{agg_var}_', '').astype(float))
                        final_names.append(label)

                # B. Handle Continuous/Simple Variables
                # (Anything in combo_list that isn't hours or months)
                simple_vars = [
                    v for v in combo_list if v not in ['hours', 'months']]
                for var in simple_vars:
                    if var in feature_names:
                        idx_var = feature_names.index(var)
                        final_shap_list.append(
                            shap_values_positive_class[:, idx_var])
                        final_val_list.append(X_df[var].values)
                        final_names.append(var)

                # Generate the Full Beeswarm plot
                if final_shap_list:
                    # Increased size to accommodate all variables
                    plt.figure(figsize=(10, 8))
                    full_shap_matrix = np.column_stack(final_shap_list)
                    full_X_matrix = np.column_stack(final_val_list)

                    shap.summary_plot(
                        full_shap_matrix,
                        full_X_matrix,
                        feature_names=final_names,
                        plot_type="dot",
                        show=False
                    )
                    plt.title(
                        f"Global Feature Importance (SHAP): {model_name}")
                    plt.savefig(os.path.join(
                        basedir, outdir, f"beeswarm_{model_name}_full.png"), bbox_inches='tight')
                    plt.close()

                # --- 3. LOG GLOBAL IMPORTANCE ---
                # This keeps the original breakdown of one-hot encoded columns for the CSV log
                # to see the importance of each level of the categorical variables
                mean_abs_shap_values = np.abs(
                    shap_values_positive_class).mean(axis=0)
                shap_feature_importance = {f: float(v) for f, v in zip(
                    feature_names, mean_abs_shap_values)}

        except Exception as e:
            print(
                f"Warning: SHAP calculation failed for {model_name}. Error: {e}")
            shap_feature_importance = None

        # Save results
        test_results.append({
            'combo': combo,
            'fold': 'final_test',
            'model': model_name,
            'best_params': best_params,
            'fit_time_s': fit_time_s,
            'accuracy_train': np.nan,
            'precision_train': np.nan,
            'recall_train': np.nan,
            'f1_train': np.nan,
            'specificity_train':  np.nan,
            'auc_train': np.nan,
            'accuracy_val': np.nan,
            'precision_val': np.nan,
            'recall_val': np.nan,
            'f1_val': np.nan,
            'specificity_val': np.nan,
            'auc_val': np.nan,
            'accuracy_test': acc_test,
            'precision_test': prec_test,
            'recall_test': rec_test,
            'f1_test': f1_test,
            'specificity_test': specificity_test,
            'auc_test': auc_test,
            'feature_importance': shap_feature_importance
        })

    # Add final test results to results_df
    results_df = pd.concat(
        [results_df, pd.DataFrame(test_results)], ignore_index=True)

    # Save output
    output_filename = f"model_performances_on_data_splitting_with_shap.csv"
    output_path = os.path.join(basedir, outdir, output_filename)
    results_df.to_csv(output_path, index=False)

    # Save the updated main dataset with "set", "target_pred", and "target_prob" columns
    prediction_dataset_path_rds = os.path.join(
        basedir, outdir, "dataset_with_predictions.rds")
    pyreadr.write_rds(prediction_dataset_path_rds, own_data)

    print(f"\n✅ Done! All model-combo test results saved to {output_path}")
    print("\n ----- End parallelization -----")
