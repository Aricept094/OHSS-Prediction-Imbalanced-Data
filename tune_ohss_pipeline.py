"""
This script performs hyperparameter optimization for predicting Ovarian 
Hyperstimulation Syndrome (OHSS) using a machine learning pipeline. 

It leverages Ray Tune with Optuna for efficient search over a large space 
of preprocessing steps, oversampling techniques (from the smote-variants library),
feature selection options, and classifier hyperparameters.

The pipeline involves:
1. Loading training and testing data (Excel format).
2. Defining a search space including:
    - Data scaling (MinMaxScaler, RobustScaler, StandardScaler, or none).
    - Normalization (Yeo-Johnson or none - currently limited to none).
    - A wide variety of oversampling algorithms from `smote-variants`.
    - Optional feature selection.
    - Selection and hyperparameter tuning for multiple classifiers 
      (SVM, Ridge, Logistic Regression, Random Forest, XGBoost, SGD, KNN).
      (Currently LightGBM and MLP are disabled in the search space but can be enabled).
3. Defining a `train_model` function compatible with Ray Tune, which:
    - Takes a configuration dictionary (`config`) sampled from the search space.
    - Applies the selected preprocessing and oversampling.
    - Conditionally builds a VotingClassifier ensemble based on the config.
    - Trains the ensemble on the processed (and potentially oversampled) data.
    - Evaluates the ensemble on the preprocessed test set using a custom 
      weighted recall metric (`recall_mean`).
    - Handles various potential errors during preprocessing or training.
    - Logs intermediate data, models, and evaluation results for each trial.
    - Reports the performance metric back to Ray Tune.
4. Running the Ray Tune optimization process using OptunaSearch and ASHAScheduler.

Key Libraries Used:
- pandas: Data manipulation.
- sklearn: Classifiers, preprocessing, metrics, model selection.
- lightgbm, xgboost: Gradient boosting classifiers.
- smote_variants: Oversampling algorithms.
- ray[tune]: Distributed hyperparameter optimization framework.
- optuna: Search algorithm for Ray Tune.
- joblib: Saving/loading Python objects (like models).
- numpy, scipy: Numerical operations.
- os, datetime: File system operations and timestamping.

**Important Note on Library Versions:**
Machine learning libraries like XGBoost, LightGBM, scikit-learn, etc., are frequently 
updated. These updates can sometimes introduce changes to parameter names, valid 
values, default behaviors, or conditions under which certain parameters are applicable 
(e.g., a parameter only being valid for a specific 'booster' type). 

If you encounter errors related to hyperparameters (e.g., "unexpected keyword argument", 
"invalid parameter value", errors related to conditional parameters), especially when 
enabling currently disabled models (like LightGBM or MLP via the `use_*` flags) or 
when using newer versions of the libraries than originally tested:

1.  **Check the Error Message:** It often indicates the specific parameter causing the issue.
2.  **Consult Documentation:** Refer to the official documentation for the *specific version* 
    of the library you are using (e.g., XGBoost docs, LightGBM docs). Look for changes 
    in the API, deprecated parameters, or updated conditions for the parameters defined 
    in the `search_space` below.
3.  **Temporarily Disable Models:** You can set the corresponding `use_*` flag in the 
    `search_space` to `tune.choice([False])` to temporarily disable a problematic model 
    while debugging others.
4.  **Adjust Search Space:** Modify the `search_space` in this script according to the 
    latest library documentation if parameters have changed.
"""
# === Imports ===
from lightgbm import LGBMClassifier
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import numpy as np
import pandas as pd  
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from ray import train
from ray.tune.search.optuna import OptunaSearch
from xgboost import XGBClassifier
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
import os
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import gmean
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import smote_variants as sv # Library for various SMOTE-based oversampling techniques
import joblib # For saving and loading Python objects (e.g., trained models)


datetime_folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_directory_base = f"[Your_Path]/{datetime_folder_name}"
os.makedirs(log_directory_base, exist_ok=True)


train_data_path = "[Your_Path]/train_data.xlsx"
test_data_path = "[Your_Path]/test_data.xlsx"

train_df = pd.read_excel(train_data_path)
#train_df = train_df.dropna()

test_df = pd.read_excel(test_data_path)
    
def save_dataframe_to_feather(df, directory, filename):
    """
    Saves a DataFrame to a Feather file in the specified directory using pyarrow.
    """
    filepath = os.path.join(directory, filename)
    df.to_feather(filepath)
    
def save_dataframe_to_csv(df, directory, filename):
    """
    Saves a DataFrame to a CSV file in the specified directory.
    """
    filepath = os.path.join(directory, filename)
    df.to_csv(filepath, index=False)


def recall_mean(y_test, y_pred, weight_class_1=0.55):
    
    recall_0 = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    recall_1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall_mean = weight_class_1 * recall_1 + (1 - weight_class_1) * recall_0
    
    return recall_mean
        
def train_model(config):
    X_train = train_df.drop(columns=['OHSS'])
    y_train = train_df['OHSS']
    X_test = test_df.drop(columns=['OHSS'])
    y_test = test_df['OHSS']

    df = pd.concat([X_train, X_test], ignore_index=True)
    X = pd.concat([X_train, X_test], ignore_index=True) 
    y = pd.concat([y_train, y_test], ignore_index=True)   

    feature_list = [
    "age", "weight", "Height", "Durationinfertility", "FSH", "LH", "numberofcunsumptiondrug", 
    "durationofstimulation", "numberRfulicule", "numberLfulicule", "numberofoocyte", 
    "metagha1oocyte", "metaghaze2oocyte", "necrozeoocyte", "lowQualityoocyte", "Gvoocyte", 
    "macrooocyte", "partogenesoocyte", "numberembrio", "countspermogram", "motilityspermogram", 
    "morfologyspermogram", "gradeembrio", "Typecycle", "reasoninfertility", "Typeofcunsumptiondrug", 
    "typeoftrigger", "Typedrug", "pregnancy", "mense", "Infertility"]
    
    feature_list_target = ["OHSS"]
    
    columns_to_scale = [
    'age', 'weight', 'Height', 'Durationinfertility', 'FSH', 'LH',
        'numberofcunsumptiondrug', 'durationofstimulation', 'numberRfulicule',
        'numberLfulicule', 'numberofoocyte', 'metagha1oocyte',
        'metaghaze2oocyte', 'necrozeoocyte', 'lowQualityoocyte', 'Gvoocyte',
        'macrooocyte', 'partogenesoocyte', 'numberembrio', 'countspermogram',
        'motilityspermogram', 'morfologyspermogram']

    if config.get('feature_selection_choice') == 'selector_1':
        features_to_use = [feature for feature in feature_list if config.get(feature) == True]
    else:
        features_to_use = df.columns.tolist()

    columns_to_scale = [col for col in columns_to_scale if col in features_to_use]
    
    transformers = []

    normalization_choice = config.get('normalization_choice')
    if normalization_choice == 'yeo-johnson':
        transformers.append(('yeo-johnson', PowerTransformer(method='yeo-johnson'), columns_to_scale))
    elif normalization_choice == 'passthrough':
        pass

    scaler_choice = config.get('scaler_choice')
    if scaler_choice in ['StandardScaler', 'MinMaxScaler', 'RobustScaler']:
        scaler = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler()
        }[scaler_choice]
        transformers.append((scaler_choice, scaler, columns_to_scale))
    elif scaler_choice == 'passthrough':
        pass

    if not transformers:
        transformers.append(('passthrough', 'passthrough', features_to_use))

    preprocessor = ColumnTransformer(transformers, remainder='passthrough')
    
    scores = []
    smote_classifiers = []  
    voting_estimators = [] 
    
    if config.get('use_rf'):
        rf_params = {
            'max_depth': config['rfc_max_depth'],
            'n_estimators': config['rf_n_estimators'],
            'min_samples_split': config['rf_min_samples_split'],
            'min_samples_leaf': config['rf_min_samples_leaf'],
            'criterion': config['rf_criterion'],
            'min_weight_fraction_leaf': config['rf_min_weight_fraction_leaf'],
            'max_leaf_nodes': config['rf_max_leaf_nodes'],
            'min_impurity_decrease': config['rf_min_impurity_decrease'],
            'bootstrap': config['rf_bootstrap'],
            'class_weight': config['rf_class_weight'],
            'ccp_alpha': config['rf_ccp_alpha']
        }

        #smote_classifiers.append(('sklearn.ensemble', 'RandomForestClassifier', rf_params))

        voting_estimators.append(('rf', RandomForestClassifier(**rf_params)))

    if config.get('use_xgboost'):
        xgb_params = {
            'n_estimators': config['xgb_n_estimators'],
            'max_depth': config['xgb_max_depth'],
            'learning_rate': config['xgb_learning_rate'],
            'grow_policy': config['xgb_grow_policy'],
            'gamma': config['xgb_gamma'],
            'min_child_weight': config['xgb_min_child_weight'],
            'max_delta_step': config['xgb_max_delta_step'],
            'subsample': config['xgb_subsample'],
            'colsample_bytree': config['xgb_colsample_bytree'],
            'colsample_bylevel': config['xgb_colsample_bylevel'],
            'colsample_bynode': config['xgb_colsample_bynode'],
            'reg_lambda': config['xgb_lambda'],
            'reg_alpha': config['xgb_alpha'],
            'booster': config.get('xgb_booster'),
            'scale_pos_weight': config.get('xgb_scale_pos_weight'),
            'tree_method': config.get('xgb_tree_method'),
            'sample_type': config.get('xgb_sample_type'),
            'normalize_type': config.get('xgb_normalize_type'),
            'rate_drop': config.get('xgb_rate_drop', 0.0),
            'one_drop': config.get('xgb_one_drop', False),
            'skip_drop': config.get('xgb_skip_drop', 0.0)
        }

        #smote_classifiers.append(('xgboost', 'XGBClassifier', xgb_params))

        voting_estimators.append(('xgb', XGBClassifier(**xgb_params)))
  
    if config.get('use_lightgbm'):
        lgbm_params = {
            'objective': 'binary',
            'boosting_type': config['lgbm_boosting_type'],
            'num_leaves': config['lgbm_num_leaves'],
            'max_depth': config['lgbm_max_depth'],
            'learning_rate': config['lgbm_learning_rate'],
            'n_estimators': config['lgbm_n_estimators'],
            'data_sample_strategy': config['lgbm_data_sample_strategy'],
            'class_weight': 'balanced',
            'subsample': config['lgbm_subsample'],
            'colsample_bytree': config['lgbm_colsample_bytree'],
            'reg_alpha': config['lgbm_reg_alpha'],
            'reg_lambda': config['lgbm_reg_lambda'],
            'bagging_freq': config['lgbm_bagging_freq'],
            'bagging_fraction': config['lgbm_bagging_fraction'],
            'feature_fraction': config['lgbm_feature_fraction'],
            'min_data_in_leaf': config['lgbm_min_data_in_leaf'],
            'verbose': -1
        }

        #smote_classifiers.append(('lightgbm', 'LGBMClassifier', lgbm_params))

        voting_estimators.append(('lgbm', LGBMClassifier(**lgbm_params)))
        
    if config.get('use_mlp'):
        mlp_params = {
            'hidden_layer_sizes': config['mlp_hidden_layer_sizes'],
            'activation': config['mlp_activation'],
            'solver': config['mlp_solver'],
            'alpha': config['mlp_alpha'],
            'learning_rate': config['mlp_learning_rate'],
            'max_iter': 500,
            'tol': config['mlp_tol']
        }

        #smote_classifiers.append(('sklearn.neural_network', 'MLPClassifier', mlp_params))

        voting_estimators.append(('mlp', MLPClassifier(**mlp_params)))

    if config.get('use_SGD'):
        sgd_params = {
            'loss': config['SGD_loss'],
            'penalty': config['SGD_penalty'],
            'alpha': config['SGD_alpha'],
            'l1_ratio': config['SGD_l1_ratio'],
            'fit_intercept': config['SGD_fit_intercept'],
            'tol': config['SGD_tol'],
            'epsilon': config['SGD_epsilon'],
            'learning_rate': config['SGD_learning_rate'],
            'eta0': config['SGD_eta0'],
            'power_t': config['SGD_power_t'],
            'validation_fraction': config['SGD_validation_fraction'],
            'class_weight': 'balanced'
        }

        smote_classifiers.append(('sklearn.linear_model', 'SGDClassifier', sgd_params))

        voting_estimators.append(('sgd', SGDClassifier(**sgd_params)))
        
    if config.get('use_svm'):
        svm_params = {
            'C': config['svm_C'],
            'kernel': config['svm_kernel'],
            'degree': config['svm_degree'],
            'gamma': config['svm_gamma'],
            'coef0': config['svm_coef0'],
            'shrinking': config['svm_shrinking'],
            'probability': True,
            'tol': config['svm_tol'],
            'class_weight': 'balanced',
            'decision_function_shape': config['svm_decision_function_shape']
        }

        smote_classifiers.append(('sklearn.svm', 'SVC', svm_params))

        voting_estimators.append(('svm', SVC(**svm_params)))
    

        
    if config.get('use_knn'):
        knn_params = {
            'n_neighbors': config['knn_n_neighbors'],
            'weights': config['knn_weights'],
            'algorithm': config['knn_algorithm'],
            'leaf_size': config['knn_leaf_size'],
            'p': config['knn_p'],
            'metric' : config['knn_metric']
        }

        smote_classifiers.append(('sklearn.neighbors', 'KNeighborsClassifier', knn_params))

        voting_estimators.append(('knn', KNeighborsClassifier(**knn_params)))
  
    if config.get('use_ridge'):
        ridge_params = {
            'alpha': config['ridge_alpha'],
            'tol': config['ridge_tol'],
            'class_weight': 'balanced',
            'fit_intercept': config['ridge_fit_intercept'],
        }

        #smote_classifiers.append(('sklearn.linear_model', 'RidgeClassifier', ridge_params))

        voting_estimators.append(('ridge', RidgeClassifier(**ridge_params)))

    if config.get('use_logistic'):
        logistic_params = {
            'C': config['logistic_C'],
            'tol': config['logistic_tol'],
            'class_weight': 'balanced',
            'penalty': config['logistic_penalty'],
            'dual': config['logistic_dual'],
            'fit_intercept': config['logistic_fit_intercept'],
            'intercept_scaling': config['logistic_intercept_scaling'],
            'solver': config['logistic_solver'],
            'l1_ratio': config.get('logistic_l1_ratio')
        }

        smote_classifiers.append(('sklearn.linear_model', 'LogisticRegression', logistic_params))

        voting_estimators.append(('logistic', LogisticRegression(**logistic_params)))

        
    classifiers = smote_classifiers

    oversampler_name = config['oversampler']
    oversampler_params = {}

    if oversampler_name == "NEATER":
        oversampler_params['b'] = config.get('NEATER_b')
        oversampler_params['proportion'] = config.get('NEATER_proportion')
        oversampler_params['smote_n_neighbors'] = config.get('NEATER_smote_n_neighbors')        
        oversampler_params['alpha'] = config.get('NEATER_alpha')
        oversampler_params['h'] = config.get('NEATER_h')        
    
    elif oversampler_name == "Lee":
        oversampler_params['n_neighbors'] = config.get('lee_n_neighbors')
        oversampler_params['proportion'] = config.get('Lee_proportion')
        oversampler_params['rejection_level'] = config.get('Lee_proportion')
        
    elif oversampler_name == "SMOTE":
        oversampler_params['proportion'] = config.get('SMOTE_proportion')
        oversampler_params['n_neighbors'] = config.get('SMOTE_n_neighbors')
        
    elif oversampler_name == "SMOTE_TomekLinks":
        oversampler_params['proportion'] = config.get('SMOTE_TomekLinks_proportion')
        oversampler_params['n_neighbors'] = config.get('SMOTE_TomekLinks_n_neighbors')

    elif oversampler_name == "SMOTE_ENN":
        oversampler_params['proportion'] = config.get('SMOTE_ENN_proportion')
        oversampler_params['n_neighbors'] = config.get('SMOTE_ENN_n_neighbors')
        
    elif oversampler_name == "Borderline_SMOTE1":
        oversampler_params['proportion'] = config.get('Borderline_SMOTE1_proportion')
        oversampler_params['n_neighbors'] = config.get('Borderline_SMOTE1_n_neighbors')
        oversampler_params['k_neighbors'] = config.get('Borderline_SMOTE1_k_neighbors')
        
    elif oversampler_name == "distance_SMOTE":
        oversampler_params['proportion'] = config.get('distance_SMOTE_proportion')
        oversampler_params['n_neighbors'] = config.get('distance_SMOTE_n_neighbors')
        
    elif oversampler_name == "ADASYN":
        oversampler_params['proportion'] = config.get('ADASYN_proportion')
        oversampler_params['n_neighbors'] = config.get('ADASYN_n_neighbors')
        oversampler_params['d_th'] = config.get('ADASYN_d_th')
        oversampler_params['beta'] = config.get('ADASYN_beta')

    elif oversampler_name == "Borderline_SMOTE2":
        oversampler_params['proportion'] = config.get('Borderline_SMOTE2_proportion')
        oversampler_params['n_neighbors'] = config.get('Borderline_SMOTE2_n_neighbors')
        oversampler_params['k_neighbors'] = config.get('Borderline_SMOTE2_k_neighbors')
        
    elif oversampler_name == "LLE_SMOTE":
        oversampler_params['proportion'] = config.get('LLE_SMOTE_proportion')
        oversampler_params['n_neighbors'] = config.get('LLE_SMOTE_n_neighbors')
        
        
    elif oversampler_name == "SMMO":
        oversampler_params['proportion'] = config.get('SMMO_proportion')
        oversampler_params['n_neighbors'] = config.get('SMMO_n_neighbors')
        
    elif oversampler_name == "ADOMS":
        oversampler_params['proportion'] = config.get('ADOMS_proportion')
        oversampler_params['n_neighbors'] = config.get('ADOMS_n_neighbors')

    elif oversampler_name == "Safe_Level_SMOTE":
        oversampler_params['proportion'] = config.get('Safe_Level_SMOTE_proportion')
        oversampler_params['n_neighbors'] = config.get('Safe_Level_SMOTE_n_neighbors')
        
    elif oversampler_name == "MSMOTE":
        oversampler_params['proportion'] = config.get('MSMOTE_proportion')
        oversampler_params['n_neighbors'] = config.get('MSMOTE_n_neighbors')

    elif oversampler_name == "SMOBD":
        oversampler_params['proportion'] = config.get('SMOBD_proportion')
        oversampler_params['eta1'] = config.get('SMOBD_eta1')
        oversampler_params['t'] = config.get('SMOBD_t')
        oversampler_params['min_samples'] = config.get('SMOBD_min_samples')
        oversampler_params['max_eps'] = config.get('SMOBD_max_eps')
        
    elif oversampler_name == "SUNDO":
        oversampler_params = {}

    elif oversampler_name == "DE_oversampling":
        oversampler_params['proportion'] = config.get('DE_oversampling_proportion')
        oversampler_params['n_neighbors'] = config.get('DE_oversampling_n_neighbors')
        oversampler_params['crossover_rate'] = config.get('DE_oversampling_crossover_rate')
        oversampler_params['similarity_threshold'] = config.get('DE_oversampling_similarity_threshold')
        oversampler_params['n_clusters'] = config.get('DE_oversampling_n_clusters')
        
    elif oversampler_name == "MSYN":
        oversampler_params['proportion'] = config.get('MSYN_proportion')
        oversampler_params['n_neighbors'] = config.get('MSYN_n_neighbors')
        oversampler_params['pressure'] = config.get('MSYN_SMOTE_n_neighbors')
        
    elif oversampler_name == "SVM_balance":
        oversampler_params['proportion'] = config.get('SVM_balance_proportion')
        oversampler_params['n_neighbors'] = config.get('SVM_balance_n_neighbors')
        
    elif oversampler_name == "TRIM_SMOTE":
        oversampler_params['proportion'] = config.get('TRIM_SMOTE_proportion')
        oversampler_params['n_neighbors'] = config.get('TRIM_SMOTE_n_neighbors')
        oversampler_params['min_precision'] = config.get('TRIM_SMOTE_min_precision')
        
    elif oversampler_name == "SMOTE_RSB":
        oversampler_params['proportion'] = config.get('SMOTE_RSB_proportion')
        oversampler_params['n_neighbors'] = config.get('SMOTE_RSB_n_neighbors')
        
    elif oversampler_name == "ProWSyn":
        oversampler_params['proportion'] = config.get('ProWSyn_proportion')
        oversampler_params['n_neighbors'] = config.get('ProWSyn_n_neighbors')
        oversampler_params['L'] = config.get('ProWSyn_L')
        oversampler_params['theta'] = config.get('ProWSyn_theta')

    elif oversampler_name == "SL_graph_SMOTE":
        oversampler_params['proportion'] = config.get('SL_graph_SMOTE_proportion')
        oversampler_params['n_neighbors'] = config.get('SL_graph_SMOTE_n_neighbors')
        
    elif oversampler_name == "NRSBoundary_SMOTE":
        oversampler_params['proportion'] = config.get('NRSBoundary_SMOTE_proportion')
        oversampler_params['n_neighbors'] = config.get('NRSBoundary_SMOTE_n_neighbors')
        
    elif oversampler_name == "LVQ_SMOTE":
        oversampler_params['proportion'] = config.get('LVQ_SMOTE_proportion')
        oversampler_params['n_neighbors'] = config.get('LVQ_SMOTE_n_neighbors')
        oversampler_params['n_clusters'] = config.get('LVQ_SMOTE_n_clusters')
        
    elif oversampler_name == "SOI_CJ":
        oversampler_params['proportion'] = config.get('SOI_CJ_proportion')
        oversampler_params['n_neighbors'] = config.get('SOI_CJ_n_neighbors')
        
    elif oversampler_name == "ROSE":
        oversampler_params['proportion'] = config.get('ROSE_SMOTE_proportion')

    elif oversampler_name == "SMOTE_OUT":
        oversampler_params['proportion'] = config.get('SMOTE_OUT_proportion')
        oversampler_params['n_neighbors'] = config.get('SMOTE_OUT_n_neighbors')

    elif oversampler_name == "SMOTE_Cosine":
        oversampler_params['proportion'] = config.get('SMOTE_Cosine_proportion')
        oversampler_params['n_neighbors'] = config.get('SMOTE_Cosine_n_neighbors')
        
    elif oversampler_name == "MWMOTE":
        oversampler_params['proportion'] = config.get('SL_graph_SMOTE_proportion')
        oversampler_params['k1'] = config.get('SL_graph_SMOTE_k1')
        oversampler_params['k2'] = config.get('SL_graph_SMOTE_k2')
        oversampler_params['k3'] = config.get('SL_graph_SMOTE_k3') 
        oversampler_params['M'] = config.get('SL_graph_SMOTE_M')
        oversampler_params['cf_th'] = config.get('SL_graph_SMOTE_cf_th')       
        oversampler_params['cmax'] = config.get('SL_graph_SMOTE_cmax')  
        
    elif oversampler_name == "Selected_SMOTE":
        oversampler_params['proportion'] = config.get('Selected_SMOTE_proportion')
        oversampler_params['n_neighbors'] = config.get('Selected_SMOTE_n_neighbors')
        oversampler_params['perc_sign_attr'] = config.get('Selected_SMOTE_perc_sign_attr')
        
    elif oversampler_name == "LN_SMOTE":
        oversampler_params['proportion'] = config.get('LN_SMOTE_proportion')
        oversampler_params['n_neighbors'] = config.get('LN_SMOTE_n_neighbors')
        
    elif oversampler_name == "PDFOS":
        oversampler_params['proportion'] = config.get('PDFOS_SMOTE_proportion')

    elif oversampler_name == "IPADE_ID":
        oversampler_params['proportion'] = config.get('IPADE_ID_proportion')
        oversampler_params['F'] = config.get('IPADE_ID_SMOTE_F')
        oversampler_params['G'] = config.get('IPADE_ID_SMOTE_G')
        oversampler_params['OT'] = config.get('IPADE_ID_SMOTE_OT') 
        oversampler_params['max_it'] = config.get('IPADE_ID_SMOTE_max_it')

    elif oversampler_name == "RWO_sampling":
        oversampler_params['proportion'] = config.get('RWO_sampling_proportion')
        
    elif oversampler_name == "DEAGO":
        oversampler_params['proportion'] = config.get('DEAGO_proportion')
        oversampler_params['n_neighbors'] = config.get('DEAGO_n_neighbors')
        oversampler_params['e'] = config.get('DEAGO_e')
        oversampler_params['h'] = config.get('DEAGO_h')
        oversampler_params['sigma'] = config.get('DEAGO_sigma')
        
    elif oversampler_name == "Gazzah":
        oversampler_params['proportion'] = config.get('Gazzah_proportion')
        oversampler_params['n_components'] = config.get('Gazzah_n_components')
        
    elif oversampler_name == "ADG":
        oversampler_params['proportion'] = config.get('ADG_proportion')
        oversampler_params['lam'] = config.get('ADG_lam')
        oversampler_params['mu'] = config.get('ADG_mu')
        oversampler_params['k'] = config.get('ADG_k')
        oversampler_params['gamma'] = config.get('ADG_gamma')
        
    elif oversampler_name == "MCT":
        oversampler_params['proportion'] = config.get('MCT_proportion')
        oversampler_params['n_neighbors'] = config.get('MCT_n_neighbors')

    elif oversampler_name == "SMOTE_IPF":
        oversampler_params['proportion'] = config.get('SMOTE_IPF_proportion')
        oversampler_params['n_neighbors'] = config.get('SMOTE_IPF_n_neighbors')
        oversampler_params['n_folds'] = config.get('SMOTE_IPF_n_folds')
        oversampler_params['k'] = config.get('SMOTE_IPF_k')
        oversampler_params['p'] = config.get('SMOTE_IPF_p')
        

    elif oversampler_name == "KernelADASYN":
        oversampler_params['proportion'] = config.get('KernelADASYN_proportion')
        oversampler_params['k'] = config.get('KernelADASYN_k')
        oversampler_params['h'] = config.get('KernelADASYN_k')
      
    elif oversampler_name == "SMOTE_PSO":
        oversampler_params['k'] = config.get('SMOTE_PSO_k')
        oversampler_params['eps'] = config.get('SMOTE_PSO_eps')
        oversampler_params['n_pop'] = config.get('SMOTE_PSO_n_pop')
        oversampler_params['w'] = config.get('SMOTE_PSO_w')
        oversampler_params['c1'] = config.get('SMOTE_PSO_c1')
        oversampler_params['c2'] = config.get('SMOTE_PSO_c2')
        oversampler_params['num_it'] = config.get('SMOTE_PSO_num_it')
        
    elif oversampler_name == "SOMO":
        oversampler_params['proportion'] = config.get('SOMO_proportion')
        oversampler_params['n_neighbors'] = config.get('SOMO_n_neighbors')

    elif oversampler_name == "CURE_SMOTE":
        oversampler_params['proportion'] = config.get('CURE_SMOTE_proportion')
        oversampler_params['n_clusters'] = config.get('CURE_SMOTE_n_clusters')
        oversampler_params['noise_th'] = config.get('CURE_SMOTE_noise_th')
        
    elif oversampler_name == "MOT2LD":
        oversampler_params['proportion'] = config.get('MOT2LD_proportion')
        oversampler_params['n_components'] = config.get('MOT2LD_n_components')
        oversampler_params['k'] = config.get('MOT2LD_k')
        
    elif oversampler_name == "ISOMAP_Hybrid":
        oversampler_params['proportion'] = config.get('ISOMAP_Hybrid_proportion')
        oversampler_params['n_neighbors'] = config.get('ISOMAP_Hybrid_n_neighbors')

    elif oversampler_name == "CE_SMOTE":
        oversampler_params['proportion'] = config.get('CE_SMOTE_proportion')
        oversampler_params['alpha'] = config.get('CE_SMOTE_alpha')
        oversampler_params['h'] = config.get('CE_SMOTE_h')
        oversampler_params['k'] = config.get('CE_SMOTE_k')
 
    elif oversampler_name == "SMOTE_PSOBAT":
        oversampler_params['maxit'] = config.get('SMOTE_PSOBAT_maxit')
        oversampler_params['c1'] = config.get('SMOTE_PSOBAT_alpha')
        oversampler_params['c2'] = config.get('SMOTE_PSOBAT_c2')
        oversampler_params['c3'] = config.get('SMOTE_PSOBAT_c3')
        oversampler_params['alpha'] = config.get('SMOTE_PSOBAT_alpha')
        oversampler_params['gamma'] = config.get('SMOTE_PSOBAT_gamma')
               
    elif oversampler_name == "V_SYNTH":
        oversampler_params['proportion'] = config.get('V_SYNTH_proportion')
        oversampler_params['n_neighbors'] = config.get('V_SYNTH_n_neighbors')
        
    elif oversampler_name == "G_SMOTE":
        oversampler_params['proportion'] = config.get('G_SMOTE_proportion')
        oversampler_params['n_neighbors'] = config.get('G_SMOTE_n_neighbors')
        
    elif oversampler_name == "NT_SMOTE":
        oversampler_params['proportion'] = config.get('NT_SMOTE_proportion')

    elif oversampler_name == "SMOTE_D":
        oversampler_params['proportion'] = config.get('SMOTE_D_proportion')
        oversampler_params['n_neighbors'] = config.get('SMOTE_D_n_neighbors')

    elif oversampler_name == "DBSMOTE":
        oversampler_params['proportion'] = config.get('OUPS_proportion')
        oversampler_params['eps'] = config.get('OUPS_eps') 
        oversampler_params['min_samples'] = config.get('OUPS_min_samples')
        oversampler_params['db_iter_limit'] = config.get('OUPS_db_iter_limit')
        
    elif oversampler_name == "Edge_Det_SMOTE":
        oversampler_params['proportion'] = config.get('Edge_Det_SMOTE_proportion')
        oversampler_params['k'] = config.get('Edge_Det_SMOTE_k')
        
    elif oversampler_name == "E_SMOTE":
        oversampler_params['proportion'] = config.get('E_SMOTE_proportion')
        oversampler_params['n_neighbors'] = config.get('E_SMOTE_n_neighbors')
        oversampler_params['min_features'] = config.get('E_SMOTE_min_features')

    elif oversampler_name == "CBSO":
        oversampler_params['proportion'] = config.get('CBSO_proportion')
        oversampler_params['n_neighbors'] = config.get('CBSO_k')
        oversampler_params['C_p'] = config.get('CBSO_C_p')
        
    elif oversampler_name == "Assembled_SMOTE":
        oversampler_params['proportion'] = config.get('Assembled_SMOTE_proportion')
        oversampler_params['n_neighbors'] = config.get('Assembled_SMOTE_n_neighbors')
        oversampler_params['pop'] = config.get('Assembled_SMOTE_pop')
        oversampler_params['thres'] = config.get('Assembled_SMOTE_thres')
        
    elif oversampler_name == "SDSMOTE":
        oversampler_params['proportion'] = config.get('SDSMOTE_proportion')
        oversampler_params['n_neighbors'] = config.get('SDSMOTE_n_neighbors')
        
    elif oversampler_name == "ASMOBD":
        oversampler_params['proportion'] = config.get('ASMOBD_proportion')
        oversampler_params['min_samples'] = config.get('ASMOBD_min_samples')
        oversampler_params['eta'] = config.get('ASMOBD_eta')
        oversampler_params['eps'] = config.get('ASMOBD_eps')
        
    elif oversampler_name == "SPY":
        oversampler_params['n_neighbors'] = config.get('SPY_n_neighbors')

    elif oversampler_name == "MDO":
        oversampler_params['proportion'] = config.get('MDO_proportion')
        oversampler_params['k2'] = config.get('MDO_k2')
        oversampler_params['K1_frac'] = config.get('MDO_K1_frac')
        
    elif oversampler_name == "ISMOTE":
        oversampler_params['n_neighbors'] = config.get('ISMOTE_n_neighbors')
        oversampler_params['minority_weight'] = config.get('ISMOTE_minority_weight')

    elif oversampler_name == "Random_SMOTE":
        oversampler_params['proportion'] = config.get('Random_SMOTE_proportion')
        oversampler_params['n_neighbors'] = config.get('Random_SMOTE_n_neighbors')       

    elif oversampler_name == "VIS_RST":
        oversampler_params['proportion'] = config.get('VIS_RST_proportion')
        oversampler_params['n_neighbors'] = config.get('VIS_RST_n_neighbors')    
        
    elif oversampler_name == "A_SUWO":
        oversampler_params['proportion'] = config.get('A_SUWO_proportion')
        oversampler_params['n_neighbors'] = config.get('A_SUWO_n_neighbors')          
        oversampler_params['n_clus_maj'] = config.get('A_SUWO_n_clus_maj')
        oversampler_params['c_thres'] = config.get('A_SUWO_c_thres')        
        
    elif oversampler_name == "SMOTE_FRST_2T":
        oversampler_params['proportion'] = config.get('SMOTE_FRST_2T_proportion')
        oversampler_params['gamma_M'] = config.get('SMOTE_FRST_2T_gamma_M')          
        oversampler_params['gamma_S'] = config.get('SMOTE_FRST_2T_gamma_S')        
        
    elif oversampler_name == "AND_SMOTE":
        oversampler_params['proportion'] = config.get('AND_SMOTE_proportion')
        oversampler_params['k'] = config.get('AND_SMOTE_k')                   
        
    elif oversampler_name == "NRAS":
        oversampler_params['proportion'] = config.get('NRAS_proportion')
        oversampler_params['n_neighbors'] = config.get('NRAS_n_neighbors')           
        oversampler_params['t'] = config.get('NRAS_t')           
        
    elif oversampler_name == "AMSCO":
        oversampler_params['n_pop'] = config.get('AMSCO_n_pop')
        oversampler_params['n_iter'] = config.get('AMSCO_n_iter')          
        oversampler_params['omega'] = config.get('AMSCO_omega')
        oversampler_params['r1'] = config.get('AMSCO_r1')        
        oversampler_params['r2'] = config.get('AMSCO_r2')        
        
    elif oversampler_name == "NDO_sampling":
        oversampler_params['proportion'] = config.get('NDO_sampling_proportion')
        oversampler_params['n_neighbors'] = config.get('NDO_sampling_n_neighbors')           
        oversampler_params['T'] = config.get('NDO_sampling_T')  
        
    elif oversampler_name == "SSO":
        oversampler_params['proportion'] = config.get('SSO_proportion')
        oversampler_params['n_neighbors'] = config.get('SSO_n_neighbors')          
        oversampler_params['h'] = config.get('SSO_h')
        oversampler_params['n_iter'] = config.get('SSO_n_iter')
        
    elif oversampler_name == "Gaussian_SMOTE":
        oversampler_params['proportion'] = config.get('Gaussian_SMOTE_proportion')
        oversampler_params['n_neighbors'] = config.get('Gaussian_SMOTE_n_neighbors')           
        oversampler_params['sigma'] = config.get('Gaussian_SMOTE_sigma')            
        
    elif oversampler_name == "kmeans_SMOTE":
        oversampler_params['proportion'] = config.get('kmeans_SMOTE_proportion')
        oversampler_params['n_neighbors'] = config.get('kmeans_SMOTE_n_neighbors')          
        oversampler_params['n_clusters'] = config.get('kmeans_SMOTE_n_clusters')
        oversampler_params['irt'] = config.get('kmeans_SMOTE_irt')
        
    elif oversampler_name == "SN_SMOTE":
        oversampler_params['proportion'] = config.get('SN_SMOTE_proportion')
        oversampler_params['n_neighbors'] = config.get('SN_SMOTE_n_neighbors')     
        
    elif oversampler_name == "Supervised_SMOTE":
        oversampler_params['proportion'] = config.get('Supervised_SMOTE_proportion')
        oversampler_params['th_lower'] = config.get('Supervised_SMOTE_th_lower')          
        oversampler_params['th_upper'] = config.get('Supervised_SMOTE_th_upper')          
                                
    elif oversampler_name == "CCR":
        oversampler_params['proportion'] = config.get('CCR_proportion')
        oversampler_params['energy'] = config.get('CCR_energy')          
        oversampler_params['scaling'] = config.get('CCR_scaling')                                  
                                
    elif oversampler_name == "ANS":
        oversampler_params['proportion'] = config.get('ANS_proportion')
                            
    elif oversampler_name == "cluster_SMOTE":
        oversampler_params['proportion'] = config.get('cluster_SMOTE_proportion')
        oversampler_params['n_neighbors'] = config.get('cluster_SMOTE_n_neighbors')          
        oversampler_params['n_clusters'] = config.get('cluster_SMOTE_n_clusters')                                                                   
                                
    elif oversampler_name == "SYMPROD":
        oversampler_params['proportion'] = config.get('SYMPROD_proportion')
        oversampler_params['std_outliers'] = config.get('SYMPROD_std_outliers')          
        oversampler_params['k_neighbors'] = config.get('SYMPROD_k_neighbors')
        oversampler_params['m_neighbors'] = config.get('SYMPROD_m_neighbors')        
        oversampler_params['cutoff_threshold'] = config.get('SYMPROD_cutoff_threshold')                                  
                                
    elif oversampler_name == "SMOTEWB":
        oversampler_params['proportion'] = config.get('SMOTEWB_proportion')
        oversampler_params['max_depth'] = config.get('SMOTEWB_max_depth')          
        oversampler_params['n_iters'] = config.get('SMOTEWB_n_iters')                                    
                                
                                
    oversamplers = [('smote_variants', oversampler_name, oversampler_params)]
    
    X_train_new = X_train[features_to_use]
    
    dataset = {
        'data': X_train_new.values, 
        'target': y_train.values,  
        'name': 'OHSS Dataset'  
    }
    
    samp_obj = None
    X_samp = None
    y_samp = None
    
    if oversamplers and len(oversamplers) > 0:
        try:
            samp_obj, cl_obj = sv.evaluation.model_selection(
                dataset=dataset,
                oversamplers=oversamplers,
                classifiers=classifiers,
                score='f1',
                validator_params={'n_splits': 2, 'n_repeats': 1}
            )

            X_samp, y_samp= samp_obj.sample(dataset['data'], dataset['target'])
        except IndexError as e:
            if "list index out of range" in str(e):
                train.report({"recall_mean": -float('inf')})
        except (TypeError, ValueError, sklearn.utils._param_validation.InvalidParameterError) as e:
            error_message = str(e)

            if "n_components' should be inferior to 4 for the barnes_hut algorithm" in error_message:
                train.report({"recall_mean": -float('inf')})

            elif isinstance(e, TypeError) and (
                "NoneType" in str(e) or 
                "'numpy.float64' object cannot be interpreted as an integer" in str(e) or 
                "'float' object cannot be interpreted as an integer" in str(e)):
                train.report({"recall_mean": -float('inf')})

            elif isinstance(e, sklearn.utils._param_validation.InvalidParameterError) and "n_neighbors" in str(e):
                train.report({"recall_mean": -float('inf')})

            elif isinstance(e, sklearn.utils._param_validation.InvalidParameterError) and "n_clusters" in str(e):
                train.report({"recall_mean": -float('inf')})
                
            elif isinstance(e, sklearn.utils._param_validation.InvalidParameterError) and "n_components" in str(e):
                train.report({"recall_mean": -float('inf')})

            elif isinstance(e, ValueError) and "Shape of passed values" in str(e):
                train.report({"recall_mean": -float('inf')})

            else:
                raise
    else:
        print("No oversamplers provided or oversamplers list is empty. Skipping model selection.")

    
    X_samp = pd.DataFrame(X_samp, columns=features_to_use) if not isinstance(X_samp, pd.DataFrame) else X_samp
    
    selected_features = [col for col in columns_to_scale if col in features_to_use]

    col_mins = X_train[selected_features].min()
    col_maxs = X_train[selected_features].max()

    X_samp[selected_features] = X_samp[selected_features].clip(lower=col_mins, upper=col_maxs, axis=1)

    X_samp.dropna(inplace=True)
    y_samp = pd.DataFrame(y_samp, columns=feature_list_target).dropna()

    X_samp[selected_features] = X_samp[selected_features].round()

    cols_not_to_scale = list(set(features_to_use) - set(columns_to_scale))
    X_samp[cols_not_to_scale] = (X_samp[cols_not_to_scale] >= 0.5).astype(int)


    y_train = y_samp
    X_train_preprocessed = []
    X_train_preprocessed = None
    X_test_preprocessed = None
    
    try:
        # === Preprocessing Step ===
        X_train_preprocessed = preprocessor.fit_transform(X_samp)
        X_test_preprocessed = preprocessor.transform(X_test[features_to_use])

        # === MOVE SUBSEQUENT LOGIC INSIDE THE TRY BLOCK ===
        if voting_estimators:
            if len(voting_estimators) > 0:
                # Create ensemble *after* successful preprocessing
                ensemble = VotingClassifier(estimators=voting_estimators, voting='hard')

                trial_id = train.get_context().get_trial_id()
                trial_directory = os.path.join(log_directory_base, str(trial_id))
                os.makedirs(trial_directory, exist_ok=True)

                # Create dataframes dictionary *after* successful preprocessing
                dataframes = {
                    "y_train": (pd.DataFrame(y_train, columns=feature_list_target), f"y_train_{trial_id}.csv"),
                    "y_test": (pd.DataFrame(y_test, columns=feature_list_target), f"y_test_{trial_id}.csv"),
                    "X_samp": (pd.DataFrame(X_samp, columns=features_to_use), f"X_samp_{trial_id}.csv"),
                    "X_train_preprocessed": (pd.DataFrame(X_train_preprocessed, columns=features_to_use), f"X_train_preprocessed_{trial_id}.csv"),
                    "X_test_preprocessed": (pd.DataFrame(X_test_preprocessed, columns=features_to_use), f"X_test_preprocessed_{trial_id}.csv")
                }

                for _, (df, filename) in dataframes.items():
                    save_dataframe_to_csv(df, trial_directory, filename)

                joblib_file_path_unfitted = os.path.join(trial_directory, f"unfitted_ensemble_{trial_id}.joblib")
                joblib.dump(ensemble, joblib_file_path_unfitted)

                # Fit the model
                # FIX DataConversionWarning: Use .values.ravel() for y_train
                ensemble.fit(X_train_preprocessed, y_train.values.ravel())
                y_pred = ensemble.predict(X_test_preprocessed)

                joblib_file_path_fitted = os.path.join(trial_directory, f"fitted_ensemble_{trial_id}.joblib")
                joblib.dump(ensemble, joblib_file_path_fitted)

                # Calculate and report score
                score = recall_mean(y_test, y_pred)
                scores.append(score) # Assuming scores is defined earlier

                # Log results
                class_report = classification_report(y_test, y_pred, zero_division=0)
                conf_matrix = confusion_matrix(y_test, y_pred)
                log_file_path = os.path.join(trial_directory, f"class_report_{trial_id}.txt")
                with open(log_file_path, "w") as file:
                    file.write(f"Classification Report for Trial {trial_id}:\n{class_report}\n")
                    file.write(f"Confusion Matrix for one of the validation sets:\n{conf_matrix}\n")
                    file.write(f"Recall Mean Score for one of the validation sets: {score}\n")

                # Report the final score for this trial
                average_score = gmean(scores) # Make sure scores list is correctly populated
                train.report({"recall_mean": average_score})

            else: # Handle case where voting_estimators is empty
                print("No models in voting_estimators. Skipping ensemble.")
                train.report({"recall_mean": -float('inf')}) # Report failure if no models
        else: # Handle case where voting_estimators is None/False
            print("voting_estimators is not defined or False. Skipping ensemble.")
            train.report({"recall_mean": -float('inf')}) # Report failure


    except ValueError as e:
        # Handle the specific preprocessing error
        if "Found array with 0 sample(s)" in str(e) or "NaN or Inf found in input tensor" in str(e): # Added NaN check
            print(f"Preprocessing failed for trial {train.get_context().get_trial_id()}: {e}")
            train.report({"recall_mean": -float('inf')})
        # Handle the n_neighbors error during ensemble fitting (moved from inner try)
        elif "Expected n_neighbors <= n_samples_fit" in str(e):
            print(f"KNN n_neighbors error during fitting for trial {train.get_context().get_trial_id()}: {e}")
            train.report({"recall_mean": -float('inf')})
        else:
            # Re-raise other ValueErrors
            print(f"Unhandled ValueError for trial {train.get_context().get_trial_id()}: {e}")
            raise
    # Catch other potential exceptions during fitting/prediction if needed
    except Exception as e:
        print(f"An unexpected error occurred for trial {train.get_context().get_trial_id()}: {e}")
        import traceback
        traceback.print_exc()
        train.report({"recall_mean": -float('inf')}) # Report failure
        
    
search_space = {

    'scaler_choice': tune.choice(['MinMaxScaler', 'RobustScaler', 'StandardScaler', 'passthrough']),
    'normalization_choice': tune.sample_from(lambda config: np.random.choice([ 'passthrough'])),
     
     
     
    'oversampler': tune.choice(['NEATER', 'Lee', 'SMOTE', 'SMOTE_TomekLinks', 'SMOTE_ENN', 'Borderline_SMOTE1', 'distance_SMOTE', 'ADASYN',
                                 'Borderline_SMOTE2', 'LLE_SMOTE', 'SMMO', 'Stefanowski', 'ADOMS',
                                 'Safe_Level_SMOTE', 'MSMOTE', 'DE_oversampling', 'SMOBD', 'MSYN', 'SVM_balance',
                                 'TRIM_SMOTE', 'ProWSyn', 'SL_graph_SMOTE', 'NRSBoundary_SMOTE',
                                 'SOI_CJ', 'ROSE', 'SMOTE_OUT', 'SMOTE_Cosine', 'Selected_SMOTE', 'LN_SMOTE', 'MWMOTE',
                                 'PDFOS', 'IPADE_ID', 'RWO_sampling', 'Gazzah', 'MCT', 'SMOTE_IPF',
                                 'KernelADASYN', 'MOT2LD', 'V_SYNTH', 'OUPS', 'SMOTE_D', 'SMOTE_PSO', 'CURE_SMOTE',
                                 'SOMO', 'CE_SMOTE', 'Edge_Det_SMOTE', 'CBSO', 'DBSMOTE',
                                 'ASMOBD', 'Assembled_SMOTE', 'SDSMOTE', 'G_SMOTE', 'NT_SMOTE', 'SPY',
                                 'SMOTE_PSOBAT', 'MDO', 'Random_SMOTE', 'ISMOTE', 'VIS_RST', 'A_SUWO',
                                 'SMOTE_FRST_2T', 'AND_SMOTE', 'NRAS', 'SSO', 'NDO_sampling',
                                 'Gaussian_SMOTE', 'kmeans_SMOTE', 'Supervised_SMOTE', 'SN_SMOTE', 'CCR', 'ANS',
                                 'cluster_SMOTE', 'SYMPROD', 'SMOTEWB']),

     
    'NEATER_b': tune.sample_from(lambda config: np.random.randint(1, 10) if config.get('oversampler') == 'NEATER' else None),
    'NEATER_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'NEATER' else None),
    'NEATER_smote_n_neighbors': tune.sample_from(lambda config: np.random.randint(1, 40) if config.get('oversampler') == 'NEATER' else None),
    'NEATER_alpha': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'NEATER' else None),
    'NEATER_h': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'NEATER' else None),

    'lee_n_neighbors': tune.sample_from(lambda config: np.random.randint(1, 40) if config.get('oversampler') == 'Lee' else None),
    'Lee_proportion': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'Lee' else None),
    'Lee_rejection_level': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'Lee' else None),


    'SMOTE_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'SMOTE' else None),
    'SMOTE_n_neighbors': tune.sample_from(lambda config: np.random.randint(1, 40) if config.get('oversampler') == 'SMOTE' else None),

    'SMOTE_TomekLinks_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'SMOTE_TomekLinks' else None),
    'SMOTE_TomekLinks_n_neighbors': tune.sample_from(lambda config: np.random.randint(1, 40) if config.get('oversampler') == 'SMOTE_TomekLinks' else None),

    'SMOTE_ENN_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'SMOTE_ENN' else None),
    'SMOTE_ENN_n_neighbors': tune.sample_from(lambda config: np.random.randint(1, 40) if config.get('oversampler') == 'SMOTE_ENN' else None),

    'Borderline_SMOTE1_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'Borderline_SMOTE1' else None),
    'Borderline_SMOTE1_n_neighbors': tune.sample_from(lambda config: np.random.randint(1, 40) if config.get('oversampler') == 'Borderline_SMOTE1' else None),
    'Borderline_SMOTE1_k_neighbors': tune.sample_from(lambda config: np.random.randint(1, 10) if config.get('oversampler') == 'Borderline_SMOTE1' else None),

    'distance_SMOTE_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'distance_SMOTE' else None),
    'distance_SMOTE_n_neighbors': tune.sample_from(lambda config: np.random.randint(1, 40) if config.get('oversampler') == 'distance_SMOTE' else None),

    'ADASYN_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'ADASYN' else None),
    'ADASYN_n_neighbors': tune.sample_from(lambda config: np.random.randint(1, 40) if config.get('oversampler') == 'ADASYN' else None),
    'ADASYN_d_th': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'ADASYN' else None),
    'ADASYN_beta': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'ADASYN' else None),
    

    'Borderline_SMOTE2_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'Borderline_SMOTE2' else None),
    'Borderline_SMOTE2_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'Borderline_SMOTE2' else None),
    'Borderline_SMOTE2_k_neighbors': tune.sample_from(lambda config: np.random.randint(3, 40) if config.get('oversampler') == 'Borderline_SMOTE2' else None),
    
    'LLE_SMOTE_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'LLE_SMOTE' else None),
    'LLE_SMOTE_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'LLE_SMOTE' else None),
    
    
    'SMMO_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'SMMO' else None),
    'SMMO_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'SMMO' else None),
    
    'ADOMS_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'ADOMS' else None),
    'ADOMS_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'ADOMS' else None),
    
    'Safe_Level_SMOTE_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'Safe_Level_SMOTE' else None),
    'Safe_Level_SMOTE_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'Safe_Level_SMOTE' else None),
    
    'MSMOTE_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'MSMOTE' else None),
    'MSMOTE_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'MSMOTE' else None),
    
    'SMOBD_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'SMOBD' else None),
    'SMOBD_eta1': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'SMOBD' else None),
    'SMOBD_t': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'SMOBD' else None),
    'SMOBD_min_samples': tune.sample_from(lambda config: np.random.randint(3, 40) if config.get('oversampler') == 'SMOBD' else None),
    'SMOBD_max_eps': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'SMOBD' else None),
    
    'DE_oversampling_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'DE_oversampling' else None),
    'DE_oversampling_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'DE_oversampling' else None),
    'DE_oversampling_crossover_rate': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'DE_oversampling' else None),
    'DE_oversampling_similarity_threshold': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'DE_oversampling' else None),
    'DE_oversampling_n_clusters': tune.sample_from(lambda config: np.random.randint(3, 400) if config.get('oversampler') == 'DE_oversampling' else None),

    'MSYN_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'MSYN' else None),
    'MSYN_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'MSYN' else None),
    'MSYN_pressure': tune.sample_from(lambda config: np.random.randint(3, 40) if config.get('oversampler') == 'MSYN' else None),
    
    'SVM_balance_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'SVM_balance' else None),
    'SVM_balance_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'SVM_balance' else None),
    
    'TRIM_SMOTE_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'TRIM_SMOTE' else None),
    'TRIM_SMOTE_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'TRIM_SMOTE' else None),
    'TRIM_SMOTE_min_precision': tune.sample_from(lambda config: np.random.uniform(0.1, 0.9) if config.get('oversampler') == 'TRIM_SMOTE' else None),
    
    'SMOTE_RSB_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'SMOTE_RSB' else None),
    'SMOTE_RSB_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'SMOTE_RSB' else None),
    
    'ProWSyn_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'ProWSyn' else None),
    'ProWSyn_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'ProWSyn' else None),
    'ProWSyn_L': tune.sample_from(lambda config: np.random.uniform(1, 10) if config.get('oversampler') == 'ProWSyn' else None),
    'ProWSyn_theta': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'ProWSyn' else None),
    
    'SL_graph_SMOTE_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'SL_graph_SMOTE' else None),
    'SL_graph_SMOTE_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'SL_graph_SMOTE' else None),
    
    'NRSBoundary_SMOTE_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'NRSBoundary_SMOTE' else None),
    'NRSBoundary_SMOTE_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'NRSBoundary_SMOTE' else None),
    
    'LVQ_SMOTE_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'LVQ_SMOTE' else None),
    'LVQ_SMOTE_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'LVQ_SMOTE' else None),
    'LVQ_SMOTE_n_clusters': tune.sample_from(lambda config: np.random.randint(3, 400) if config.get('oversampler') == 'LVQ_SMOTE' else None),
    
    'SOI_CJ_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'SOI_CJ' else None),
    'SOI_CJ_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'SOI_CJ' else None),
    
    'ROSE_SMOTE_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'ROSE' else None),
    
    'SMOTE_OUT_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'SMOTE_OUT' else None),
    'SMOTE_OUT_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'SMOTE_OUT' else None),
    
    'SMOTE_Cosine_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'SMOTE_Cosine' else None),
    'SMOTE_Cosine_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'SMOTE_Cosine' else None),

    
    'MWMOTE_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'MWMOTE' else None),
    'MWMOTE_k1': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'MWMOTE' else None),
    'MWMOTE_k2': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'MWMOTE' else None),
    'MWMOTE_k3': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'MWMOTE' else None),
    'MWMOTE_M': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'MWMOTE' else None),
    'MWMOTE_cf_th': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'MWMOTE' else None),
    'MWMOTE_cmax': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'MWMOTE' else None),
    
    'Selected_SMOTE_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'Selected_SMOTE' else None),
    'Selected_SMOTE_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'Selected_SMOTE' else None),
    'Selected_SMOTE_perc_sign_attr': tune.sample_from(lambda config: np.random.uniform(0.1, 0.9) if config.get('oversampler') == 'Selected_SMOTE' else None),
    
    'LN_SMOTE_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'LN_SMOTE' else None),
    'LN_SMOTE_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'LN_SMOTE' else None),
    
    'PDFOS_SMOTE_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'PDFOS' else None),
    
    'IPADE_ID_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'IPADE_ID' else None),
    'IPADE_ID_SMOTE_F': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'IPADE_ID' else None),
    'IPADE_ID_SMOTE_G': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'IPADE_ID' else None),
    'IPADE_ID_SMOTE_OT': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'IPADE_ID' else None),
    'IPADE_ID_SMOTE_max_it': tune.sample_from(lambda config: np.random.randint(3, 40) if config.get('oversampler') == 'IPADE_ID' else None),

    'RWO_sampling_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'RWO_sampling' else None),
    
    'DEAGO_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'DEAGO' else None),
    'DEAGO_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'DEAGO' else None),
    'DEAGO_e': tune.sample_from(lambda config: np.random.uniform(1.1, 10) if config.get('oversampler') == 'DEAGO' else None),
    'DEAGO_h': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'DEAGO' else None),
    'DEAGO_sigma': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'DEAGO' else None),
    
    'Gazzah_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'Gazzah' else None),
    'Gazzah_n_components': tune.sample_from(lambda config: np.random.randint(3, 40) if config.get('oversampler') == 'Gazzah' else None),
    
    'ADG_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'ADG' else None),
    'ADG_lam': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'ADG' else None),
    'ADG_mu': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'ADG' else None),
    'ADG_k': tune.sample_from(lambda config: np.random.uniform(1, 10) if config.get('oversampler') == 'ADG' else None),
    'ADG_gamma': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'ADG' else None),
    
    'MCT_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'MCT' else None),
    'MCT_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'MCT' else None),
    
    'SMOTE_IPF_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'SMOTE_IPF' else None),
    'SMOTE_IPF_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'SMOTE_IPF' else None),
    'SMOTE_IPF_n_folds': tune.sample_from(lambda config: np.random.randint(3, 6) if config.get('oversampler') == 'SMOTE_IPF' else None),
    'SMOTE_IPF_k': tune.sample_from(lambda config: np.random.uniform(1, 10) if config.get('oversampler') == 'SMOTE_IPF' else None),
    'SMOTE_IPF_p': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'SMOTE_IPF' else None),
    
    'KernelADASYN_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'KernelADASYN' else None),
    'KernelADASYN_k': tune.sample_from(lambda config: np.random.uniform(1, 10) if config.get('oversampler') == 'KernelADASYN' else None),
    'KernelADASYN_h': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'KernelADASYN' else None),
    
    'SMOTE_PSO_k': tune.sample_from(lambda config: np.random.uniform(1, 10) if config.get('oversampler') == 'SMOTE_PSO' else None),
    'SMOTE_PSO_eps': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'SMOTE_PSO' else None),
    'SMOTE_PSO_n_pop': tune.sample_from(lambda config: np.random.randint(3, 40) if config.get('oversampler') == 'SMOTE_PSO' else None),
    'SMOTE_PSO_w': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'SMOTE_PSO' else None),
    'SMOTE_PSO_c1': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'SMOTE_PSO' else None),
    'SMOTE_PSO_c2': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'SMOTE_PSO' else None),
    'SMOTE_PSO_num_it': tune.sample_from(lambda config: np.random.randint(3, 40) if config.get('oversampler') == 'SMOTE_PSO' else None),

    'SOMO_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'SOMO' else None),
    'SOMO_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'SOMO' else None),
    
    'CURE_SMOTE_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'CURE_SMOTE' else None),
    'CURE_SMOTE_n_clusters': tune.sample_from(lambda config: np.random.randint(3, 400) if config.get('oversampler') == 'CURE_SMOTE' else None),
    'CURE_SMOTE_noise_th': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'CURE_SMOTE' else None),
    
    'MOT2LD_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'MOT2LD' else None),
    'MOT2LD_n_components': tune.sample_from(lambda config: np.random.randint(3, 40) if config.get('oversampler') == 'MOT2LD' else None),
    'MOT2LD_k': tune.sample_from(lambda config: np.random.randint(3, 40) if config.get('oversampler') == 'MOT2LD' else None),
    
    'ISOMAP_Hybrid_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'ISOMAP_Hybrid' else None),
    'ISOMAP_Hybrid_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'ISOMAP_Hybrid' else None),
    
    'CE_SMOTE_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'CE_SMOTE' else None),
    'CE_SMOTE_alpha': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'CE_SMOTE' else None),
    'CE_SMOTE_h': tune.sample_from(lambda config: np.random.uniform(1, 10) if config.get('oversampler') == 'CE_SMOTE' else None),
    'CE_SMOTE_k': tune.sample_from(lambda config: np.random.uniform(1, 10) if config.get('oversampler') == 'CE_SMOTE' else None),
    
    'SMOTE_PSOBAT_maxit': tune.sample_from(lambda config: np.random.randint(3, 40) if config.get('oversampler') == 'SMOTE_PSOBAT' else None),
    'SMOTE_PSOBAT_alpha': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'SMOTE_PSOBAT' else None),
    'SMOTE_PSOBAT_c2': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'SMOTE_PSOBAT' else None),
    'SMOTE_PSOBAT_c3': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'SMOTE_PSOBAT' else None),
    'SMOTE_PSOBAT_gamma': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'SMOTE_PSOBAT' else None),
    
    'V_SYNTH_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'V_SYNTH' else None),
    'V_SYNTH_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'V_SYNTH' else None),
    
    'G_SMOTE_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'G_SMOTE' else None),
    'G_SMOTE_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'G_SMOTE' else None),
    
    'NT_SMOTE_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'NT_SMOTE' else None),
    
    'SMOTE_D_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'SMOTE_D' else None),
    'SMOTE_D_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'SMOTE_D' else None),
    
    'DBSMOTE_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'DBSMOTE' else None),
    'DBSMOTE_eps': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'DBSMOTE' else None),
    'DBSMOTE_min_samples': tune.sample_from(lambda config: np.random.randint(3, 40) if config.get('oversampler') == 'DBSMOTE' else None),
    'DBSMOTE_db_iter_limit': tune.sample_from(lambda config: np.random.randint(3, 101) if config.get('oversampler') == 'DBSMOTE' else None),
    
    'Edge_Det_SMOTE_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'Edge_Det_SMOTE' else None),
    'Edge_Det_SMOTE_k': tune.sample_from(lambda config: np.random.randint(3, 40) if config.get('oversampler') == 'Edge_Det_SMOTE' else None),
    
    'E_SMOTE_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'E_SMOTE' else None),
    'E_SMOTE_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'E_SMOTE' else None),
    'E_SMOTE_min_features': tune.sample_from(lambda config: np.random.randint(3, 40) if config.get('oversampler') == 'E_SMOTE' else None),
    
    'CBSO_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'CBSO' else None),
    'CBSO_k': tune.sample_from(lambda config: np.random.randint(3, 40) if config.get('oversampler') == 'CBSO' else None),
    'CBSO_C_p': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'CBSO' else None),
  
    'Assembled_SMOTE_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'Assembled_SMOTE' else None),
    'Assembled_SMOTE_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'Assembled_SMOTE' else None),
    'Assembled_SMOTE_pop': tune.sample_from(lambda config: np.random.uniform(10, 10) if config.get('oversampler') == 'Assembled_SMOTE' else None),
    'Assembled_SMOTE_thres': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'Assembled_SMOTE' else None),
    
    'SDSMOTE_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'SDSMOTE' else None),
    'SDSMOTE_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'SDSMOTE' else None),
    
    'ASMOBD_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'ASMOBD' else None),
    'ASMOBD_min_samples': tune.sample_from(lambda config: np.random.randint(3, 40) if config.get('oversampler') == 'ASMOBD' else None),
    'ASMOBD_eta': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'ASMOBD' else None),
    'ASMOBD_eps': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'ASMOBD' else None),
    
    'SPY_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'SPY' else None),
    
    'MDO_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'MDO' else None),
    'MDO_k2': tune.sample_from(lambda config: np.random.randint(3, 40) if config.get('oversampler') == 'MDO' else None),
    'MDO_K1_frac': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'MDO' else None),
    
    'ISMOTE_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'ISMOTE' else None),
    'ISMOTE_minority_weight': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'ISMOTE' else None),
    
    'Random_SMOTE_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'Random_SMOTE' else None),
    'Random_SMOTE_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'Random_SMOTE' else None),
    
    'VIS_RST_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'VIS_RST' else None),
    'VIS_RST_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'VIS_RST' else None),
    
    'A_SUWO_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'A_SUWO' else None),
    'A_SUWO_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'A_SUWO' else None),
    'A_SUWO_n_clus_maj': tune.sample_from(lambda config: np.random.randint(3, 40) if config.get('oversampler') == 'A_SUWO' else None),
    'A_SUWO_c_thres': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'A_SUWO' else None),
    
    'SMOTE_FRST_2T_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'SMOTE_FRST_2T' else None),
    'SMOTE_FRST_2T_gamma_M': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'SMOTE_FRST_2T' else None),
    'SMOTE_FRST_2T_gamma_S': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'SMOTE_FRST_2T' else None),
    
    'AND_SMOTE_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'AND_SMOTE' else None),
    'AND_SMOTE_k': tune.sample_from(lambda config: np.random.randint(3, 40) if config.get('oversampler') == 'AND_SMOTE' else None),
    
    'NRAS_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'NRAS' else None),
    'NRAS_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'NRAS' else None),
    'NRAS_t': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'NRAS' else None),
    
    'AMSCO_n_pop': tune.sample_from(lambda config: np.random.randint(3, 40) if config.get('oversampler') == 'AMSCO' else None),
    'AMSCO_n_iter': tune.sample_from(lambda config: np.random.randint(3, 400) if config.get('oversampler') == 'AMSCO' else None),
    'AMSCO_omega': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'AMSCO' else None),
    'AMSCO_r1': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'AMSCO' else None),
    'AMSCO_r2': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'AMSCO' else None),
    
    'NDO_sampling_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'NDO_sampling' else None),
    'NDO_sampling_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'NDO_sampling' else None),
    'NDO_sampling_T': tune.sample_from(lambda config: np.random.randint(3, 40) if config.get('oversampler') == 'NDO_sampling' else None),
    
    'SSO_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'SSO' else None),
    'SSO_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'SSO' else None),
    'SSO_h': tune.sample_from(lambda config: np.random.uniform(1, 10) if config.get('oversampler') == 'SSO' else None),
    'SSO_n_iter': tune.sample_from(lambda config: np.random.randint(3, 101) if config.get('oversampler') == 'SSO' else None),
    'Gaussian_SMOTE_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'Gaussian_SMOTE' else None),
    'Gaussian_SMOTE_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'Gaussian_SMOTE' else None),
    'Gaussian_SMOTE_sigma': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'Gaussian_SMOTE' else None),
    
    'kmeans_SMOTE_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'kmeans_SMOTE' else None),
    'kmeans_SMOTE_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'kmeans_SMOTE' else None),
    'kmeans_SMOTE_n_clusters': tune.sample_from(lambda config: np.random.randint(3, 400) if config.get('oversampler') == 'kmeans_SMOTE' else None),
    'kmeans_SMOTE_irt': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'kmeans_SMOTE' else None),
    
    'SN_SMOTE_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'SN_SMOTE' else None),
    'SN_SMOTE_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'SN_SMOTE' else None),
    
    'Supervised_SMOTE_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'Supervised_SMOTE' else None),
    'Supervised_SMOTE_th_lower': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'Supervised_SMOTE' else None),
    'Supervised_SMOTE_th_upper': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'Supervised_SMOTE' else None),
    
    'CCR_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'CCR' else None),
    'CCR_energy': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'CCR' else None),
    'CCR_scaling': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'CCR' else None),
    
    'ANS_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'ANS' else None),
    
    'cluster_SMOTE_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'cluster_SMOTE' else None),
    'cluster_SMOTE_n_neighbors': tune.sample_from(lambda config: np.random.randint(3, 300) if config.get('oversampler') == 'cluster_SMOTE' else None),
    'cluster_SMOTE_n_clusters': tune.sample_from(lambda config: np.random.randint(3, 400) if config.get('oversampler') == 'cluster_SMOTE' else None),
    
    'SYMPROD_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'SYMPROD' else None),
    'SYMPROD_std_outliers': tune.sample_from(lambda config: np.random.uniform(1.0, 10) if config.get('oversampler') == 'SYMPROD' else None),
    'SYMPROD_k_neighbors': tune.sample_from(lambda config: np.random.randint(3, 40) if config.get('oversampler') == 'SYMPROD' else None),
    'SYMPROD_m_neighbors': tune.sample_from(lambda config: np.random.randint(3, 40) if config.get('oversampler') == 'SYMPROD' else None),
    'SYMPROD_cutoff_threshold': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('oversampler') == 'SYMPROD' else None),
    
    'SMOTEWB_proportion': tune.sample_from(lambda config: np.random.uniform(0.8, 3) if config.get('oversampler') == 'SMOTEWB' else None),
    'SMOTEWB_max_depth': tune.sample_from(lambda config: np.random.randint(3, 40) if config.get('oversampler') == 'SMOTEWB' else None),
    'SMOTEWB_n_iters': tune.sample_from(lambda config: np.random.randint(3, 400) if config.get('oversampler') == 'SMOTEWB' else None),
  
    'feature_selection_choice': tune.choice(['selector_1' ]),

    'age': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),
    'weight': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),
    'Height': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),
    'Durationinfertility': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),
    'FSH': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),
    'LH': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),
    'numberofcunsumptiondrug': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),
    'durationofstimulation': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),
    'numberRfulicule': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),
    'numberLfulicule': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),
    'numberofoocyte': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),
    'metaghaze1oocyte': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),
    'metaghaze2oocyte': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),
    'necrozeoocyte': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),
    'lowQualityoocyte': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),
    'Gvoocyte': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),
    'macrooocyte': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),
    'partogenesoocyte': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),
    'numberembrio': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),
    'countspermogram': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),
    'motilityspermogram': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),
    'morfologyspermogram': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),
    'gradeembrio': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),
    'Typecycle': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),
    'reasoninfertility': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),
    'Typeofcunsumptiondrug': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),
    'typeoftrigger': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),
    'Typedrug': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),
    'pregnancy': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),
    'mense': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),
    'Infertility': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('feature_selection_choice') == 'selector_1' else None),

    'use_svm': tune.choice([False, True]),
    'use_ridge': tune.choice([False, True]),
    'use_logistic': tune.choice([False, True]),
    'use_rf': tune.choice([False, True]),
    'use_xgboost': tune.choice([False, True]),
    'use_SGD': tune.choice([False, True]),
    'use_lightgbm': tune.choice([False]),
    'use_mlp': tune.choice([False]),
    'use_knn': tune.choice([False, True]),
            
    'ridge_alpha': tune.sample_from(lambda config: np.random.uniform(1, 200) if config.get('use_ridge') else None),
    'ridge_tol': tune.sample_from(lambda config: np.random.uniform(1e-5, 1) if config.get('use_ridge') else None),
    'ridge_fit_intercept': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('use_ridge') else None),
    
    'svm_C': tune.sample_from(lambda config: np.random.uniform(1e-3, 10) if config.get('use_svm') else None),
    'svm_kernel': tune.sample_from(lambda config: np.random.choice(['linear', 'poly', 'rbf', 'sigmoid']) if config.get('use_svm') else None),
    'svm_degree': tune.sample_from(lambda config: np.random.randint(0, 10) if config.get('use_svm') else None),
    'svm_gamma': tune.sample_from(lambda config: np.random.choice(['scale', 'auto']) if config.get('use_svm') else None),
    'svm_coef0': tune.sample_from(lambda config: np.random.uniform(-2, 2) if config.get('use_svm') else None),
    'svm_shrinking': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('use_svm') else None),
    'svm_tol': tune.sample_from(lambda config: np.random.uniform(1e-9, 1e-1) if config.get('use_svm') else None),
    'svm_decision_function_shape': tune.sample_from(lambda config: np.random.choice(['ovo', 'ovr']) if config.get('use_svm') else None),

    'logistic_C': tune.sample_from(lambda config: np.random.uniform(0.01, 50) if config.get('use_logistic') else None),
    'logistic_tol': tune.sample_from(lambda config: np.random.uniform(1e-9, 1e-1) if config.get('use_logistic') else None),
    'logistic_solver': tune.sample_from(lambda config: 
        np.random.choice(['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']) if config.get('use_logistic') else None),
    'logistic_penalty': tune.sample_from(lambda config: 
        np.random.choice(['l1', 'l2']) if config.get('logistic_solver') == 'liblinear' else
        np.random.choice(['l2', None]) if config.get('logistic_solver') in ['lbfgs', 'newton-cg', 'sag'] else
        np.random.choice(['l1', 'l2', 'elasticnet', None]) if config.get('logistic_solver') == 'saga' else
        None),
    'logistic_dual': tune.sample_from(lambda config: 
        np.random.choice([True, False]) if config.get('use_logistic') and config.get('logistic_solver') == 'liblinear' and config.get('logistic_penalty') in ['l2'] else False),
    'logistic_fit_intercept': tune.sample_from(lambda config: 
        np.random.choice([True, False]) if config.get('use_logistic') else True),
    'logistic_intercept_scaling': tune.sample_from(lambda config: 
        np.random.uniform(0.05, 25) if config.get('use_logistic') else 1.0),
    'logistic_l1_ratio': tune.sample_from(lambda config: 
        np.random.uniform(0, 1) if config.get('use_logistic') and config.get('logistic_penalty') == 'elasticnet' else None),


    
    'rfc_max_depth': tune.sample_from(lambda config: np.random.randint(1, 300) if config.get('use_rf') else None),
    'rf_n_estimators': tune.sample_from(lambda config: np.random.randint(2, 2501) if config.get('use_rf') else None),
    'rf_min_samples_split': tune.sample_from(lambda config: np.random.randint(2, 151) if config.get('use_rf') else None),
    'rf_min_samples_leaf': tune.sample_from(lambda config: np.random.randint(1, 151) if config.get('use_rf') else None),
    'rf_criterion': tune.sample_from(lambda config: np.random.choice(["gini", "entropy"]) if config.get('use_rf') else None),
    'rf_min_weight_fraction_leaf': tune.sample_from(lambda config: np.random.uniform(0.0, 0.5) if config.get('use_rf') else None),
    'rf_max_leaf_nodes': tune.sample_from(lambda config: np.random.randint(2, 151) if config.get('use_rf') else None),
    'rf_min_impurity_decrease': tune.sample_from(lambda config: np.random.uniform(0.0, 0.5) if config.get('use_rf') else None),
    'rf_bootstrap': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('use_rf') else None),
    'rf_class_weight': tune.sample_from(lambda config: np.random.choice(["balanced", "balanced_subsample", None]) if config.get('use_rf') else None),
    'rf_ccp_alpha': tune.sample_from(lambda config: np.random.uniform(0.0, 0.036) if config.get('use_rf') else None),

    'SGD_loss': tune.sample_from(lambda config: np.random.choice([ 'log_loss', 'modified_huber']) if config.get('use_SGD') else None),
    'SGD_penalty': tune.sample_from(lambda config: np.random.choice(['l2', 'l1', 'elasticnet', None]) if config.get('use_SGD') else None),
    'SGD_alpha': tune.sample_from(lambda config: np.random.uniform(1e-3, 10.0) if config.get('use_SGD') else None),
    'SGD_l1_ratio': tune.sample_from(lambda config: np.random.uniform(0, 1) if config.get('use_SGD') else None),
    'SGD_fit_intercept': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('use_SGD') else None),
    'SGD_tol': tune.sample_from(lambda config: np.random.uniform(1e-4, 1) if config.get('use_SGD') else None),
    'SGD_epsilon': tune.sample_from(lambda config: np.random.uniform(0, 3) if config.get('use_SGD') else None),
    'SGD_learning_rate': tune.sample_from(lambda config: np.random.choice(['constant', 'optimal', 'invscaling', 'adaptive']) if config.get('use_SGD') else None),
    'SGD_eta0': tune.sample_from(lambda config: np.random.uniform(0, 10) if config.get('use_SGD') else None),
    'SGD_power_t': tune.sample_from(lambda config: np.random.uniform(-10, 10) if config.get('use_SGD') else None),
    'SGD_validation_fraction': tune.sample_from(lambda config: np.random.uniform(0, 1) if config.get('use_SGD') else None),
    
    'knn_n_neighbors': tune.sample_from(lambda config: np.random.randint(10, 300) if config.get('use_knn') else None),
    'knn_weights': tune.sample_from(lambda config: np.random.choice(['uniform', 'distance']) if config.get('use_knn') else None),
    'knn_algorithm': tune.sample_from(lambda config: np.random.choice(['ball_tree', 'kd_tree', 'brute']) if config.get('use_knn') else None),
    'knn_leaf_size': tune.sample_from(lambda config: np.random.randint(10, 300) if config.get('use_knn') else None),
    'knn_p': tune.sample_from(lambda config: np.random.randint(1, 21) if config.get('knn_metric') == 'minkowski' and config.get('use_knn') else None),
    'knn_metric': tune.sample_from(lambda config: str(np.random.choice(['minkowski', 'cityblock', 'euclidean', 'l1', 'l2', 'manhattan']))  if config.get('use_knn') else None),

    'xgb_n_estimators': tune.sample_from(lambda config: np.random.randint(20, 1300) if config.get('use_xgboost') else None),
    'xgb_max_depth': tune.sample_from(lambda config: np.random.randint(1, 500) if config.get('use_xgboost') else None),
    'xgb_learning_rate': tune.sample_from(lambda config: np.random.uniform(0.01, 1.0) if config.get('use_xgboost') else None),
    'xgb_grow_policy': tune.sample_from(lambda config: np.random.choice(['depthwise', 'lossguide']) if config.get('use_xgboost') else None),
    'xgb_gamma': tune.sample_from(lambda config: np.random.uniform(0, 5) if config.get('use_xgboost') else None),
    'xgb_min_child_weight': tune.sample_from(lambda config: np.random.uniform(0, 10) if config.get('use_xgboost') else None),
    'xgb_max_delta_step': tune.sample_from(lambda config: np.random.uniform(0, 10) if config.get('use_xgboost') else None),
    'xgb_subsample': tune.sample_from(lambda config: np.random.uniform(0.5, 1) if config.get('use_xgboost') else None),
    'xgb_colsample_bytree': tune.sample_from(lambda config: np.random.uniform(0.5, 1) if config.get('use_xgboost') else None),
    'xgb_colsample_bylevel': tune.sample_from(lambda config: np.random.uniform(0.5, 1) if config.get('use_xgboost') else None),
    'xgb_colsample_bynode': tune.sample_from(lambda config: np.random.uniform(0.5, 1) if config.get('use_xgboost') else None),
    'xgb_lambda': tune.sample_from(lambda config: np.random.uniform(0, 10) if config.get('use_xgboost') else None),
    'xgb_alpha': tune.sample_from(lambda config: np.random.uniform(0, 10) if config.get('use_xgboost') else None),
    'xgb_booster': tune.sample_from(lambda config: np.random.choice(['gbtree', 'dart']) if config.get('use_xgboost') else None),
    'xgb_scale_pos_weight': tune.sample_from(lambda config: np.random.uniform(0.1, 10) if config.get('use_xgboost') else None),
    'xgb_tree_method': tune.sample_from(lambda config: np.random.choice(['auto', 'approx', 'hist']) if config.get('use_xgboost') else None),
    'xgb_sample_type': tune.sample_from(lambda config: np.random.choice(['uniform', 'weighted']) if config.get('use_xgboost') and config['xgb_booster'] == 'dart' else None),
    'xgb_normalize_type': tune.sample_from(lambda config: np.random.choice(['tree', 'forest']) if config.get('use_xgboost') and config['xgb_booster'] == 'dart' else None),
    'xgb_rate_drop': tune.sample_from(lambda config: np.random.uniform(0, 1) if config.get('use_xgboost') and config['xgb_booster'] == 'dart' else None),
    'xgb_one_drop': tune.sample_from(lambda config: np.random.choice([True, False]) if config.get('use_xgboost') and config['xgb_booster'] == 'dart' else None),
    'xgb_skip_drop': tune.sample_from(lambda config: np.random.uniform(0, 1) if config.get('use_xgboost') and config['xgb_booster'] == 'dart' else None),

    
    'lgbm_boosting_type': tune.sample_from(lambda config: np.random.choice(['gbdt']) if config.get('use_lightgbm') else None),
    'lgbm_num_leaves': tune.sample_from(lambda config: np.random.randint(10, 150) if config.get('use_lightgbm') else None),
    'lgbm_max_depth': tune.sample_from(lambda config: np.random.randint(1, 20) if config.get('use_lightgbm') else None),
    'lgbm_learning_rate': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('use_lightgbm') else None),
    'lgbm_n_estimators': tune.sample_from(lambda config: np.random.randint(10, 50) if config.get('use_lightgbm') else None),
    'lgbm_data_sample_strategy':tune.sample_from(lambda config: np.random.choice(['bagging', 'goss']) if config.get('use_lightgbm') else None),
    'lgbm_subsample': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('use_lightgbm') else None),
    'lgbm_colsample_bytree': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('use_lightgbm') else None),
    'lgbm_reg_alpha': tune.sample_from(lambda config: np.random.uniform(0, 5) if config.get('use_lightgbm') else None),
    'lgbm_reg_lambda': tune.sample_from(lambda config: np.random.uniform(0, 5) if config.get('use_lightgbm') else None),
    'lgbm_bagging_freq': tune.sample_from(lambda config: np.random.randint(0, 10) if config.get('use_lightgbm') and config.get('lgbm_boosting_type') != 'goss' else None),
    'lgbm_bagging_fraction': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('use_lightgbm') and config.get('lgbm_boosting_type') != 'goss' else None),
    'lgbm_feature_fraction': tune.sample_from(lambda config: np.random.uniform(0.1, 1.0) if config.get('use_lightgbm') else None),
    'lgbm_min_data_in_leaf': tune.sample_from(lambda config: np.random.randint(1, 50) if config.get('use_lightgbm') else None),


    'mlp_hidden_layer_sizes': tune.sample_from(lambda config: tuple(np.random.randint(5, 500, size=np.random.randint(1, 5))) if config.get('use_mlp') else None),
    'mlp_activation': tune.sample_from(lambda config: np.random.choice(['identity', 'logistic', 'tanh', 'relu']) if config.get('use_mlp') else None),
    'mlp_solver': tune.sample_from(lambda config: np.random.choice(['lbfgs', 'sgd', 'adam']) if config.get('use_mlp') else None),
    'mlp_alpha': tune.sample_from(lambda config: np.random.uniform(1e-7, 1) if config.get('use_mlp') else None),
    'mlp_learning_rate': tune.sample_from(lambda config: np.random.choice(['constant', 'invscaling', 'adaptive']) if config.get('use_mlp') else None),
    'mlp_tol': tune.sample_from(lambda config: np.random.uniform(1e-7, 1e-1) if config.get('use_mlp') else None),
}




analysis = tune.run(
    train_model,
    config=search_space,
    metric="recall_mean",
    mode="max",
    reuse_actors=True,
    search_alg=OptunaSearch(),
    scheduler=ASHAScheduler(),
    num_samples=15000,

)

