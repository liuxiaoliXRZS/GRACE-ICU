import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, average_precision_score
from sklearn.utils import resample
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.svm import SVC
import csv
import ast
# import train_models as tmm
from sklearn import preprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pyroc
import os
import warnings
warnings.filterwarnings('ignore',
                        '.*',
                        UserWarning,
                        'warnings_filtering')
import re
import shap
from tableone import TableOne
from sklearn.utils import resample
import pickle
warnings.filterwarnings("ignore") # stop showing abnormal

import general_utils as gu
import train_tuning_models as ttm
import json

project_path = '.../gemini_icu/' # change to the project path



#             Part 1. Compare the different hyperparameters [GEMINI-ICU]               #

# ------------------------------------------------------------------------------------ #
# 1. get the bootstrap results of all parameters setting of bert models
data_path = project_path + 'result/note/'
gu.all_metric_95ci(data_path, 500, data_path)

# 2. plot the comparison results of bert models
data_path = project_path + 'result/note/'
data_raw = pd.read_csv(data_path + 'all_metric_95ci.csv')
gu.all_model_seq_len_lr_compare(data_raw, data_path)

# 3. get the selected model and other models/scores all metrics with 95%CI
data_path = project_path + 'result/note_num/nomogram/'
gu.all_selected_models_95ci(data_path, 500, data_path)

# 4. calculate nomogram
data_path = project_path + 'result/note_num/nomogram/'
lr_data, ls_or = pd.DataFrame(), []
lr_data = pd.read_csv(data_path + 'multivariable_info_noround.csv')
lr_data.rename(columns={'Unnamed: 0': 'feature_name'}, inplace=True)
lr_data = lr_data.loc[lr_data['feature_name'] == 'preICU_risk_score'].append(lr_data.loc[lr_data['feature_name'] != 'preICU_risk_score'])
lr_data.reset_index(drop=True, inplace=True)

ls_or = round(lr_data['OR'],4).to_list()
ls_unit_score_use, ls_score, total_score=gu.nomogram_score(
    ls_right_value=[10,1.8,1,1,1,24,20,45,31,3,6],
    ls_left_value=[0,0.2,0,0,0,0,2,5,40,15,24],
    ls_or=ls_or,
    ls_xvar=[2, 0.74, 0, 0, 0, 3.4, 9, 16.2, 36.5, 14, 14] # sample test
    # [preICU_risk_score, shock_index, code_status1, activity_bed1, vent1, lactate_max, cci_score, resp_rate_mean, temperature_mean, gcs_min, braden_score]
    # 30319719 [0, 0.63, 0, 1, 1, 5.1, 10, 20.6, 36.6, 15, 10]
    # 31261676 [3, 0.79, 0, 0, 0, 0, 8, 19, 36.7, 15, 16]
    # 31673161 [1, 0.81, 0, 1, 1, 1.9, 4, 20.8, 36.6, 3, 13]
    # 30209929 [2, 0.74, 0, 0, 0, 3.4, 9, 16.2, 36.5, 14, 14]
)
lr_data['feature_value'] = ls_unit_score_use
lr_data[['feature_name', 'feature_value']].to_csv(data_path + 'nomogram_function.csv',index=False)

print(lr_data['feature_name'].tolist())
print(ls_unit_score_use)
print(total_score)

# 5. get the split notes AUROC with 95%CI
data_path = project_path + 'result/subtext/roc_result/'
result_path = project_path + 'result/subtext/'
gu.split_notes_lr_models_95ci(data_path, 500, result_path)



#                           Part 2. ELEDEICU evaludation                           #

# -------------------------------------------------------------------------------- #
# 1. remove outlier and process to the same columns
data_path = project_path + 'data/'
data_save_path = project_path + 'data/use/seq_240/eldericu/'
if not os.path.exists(data_save_path):
    os.makedirs(data_save_path)
outlier_range_check = pd.read_csv(data_path + 'outlier_range_check.csv')
for i in ['mimiciii', 'mimiciv', 'eicu']:
    data_initial, data_no_outlier = pd.DataFrame(), pd.DataFrame()
    data_initial = pd.read_csv(data_path + 'db_generation/' + 'numeric_' + i + '_eldericu_older.csv')
    columns_list = ['icustay_id', 'stay_id', 'patientunitstayid']
    new_names = ['id', 'id', 'id']
    columns_to_rename = {old: new for old, new in zip(columns_list, new_names) if old in data_initial.columns}
    data_initial.rename(columns=columns_to_rename, inplace=True)
    data_no_outlier = gu.data_process_eldericu(data_initial, i, outlier_range_check, data_save_path)
    data_no_outlier.to_csv(data_save_path + i + '_no_outlier.csv', index=False)

# 2. imputation using the miceforest method
data_path = project_path + 'data/use/seq_240/eldericu/'
data_id_path = project_path + 'result/note/clinical_longformer/seq_240/max_len-512_bs-16_epoch-2_lr-1e-05/'
data_save_path = data_path
# upload the needed initial data
data_mimic, data_eicu, data_split_info = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
data_mimic = pd.concat([pd.read_csv(data_path + 'mimiciii_no_outlier.csv'), pd.read_csv(data_path + 'mimiciv_no_outlier.csv')], axis=0, ignore_index=True)
data_eicu = pd.read_csv(data_path + 'eicu_no_outlier.csv')
data_split_info = pd.read_csv(data_id_path + 'text_all.csv')
data_all = {}
for i in ['train', 'val', 'test', 'temp', 'ext']:
    id_list = []
    id_list = data_split_info[data_split_info['db_type'] == i]['all_patient_ids'].tolist()
    if i == 'ext':
        data_all[i] = data_eicu[data_eicu['id'].isin(id_list)].reset_index(drop=True)
    else:
        data_all[i] = data_mimic[data_mimic['id'].isin(id_list)].reset_index(drop=True)
gu.generate_data_imputation_miceforest(data_all, data_split_info, data_save_path)

# 3. upload the model and evaluate the risk probability
if not os.path.exists(project_path + 'result/eldericu/'):
    os.makedirs(project_path + 'result/eldericu/')
model_result_info = {}
model_result_info = {
    'model_path_full_info': project_path + 'pre-analysis/xgb_no_cal.dat',
    'result_path': project_path + 'result/eldericu/',
    'ths_use':'False',
    'ths_value':0
}
data = {}
data_path = project_path + 'data/use/seq_240/eldericu/'
data = {
    'test':pd.read_csv(data_path + 'test_imputation.csv'),
    'temp':pd.read_csv(data_path + 'temp_imputation.csv'),
    'ext':pd.read_csv(data_path + 'ext_imputation.csv')
}
gu.older_eldericu_model_eva(data, model_result_info)


#                        Part 3. train xgboost and rf model                        #

# -------------------------------------------------------------------------------- #
# 1. get the processed data (generate for trainning and evaluating the xgb and rf models' data)
#    merge the needed numeric and preICU_risk_score (*10) data
data_num_path = project_path + 'data/use/seq_240/'
data_text_path = project_path + 'result/note/clinical_longformer/seq_240/max_len-512_bs-16_epoch-2_lr-1e-05/'
data_save_path = project_path + 'data/use/seq_240/ml_models/'
if not os.path.exists(data_save_path):
    os.makedirs(data_save_path)
development_num, temporal_num, external_num, text_prob_all, columns_needed = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []
columns_needed = ['id', 'label', 'shock_index', 'code_status', 'activity_bed', 'vent', \
                  'lactate_max', 'cci_score', 'resp_rate_mean', 'temperature_mean', 'gcs_min', 'braden_score'] # will add preICU_risk_score
development_num = pd.read_csv(data_num_path + 'development_num_older.csv')[columns_needed]
temporal_num = pd.read_csv(data_num_path + 'temporal_num_older.csv')[columns_needed]
external_num = pd.read_csv(data_num_path + 'external_num_older.csv')[columns_needed]
text_prob_all = pd.read_csv(data_text_path + 'text_all.csv')[['all_patient_ids', 'probs_1', 'db_type']]
text_prob_all['probs_1'] = (10*text_prob_all['probs_1']).round()
text_prob_all.rename(columns={'probs_1':'preICU_risk_score', 'all_patient_ids':'id'}, inplace = True)		
for i in ['train', 'val', 'test', 'temp', 'ext']:
    data_need = pd.DataFrame()
    if i == 'temp':
        data_need = pd.merge(temporal_num, text_prob_all[text_prob_all['db_type'] == i][['id','preICU_risk_score']], on='id')
    elif i == 'ext':
        data_need = pd.merge(external_num, text_prob_all[text_prob_all['db_type'] == i][['id','preICU_risk_score']], on='id')
    else:
        data_need = pd.merge(development_num, text_prob_all[text_prob_all['db_type'] == i][['id','preICU_risk_score']], on='id')
    data_need[['label', 'code_status', 'activity_bed', 'vent', 'cci_score', 'gcs_min', 'braden_score']] = \
        data_need[['label', 'code_status', 'activity_bed', 'vent', 'cci_score', 'gcs_min', 'braden_score']].astype('int')
    data_need.to_csv(data_save_path + i + '_imputation.csv', index=False)

# 2. find the optimal parameters
data_path = project_path + 'data/use/seq_240/ml_models/'
result_path = project_path + 'result/ml_models/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
para_clf, roc_plot_clf, parameters_clf = ttm.train_xgb_model(
    pd.read_csv(data_path + 'train_imputation.csv').drop(['id', 'label'], axis=1), 
    pd.read_csv(data_path + 'train_imputation.csv')['label'], 
    pd.read_csv(data_path + 'val_imputation.csv').drop(['id', 'label'], axis=1), 
    pd.read_csv(data_path + 'val_imputation.csv')['label'], 
    pd.read_csv(data_path + 'test_imputation.csv').drop(['id', 'label'], axis=1), 
    pd.read_csv(data_path + 'test_imputation.csv')['label'], 
    80, 1)

para_clf, roc_plot_clf, parameters_clf = ttm.train_rf_model(
    pd.read_csv(data_path + 'train_imputation.csv').drop(['id', 'label'], axis=1), 
    pd.read_csv(data_path + 'train_imputation.csv')['label'], 
    pd.read_csv(data_path + 'val_imputation.csv').drop(['id', 'label'], axis=1), 
    pd.read_csv(data_path + 'val_imputation.csv')['label'], 
    pd.read_csv(data_path + 'test_imputation.csv').drop(['id', 'label'], axis=1), 
    pd.read_csv(data_path + 'test_imputation.csv')['label']
    )

# 3. train and evaluate the performance in all datasets
data_path = project_path + 'data/use/seq_240/ml_models/'
result_path = project_path + 'result/ml_models/'
data = {}
data = {
    'train':pd.read_csv(data_path + 'train_imputation.csv'),
    'val':pd.read_csv(data_path + 'val_imputation.csv'),
    'test':pd.read_csv(data_path + 'test_imputation.csv'),
    'temp':pd.read_csv(data_path + 'temp_imputation.csv'),
    'ext':pd.read_csv(data_path + 'ext_imputation.csv')
}
model_xgb_result_info = {}
model_xgb_result_info = {
    'result_path': result_path,
    'no_cal_model_full_info': result_path + 'xgb_no_cal.dat',
    'ths_use': 'False',
    'ths_value': 0
}
model_rf_result_info = {}
model_rf_result_info = {
    'result_path': result_path,
    'no_cal_model_full_info': result_path + 'rf_no_cal.dat',
    'ths_use': 'False',
    'ths_value': 0
}
gu.older_xgb_model(data, model_xgb_result_info)
gu.older_rf_model(data, model_rf_result_info)



#                           Part 4. acquire all metric of models                             #
# 1. GEMINI-ICU, preICU risk score, structured data, saps, sofa, eldericu, xgboost, rf
# 2. GEMINI-ICU, subtext of notes performance (chief_complaint, history_of_present_illness, medications_on_admission, past_medical_history, physical_exam)
# # ------------------------------------------------------------------------------------------ #
# 1. acquire the optimal thresholds for all models
note_num_path = project_path + 'result/note_num/nomogram/'
eldericu_path = project_path + 'result/eldericu/'
xgb_rf_path = project_path + 'result/ml_models/'
result_path = project_path + 'result/plot_need/'
with open(result_path + 'all_models_performance.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['db_type', 'model_name', 'roc_auc', 'sensitivity', 'specificity', 'accuracy', 'F1', 'precision', 'ap', 'brier_score', 'threshold'])
for i in ['test', 'temp', 'ext']:
    data_need = pd.DataFrame()
    data_need = pd.read_csv(note_num_path + 'roc_' + i + '_set.csv')[['id', 'label', 'pred_f_num_text', 'preICU_risk_score_raw', 'pred_f_num', 'saps', 'sofa']]
    data_need.rename(columns={'preICU_risk_score_raw': 'preICU_risk_score'}, inplace=True)
    # # sofa, saps - map to probability
    # # 1 / (1 + exp(- (-3.3890 + 0.2439*(sf.sofa) ))) as sofa_prob
    # # 1 / (1 + exp(- (-7.7631 + 0.0737*(sapsii) + 0.9971*(ln(sapsii + 1))) )) as sapsii_prob
    data_need['sofa_pb'] = 1/(1 + np.exp(-(-3.3890 + 0.2439*data_need['sofa'])))
    data_need['saps_pb'] = 1/(1+np.exp(-(-7.7631 + 0.0737*(data_need['saps']) + 0.9971*(np.log(data_need['saps'] + 1)))))
    data_need = data_need.round({'sofa_pb': 5, 'saps_pb': 5})
    data_need.loc[data_need['sofa_pb'] > 1] = 1
    data_need.loc[data_need['sofa_pb'] < 0] = 0
    data_need.loc[data_need['saps_pb'] > 1] = 1
    data_need.loc[data_need['saps_pb'] < 0] = 0
    data_need.drop(['sofa', 'saps'], axis=1, inplace=True)
    data_need.rename(columns={'sofa_pb': 'sofa', 'saps_pb': 'saps'}, inplace=True)
    data_need = pd.merge(data_need, pd.read_csv(eldericu_path + i + '_eledericu_evaluate.csv')[['id', 'eldericu']], on='id', how='outer')	
    data_need = pd.merge(data_need, pd.read_csv(xgb_rf_path + i + '_xgb_evaluate.csv')[['id', 'xgb']], on='id', how='outer')
    data_need = pd.merge(data_need, pd.read_csv(xgb_rf_path + i + '_rf_evaluate.csv')[['id', 'rf']], on='id', how='outer')
    data_need.to_csv(result_path + i + '_all_models_scores_probability.csv', index=False)
    for j in [x for x in data_need.columns.tolist() if x not in ['id', 'label']]:
        par_model, _ = gu.model_performance_params(data_need['label'], data_need[j], 'False', 0)
        result_each = []
        result_each = [i, j, round(par_model['auc'],3), 
                       round(par_model['sensitivity'],3), round(par_model['specificity'],3), round(par_model['accuracy'],3), 
                       round(par_model['F1'],3), round(par_model['precision'],3), round(par_model['ap'],3), 
                       round(par_model['brier_score'],3), par_model['threshold']]
        result_all = []
        result_all = open(result_path + 'all_models_performance.csv', 'a', newline='')
        writer = csv.writer(result_all)
        writer.writerow(result_each)
        result_all.close()

# 2. acquire the 95% CI for all metric and all models
#   set the paths & threshold and load data
note_num_path = project_path + 'result/note_num/nomogram/'
eldericu_path = project_path + 'result/eldericu/'
xgb_rf_path = project_path + 'result/ml_models/'
result_path = project_path + 'result/plot_need/'
threshold_set = {}
threshold_set = {
    'test':{'eldericu':0.075693674,'pred_f_num':0.140925873738541,'pred_f_num_text':0.0996157867350402,'preICU_risk_score':0.11950221,\
            'rf':0.1153680871074038, 'saps':0.25,'sofa':0.102525,'xgb':0.12},
    'temp':{'eldericu':0.075693674,'pred_f_num':0.140925873738541,'pred_f_num_text':0.0996157867350402,'preICU_risk_score':0.11950221,\
            'rf':0.1153680871074038, 'saps':0.25,'sofa':0.102525,'xgb':0.12},
    'ext':{'eldericu':0.075693674,'pred_f_num':0.104368511795664,'pred_f_num_text':0.16,'preICU_risk_score':0.20117834,\
            'rf':0.2, 'saps':0.35,'sofa':0.15687,'xgb':0.16978285}
}
#    set the save results
with open(result_path + 'all_models_performance_95ci.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['db_type', 'model_name', 'roc_auc', 'sensitivity', 'specificity', 'accuracy', 'F1', 'precision', 'npv', 'ap', 'brier_score'])
stats_all = pd.DataFrame()
#     loop to save all needed data
for i in ['test', 'temp', 'ext']: 
    data_need = pd.DataFrame()
    data_need = pd.read_csv(note_num_path + 'roc_' + i + '_set.csv')[['id', 'label', 'pred_f_num_text', 'preICU_risk_score_raw', 'pred_f_num', 'saps', 'sofa']]
    data_need.rename(columns={'preICU_risk_score_raw': 'preICU_risk_score'}, inplace=True)
    # # sofa, saps - map to probability
    # # 1 / (1 + exp(- (-3.3890 + 0.2439*(sf.sofa) ))) as sofa_prob
    # # 1 / (1 + exp(- (-7.7631 + 0.0737*(sapsii) + 0.9971*(ln(sapsii + 1))) )) as sapsii_prob
    data_need['sofa_pb'] = 1/(1 + np.exp(-(-3.3890 + 0.2439*data_need['sofa'])))
    data_need['saps_pb'] = 1/(1+np.exp(-(-7.7631 + 0.0737*(data_need['saps']) + 0.9971*(np.log(data_need['saps'] + 1)))))
    data_need = data_need.round({'sofa_pb': 5, 'saps_pb': 5})
    data_need.loc[data_need['sofa_pb'] > 1] = 1
    data_need.loc[data_need['sofa_pb'] < 0] = 0
    data_need.loc[data_need['saps_pb'] > 1] = 1
    data_need.loc[data_need['saps_pb'] < 0] = 0
    data_need.drop(['sofa', 'saps'], axis=1, inplace=True)
    data_need.rename(columns={'sofa_pb': 'sofa', 'saps_pb': 'saps'}, inplace=True)
    data_need = pd.merge(data_need, pd.read_csv(eldericu_path + i + '_eledericu_evaluate.csv')[['id', 'eldericu']], on='id', how='outer')	
    data_need = pd.merge(data_need, pd.read_csv(xgb_rf_path + i + '_xgb_evaluate.csv')[['id', 'xgb']], on='id', how='outer')
    data_need = pd.merge(data_need, pd.read_csv(xgb_rf_path + i + '_rf_evaluate.csv')[['id', 'rf']], on='id', how='outer')
    for j in [x for x in data_need.columns.tolist() if x not in ['id', 'label']]:
        stats_each_all, stats_cal = pd.DataFrame(), pd.DataFrame()
        stats_each_all, stats_cal = gu.model_performance_params_bootstrap_95CI(
             data_need[['label', j]].rename(columns={'label': 'true_label', j:'probability'}), threshold_set[i][j], 500)
        # save the 95% CI metrics
        result_each = []
        result_each = [i, j] + stats_cal.values.flatten().tolist()
        result_all = []
        result_all = open(result_path + 'all_models_performance_95ci.csv', 'a', newline='')
        writer = csv.writer(result_all)
        writer.writerow(result_each)
        result_all.close()
        # save all the metrics for 500 bootstraps
        stats_each_all['db_type'] = i
        stats_each_all['model_name'] = j
        stats_all = pd.concat([stats_all, stats_each_all], ignore_index=True)
cols_to_convert = []
cols_to_convert = ['roc_auc', 'sensitivity', 'specificity', 'accuracy', 'F1', 'precision', 'npv', 'ap', 'brier_score']
stats_all[cols_to_convert] = stats_all[cols_to_convert].apply(pd.to_numeric, errors='coerce')
stats_all.to_csv(result_path + 'all_models_performance_bootstraps.csv', index=False)
#    Check if there is a significant difference in AUROC and AUPRC
stats_all = pd.read_csv(project_path + 'result/plot_need/' + 'all_models_performance_bootstraps.csv')
result_path = project_path + 'result/plot_need/'
for i in ['test', 'temp', 'ext']:
    data = pd.DataFrame()
    data = stats_all.loc[stats_all['db_type'] == i].reset_index(drop=True)
    gu.auc_ap_model_score_vs_pvalue(data, i, result_path)



#                              Part 5. Compare the split notes                              #

# ----------------------------------------------------------------------------------------- #
#   set the paths & threshold and load data
data_path = project_path + 'result/subtext/roc_result/'
result_path = project_path + 'result/plot_need/'
#    set the save results
with open(result_path + 'all_subtext_performance_95ci.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['db_type', 'text_name', 'roc_auc', 'sensitivity', 'specificity', 'accuracy', 'F1', 'precision', 'npv', 'ap', 'brier_score'])
#     loop to save all needed data
for i in ['test', 'temp', 'ext']: 
    data_need = pd.DataFrame()
    for m in ['', 'chief_complaint', 'history_of_present_illness', 'medications_on_admission', 'past_medical_history', 'physical_exam']:
        if m == '':
            data_need = pd.read_csv(data_path + 'roc_' + i + '_set.csv')[['id', 'label', 'pred_f_num_text']].rename(columns={'pred_f_num_text': 'all'})
        else:
            data_need = pd.merge(data_need, \
                                 pd.read_csv(data_path + 'roc_' + i + '_' + m + '_set.csv')[['id', 'pred_f_num_text']].rename(columns={'pred_f_num_text': m}), \
                                    on='id', how='outer')

    for j in [x for x in data_need.columns.tolist() if x not in ['id', 'label']]:
        stats_cal = pd.DataFrame()
        par_model, _ = gu.model_performance_params(data_need['label'], data_need[j], 'False', 0)
        _, stats_cal = gu.model_performance_params_bootstrap_95CI(
             data_need[['label', j]].rename(columns={'label': 'true_label', j:'probability'}), par_model['threshold'], 500)
        # save the 95% CI metrics
        result_each = []
        result_each = [i, j] + stats_cal.values.flatten().tolist()
        result_all = []
        result_all = open(result_path + 'all_subtext_performance_95ci.csv', 'a', newline='')
        writer = csv.writer(result_all)
        writer.writerow(result_each)
        result_all.close()



#                         Part 6. subgroup analysis of GEMINI-ICU                           #

# ----------------------------------------------------------------------------------------- #
data_num_path = project_path + 'data/use/seq_240/'
data_path = project_path + 'result/note_num/nomogram/'
result_path = project_path + 'result/plot_need/'
data_num = pd.DataFrame()
data_num = pd.read_csv(data_num_path + 'development_num_older.csv')[['id', 'age', 'sofa', 'cci_score']]
data_num = pd.concat([data_num, pd.read_csv(data_num_path + 'temporal_num_older.csv')[['id', 'age', 'sofa', 'cci_score']]], ignore_index=True)
data_num = pd.concat([data_num, pd.read_csv(data_num_path + 'external_num_older.csv')[['id', 'age', 'sofa', 'cci_score']]], ignore_index=True)
subgroup_list = {}
subgroup_list = {'age_group':[[65,80], [80,200]], \
                 'cci_score_group':[[3,5], [5,21]], \
                 'sofa_group':[[0,2], [2,4], [4,6], [6,8], [8,10], [10,12], [12,24]]
                }
#    set the save results
with open(result_path + 'all_subgroups_performance_95ci.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['db_type', 'subgroup_name', 'value', 'percent', 'roc_auc', 'sensitivity', 'specificity', 'accuracy', 'F1', 'precision', 'npv', 'ap', 'brier_score'])
#     loop to save all needed data
for i in ['test', 'temp', 'ext']: 
    data_need = pd.DataFrame()
    data_need = pd.read_csv(data_path + 'roc_' + i + '_set.csv')[['id', 'label', 'pred_f_num_text']]
    data_need = pd.merge(data_need, data_num, on='id', how='left')
    for j in ['age_group', 'cci_score_group', 'sofa_group']:
        for m in subgroup_list[j]:
            print([i, j, m])
            data_need_each = pd.DataFrame()
            data_need_each = data_need.loc[(data_need[j.replace("_group", "")] >= m[0]) & (data_need[j.replace("_group", "")] < m[1])].reset_index(drop=True)
            stats_cal = pd.DataFrame()
            par_model, _ = gu.model_performance_params(data_need_each['label'], data_need_each['pred_f_num_text'], 'False', 0)
            _, stats_cal = gu.model_performance_params_bootstrap_95CI(
                data_need_each[['label', 'pred_f_num_text']].rename(columns={'label': 'true_label', 'pred_f_num_text':'probability'}), par_model['threshold'], 500)
            # save the 95% CI metrics
            result_each = []
            result_each = [i, j, m, round(100*data_need_each.shape[0]/data_need.shape[0],1)] + stats_cal.values.flatten().tolist()
            result_all = []
            result_all = open(result_path + 'all_subgroups_performance_95ci.csv', 'a', newline='')
            writer = csv.writer(result_all)
            writer.writerow(result_each)
            result_all.close()

data_path = project_path + 'result/plot_need/'
result_path = project_path + 'result/plot_need/'
gu.split_notes_auroc(data_path, result_path)



#                         Part 6. plot cases for explaination                           #

# ------------------------------------------------------------------------------------- #
data_path = project_path + 'data/'
result_path = project_path + 'result/plot_need/'

data_need_1, data_need_2 = pd.DataFrame(), pd.DataFrame()
data_need_1 = pd.read_csv(data_path + 'plot_1_mimiciv_older.csv')
data_need_2 = pd.read_csv(data_path + 'plot_2_mimiciv_older.csv')
id_need = [30319719, 31261676, 31673161, 31547309, 30209929]
for i in id_need:
    data_need_1_each, data_need_2_each = pd.DataFrame(), pd.DataFrame()
    data_need_1_each = data_need_1.loc[data_need_1['stay_id'] ==i].reset_index(drop=True)
    data_need_2_each = data_need_2.loc[data_need_2['stay_id'] ==i].reset_index(drop=True)
    plt.rcParams['figure.dpi'] = 600
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=4,ncols=1,height_ratios=[4,2,1,1])
    # figure1
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(data_need_1_each.loc[data_need_1_each['label'] == 'braden_score','hr'].tolist(),
             data_need_1_each.loc[data_need_1_each['label'] == 'braden_score','value'].tolist(),
             'bo-', linewidth = 1,  label = 'braden_score')
    ax1.plot(data_need_1_each.loc[data_need_1_each['label'] == 'gcs','hr'].tolist(),
             data_need_1_each.loc[data_need_1_each['label'] == 'gcs','value'].tolist(),
             'go-', linewidth = 1,  label = 'gcs')
    ax1.plot(data_need_1_each.loc[data_need_1_each['label'] == 'resp_rate','hr'].tolist(),
             data_need_1_each.loc[data_need_1_each['label'] == 'resp_rate','value'].tolist(),
             'ro-', linewidth = 1,  label = 'resp_rate')
    ax1.plot(data_need_1_each.loc[data_need_1_each['label'] == 'shock_index','hr'].tolist(),
             [x*10 for x in data_need_1_each.loc[data_need_1_each['label'] == 'shock_index', 'value'].tolist()],
             'co-', linewidth = 1,  label = '10*shock_index')
    ax1.plot(data_need_1_each.loc[data_need_1_each['label'] == 'temperature','hr'].tolist(),
             data_need_1_each.loc[data_need_1_each['label'] == 'temperature','value'].tolist(),
             'mo-', linewidth = 1,  label = 'temperature')
    plt.xlim([0, 24])
    ax1.legend(loc='lower right')
    # handles, labels = ax1.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=1)
    # figure 2
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(data_need_1_each.loc[data_need_1_each['label'] == 'lactate','hr'].tolist(),
             data_need_1_each.loc[data_need_1_each['label'] == 'lactate','value'].tolist(),
             'bo-', linewidth = 1,  label = 'lactate')
    ax2.plot(data_need_1_each.loc[data_need_1_each['label'] == 'pao2','hr'].tolist(),
             [x/100 for x in data_need_1_each.loc[data_need_1_each['label'] == 'pao2', 'value'].tolist()],
             'go-', linewidth = 1,  label = 'pao2')
    ax2.plot(data_need_1_each.loc[data_need_1_each['label'] == 'platelet','hr'].tolist(),
             [x/100 for x in data_need_1_each.loc[data_need_1_each['label'] == 'platelet', 'value'].tolist()],
             'co-', linewidth = 1,  label = 'platelet')
    plt.xlim([0, 24])
    ax2.legend(loc='lower right')
    # handles, labels = ax2.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2)
    # figure 3
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(data_need_1_each.loc[data_need_1_each['label'] == 'urine_output','hr'].tolist(),
             data_need_1_each.loc[data_need_1_each['label'] == 'urine_output','value'].tolist(),
             'bo-', linewidth = 1,  label = 'urine')
    plt.xlim([0, 24])
    ax3.legend(loc='lower right')
    # figure 4
    ax4 = fig.add_subplot(gs[3])
    for j in range(0,data_need_2_each.shape[0]):
        ax4.plot([data_need_2_each.loc[j,'starttime'], data_need_2_each.loc[j,'endtime']], [1, 1], color='purple')
    plt.xlim([0, 24])
    plt.tight_layout()
    fig.savefig(result_path + str(i) + '_plot.png')