from lib2to3.pgen2 import driver
from xml.dom.minidom import Element
from tqdm import tqdm
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
from sklearn import preprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pyroc
import math
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
import miceforest as mf
warnings.filterwarnings("ignore") # Intercept exceptions



#                       Part 1. text data process functions                             #

# ------------------------------------------------------------------------------------- #
def not_pre_admission_check(data):
    special_characters = ['hospital course:', 'discharge', 'major surgical or invasive procedure:', 'pertinent results:', 'followup instructions:']
    stop_loc = []
    for cha_name in special_characters:
        stop_loc_each = []
        stop_loc_each = [i.start() for i in re.finditer(cha_name, data.lower())]
        stop_loc = stop_loc + stop_loc_each
    if len(stop_loc) > 0:
        stop_loc = [min(stop_loc)]

    return stop_loc


def element_start_end(data, element_name):
    # data: text | element_name: string
    # element_list: [chief complaint, family history, history of present illness, medications on admission, past medical history, physical exam, social history]
    # special point: 
    #                physical exam & PHYSICAL EXAMINATION ON ADMISSION
    #                history of present illness & history of the present illness
    extract_info = [] # start and end location
    loc_info_start = [(i.start(), i.end()) for i in re.finditer(element_name + ':', data.lower())]
    if (element_name == 'history of present illness') & (len(loc_info_start) == 0):
        loc_info_start = [(i.start(), i.end()) for i in re.finditer('history of the present illness:', data.lower())]
    if (element_name == 'medications on admission') & (len(loc_info_start) == 0):
        loc_info_start = [(i.start(), i.end()) for i in re.finditer('medications upon admission:', data.lower())]

    if (len(loc_info_start) > 0) & (element_name != 'physical exam'):
        loc_info_end = [i.start() + loc_info_start[0][1] for i in re.finditer('\n[A-Za-z ]+:', data[loc_info_start[0][1]:].lower())]
        if len(loc_info_end) == 0:
            loc_info_end = [i.start() + loc_info_start[0][1] for i in re.finditer('\n\n', data[loc_info_start[0][1]:].lower())]
            if len(loc_info_end) == 0:
                loc_info_end = [i.start() + loc_info_start[0][1] for i in re.finditer('\n', data[loc_info_start[0][1]:].lower())]
    elif (len(loc_info_start) > 0) & (element_name == 'physical exam'):
        loc_info_end = [i.start() + loc_info_start[0][1] for i in re.finditer('\n\n[A-Za-z ]+:', data[loc_info_start[0][1]:].lower())]
        if len(loc_info_end) == 0:
            loc_info_end = [i.start() + loc_info_start[0][1] for i in re.finditer('\n\n', data[loc_info_start[0][1]:].lower())]
            if len(loc_info_end) == 0:
                loc_info_end = [i.start() + loc_info_start[0][1] for i in re.finditer('\n', data[loc_info_start[0][1]:].lower())]
        # check whether existing discharge exam (Discharge exam | PHYSICAL EXAMINATION ON DISCHARGE)
        check_info = []
        check_info = [i.start() + loc_info_start[0][0] for i in re.finditer('discharge', data[loc_info_start[0][0]:loc_info_end[0]].lower())]
        if len(check_info) > 0:
            loc_info_end[0] = check_info[0]
    else:
        loc_info_end = []        

    if len(loc_info_start) > 0:
        stop_loc = []
        stop_loc = not_pre_admission_check(data[loc_info_start[0][0]:loc_info_end[0]])
        if len(stop_loc) == 0:
            extract_info = [loc_info_start[0][0], loc_info_end[0]]
        else:
            extract_info = [loc_info_start[0][0], stop_loc[0] + loc_info_start[0][0]]
    else:
        extract_info = []
    
    return extract_info


def get_need_type_text(data):
    """
    data: [id, text]
    data_result: [id, text, text_need] contain the element and text info
    """
    data_result = pd.DataFrame() # save the text info by text
    data_result = data.copy()
    element_list = ['chief complaint', 'history of present illness', 'medications on admission',
                    'past medical history', 'physical exam'] # 'family history', 'social history'
    for i in range(data.shape[0]):
        data_each, data_each_need = '', ''
        data_each = data.loc[i, 'text']
        for element_each in element_list:
            # print(data.loc[i, 'id'], element_each)
            loc_need = []
            loc_need = element_start_end(data_each, element_each)
            if len(loc_need) > 0:
                data_each_need = '\n\n'.join([data_each_need, data_each[loc_need[0]:loc_need[1]]])
            else:
                data_each_need = '\n\n'.join([data_each_need, ''])
        data_result.loc[i, 'text_need'] = data_each_need
    
    return data_result


def get_need_type_text_spec(data):
    """
    data: [id, history_of_present_illness, physical_exam, past_medical_history, medications_on_admission]
    data_result: [id, text_need] contain the element and text info
    """
    data_result = pd.DataFrame()  # save the text info by text
    element_list = ['history_of_present_illness', 'medications_on_admission',
                    'past_medical_history', 'physical_exam']
    data[element_list] = data[element_list].fillna('').astype(str)
    data_result = data[['id', 'death_hosp']].copy()
    for i in range(data.shape[0]):
        data_each, data_each_need = '', ''
        data_each = data.loc[i]
        for element_each in element_list:
            # print(data.loc[i, 'id'], element_each)
            loc_need = []
            loc_need = data_each[element_each] # notice: it is text needed not locations
            if len(loc_need) > 0:
                data_each_need = '\n\n'.join([data_each_need, element_each.replace('_', " ") + ':  ' + loc_need])
            else:
                data_each_need = '\n\n'.join([data_each_need, ''])
        data_result.loc[i, 'text_need'] = data_each_need

    return data_result


def preprocess1(x):
    y = re.sub('\\[(.*?)\\]', '', x)  # remove de-identified brackets
    # y = re.sub('[0-9]+\.', '', y)  # remove 1.2. since the segmenter segments based on this
    y = re.sub('dr\.', 'doctor', y)
    y = re.sub('m\.d\.', 'md', y)
    y = re.sub('admission date:', '', y)
    y = re.sub('discharge date:', '', y)
    y = re.sub('--|__|==', '', y)
    return y


def preprocessing(df_less_n):
    # df_less_n ['id', 'text', 'label']
    df_less_n['text'] = df_less_n['text'].fillna(' ')
    df_less_n['text'] = df_less_n['text'].str.replace('\n', ' ')
    df_less_n['text'] = df_less_n['text'].str.replace('\r', ' ')
    df_less_n['text'] = df_less_n['text'].apply(str.strip)
    df_less_n['text'] = df_less_n['text'].str.lower()

    df_less_n['text'] = df_less_n['text'].apply(lambda x: preprocess1(x))

    return df_less_n


def name_the_same(data):
    for i in range(data.shape[0]):
        data_each = ''
        data_each = data.loc[i, 'text']
        for r in (('history of the present illness', 'history of present illness'),
                  ('medications upon admission', 'medications on admission')):
            data_each = data_each.replace(*r)
        data.loc[i, 'text'] = data_each

    return data


def notes_split_process(data, no_meaning_contents):
    """
    :param data: dataframe [id, label, text]
    :param no_meaning_contents: list of non meaning words or sentences
    :return: data_new_sep: dataframe [id, label, text_name, sub_text]
    :return: data_new_merge: dataframe [id, label, text]
    """
    element_list = ['chief complaint', 'history of present illness', 'medications on admission',
                'past medical history', 'physical exam'] # 'family history', 'social history'

    data_new = pd.DataFrame()
    for j in range(data.shape[0]):
        data_new_each = pd.DataFrame()
        data_each = ' '.join(data.loc[j, 'text'].split())
        data_each_need = []
        for col_name in element_list:
            text_each_loc = []
            text_each_loc = [(col_name, i.start(), i.end()) for i in re.finditer(col_name + ':', data_each.lower())]
            if len(text_each_loc)>0:
                data_each_need.append(text_each_loc[0])
        data_each_need = pd.DataFrame(data_each_need, columns=['col_name', 'start_loc', 'end_loc'])
        if not data_each_need.empty:
            data_each_need['start_loc_use'] = data_each_need['end_loc'] + 1
            data_each_need['end_loc_use'] = data_each_need['start_loc'].shift(-1)
            data_each_need['end_loc_use'] = data_each_need['end_loc_use'].fillna(len(data_each)+1).astype('int64')
            data_each_need['end_loc_use'] = data_each_need['end_loc_use'] - 1 # drop empty place
            for i in range(data_each_need.shape[0]):
                data_new_each.loc[i, ['id', 'label']] = data.loc[j, ['id', 'label']]
                data_new_each.loc[i, 'text_name'] = data_each_need.loc[i, 'col_name']
                data_new_each.loc[i, 'sub_text'] = data_each[
                    data_each_need.loc[i,'start_loc_use']:data_each_need.loc[i,'end_loc_use']]
            data_new = data_new.append(data_new_each)

    # calculate the length of substr, remove nan, <=1, non meaning rows with special characters
    data_new.reset_index(drop=True, inplace=True)
    data_new['sub_text_len'] = data_new['sub_text'].str.len()
    data_new.dropna(subset=['sub_text'], inplace=True)
    data_new = data_new.loc[data_new['sub_text_len'] > 1]
    data_new = data_new[~data_new['sub_text'].isin(no_meaning_contents)].reset_index(drop=True)
    data_new = data_new.astype({'id':'int64', 'label':int})
    data_new.reset_index(drop=True, inplace=True)

    # get the separate and merged data
    data_new_sep, data_new_merge = pd.DataFrame(), pd.DataFrame()
    data_new_sep = data_new.copy()
    data_new_sep.drop(['sub_text_len'], axis=1, inplace=True)
    data_new_merge = data_new.copy()
    data_new_merge['text'] = data_new_merge['text_name'] + ': ' + data_new_merge['sub_text']
    data_new_merge.drop(['text_name', 'sub_text'], axis=1, inplace=True)
    data_new_merge = data_new_merge.groupby(['id', 'label']).apply(lambda group: '\n\n'.join(group['text']))
    data_new_merge = pd.DataFrame(data_new_merge, columns=['text'])
    data_new_merge.reset_index(inplace=True)
    
    return data_new_sep, data_new_merge


# Used to obtain the text type that contains '\n\n ** :'
def get_all_type_info(data):
    """
    data: text type
    """
    loc_info = pd.DataFrame()
    loc_info = pd.DataFrame([(i.start(),i.end()) for i in re.finditer('\n\n', data)], columns=['start_loc','end_loc'])
    loc_info['start_next_loc'] = loc_info['start_loc'].shift(-1)
    loc_info.dropna(inplace=True)
    loc_info = loc_info.astype('int64')
    data_result = []
    for i in range(loc_info.shape[0]):
        data_each = ''
        data_each = data[loc_info.loc[i,'end_loc']:loc_info.loc[i,'start_next_loc']]
        data_each = data_each.replace('\n', '')
        loc_need = []
        loc_need = [m.start() for m in re.finditer(':', data_each)]
        if len(loc_need) > 0:
            data_each = data_each[:min(loc_need)]
            if len(data_result) == 0:
                data_result = [data_each]
            else:
                data_result.insert(len(data_result), data_each)
    
    data_result = list(set(data_result)) # get unique value

    return data_result


def data_element_info(data):
    # firstly, remove nan rows
    data.dropna(subset=['text'], inplace=True)
    # get all patients' all element types info - to check the need info
    patients_info, element_info = [], [] # patient info of owning, element info of counts
    for i in list(data['id'].unique()):
        data_each, element_each = pd.DataFrame(), []
        data_each = data.loc[data['id'] == i]
        for j in data_each['text']:
            if len(element_each) == 0:
                element_each = get_all_type_info(j)
            else:
                element_each = element_each + get_all_type_info(j)
        element_each = list(set(element_each))
        patients_info.append([i, element_each])

    patients_info = pd.DataFrame(patients_info, columns=['id', 'element_type']).explode('element_type').reset_index(drop=True)

    return patients_info


def get_split_text(text, split_len, overlap_len):
    split_text = []
    for w in range(math.ceil(len(text)/(split_len-overlap_len))):
        if w == 0:
            # Put in directly for the first time after splitting
            text_piece = text[:split_len]
            text_piece = ' '.join(text_piece)
        else:
            # Otherwise, move forward by (split length - overlap)
            window = split_len - overlap_len
            text_piece = text[(w*window):(w*window + split_len)]
            text_piece =' '.join(text_piece)
        split_text.append(text_piece)
    
    return split_text


def text_after_split(data, split_len, overlap_len):
    want = pd.DataFrame()
    for i in range(data.shape[0]):
        want_each =pd.DataFrame()
        data_each = []
        data_each = data.text.iloc[i].split()
        data_each = get_split_text(data_each, split_len, overlap_len)
        want_each['text'] = data_each
        want_each['id'] = data.id.iloc[i]
        want_each['label'] = data.label.iloc[i]
        want = want.append(want_each)
    
    want.reset_index(drop=True, inplace=True)

    return want


def check_no_error_note(data):
    data_check = pd.DataFrame()
    data_check = data.copy()
    data_check['error_num'] = 0
    for i in range(data.shape[0]):
        data_each = data.loc[i, 'text']
        error_num = 0
        for j in ['hospital course:', 'discharge', 'major surgical or invasive procedure:', 'pertinent results:', 'followup instructions:']:
            error_num_each = len([i.start() for i in re.finditer(j, data_each.lower())])
            error_num = error_num + error_num_each
        data_check.loc[i, 'error_num'] = error_num

    return data_check



#                                     Part 2. numeric data process function                                   #

#  ---------------------------------------------------------------------------------------------------------- #
# define the outliers' drop function and remove values out of physiological reasonable range
def outlier_value_nan(data, outlier_range_check_new):
    """
    :param data: dataframe - the input data generating from the databases ['id', 'f1', 'f2']
    :param outlier_range_check_new: dataframe - the upper and lower bound of features [f_name, lower bound, upper bound]
    :return data: dataframe - remove the out of range values
    """
    # change columns' name to lowercase (in order to map the outlier_range_check's name)
    columns_name = []
    columns_name = [x.lower() for x in data.columns.values.tolist()]
    data.columns = columns_name
    for i in range(outlier_range_check_new.shape[0]):
        if outlier_range_check_new['index_name'].tolist()[i].lower() in data.columns.values.tolist():
            data.loc[
                (data[outlier_range_check_new['index_name'].tolist()[i].lower()] > outlier_range_check_new.loc[
                    i].upper_bound) |
                (data[outlier_range_check_new['index_name'].tolist()[i].lower()] < outlier_range_check_new.loc[
                    i].lower_bound),
                outlier_range_check_new['index_name'].tolist()[i].lower()] = np.nan  # and : & ; or : |
    return data


# calculate missing ratio
def calculate_missing_ratio(data):
    """
    :param data: dataframe ['id', 'f1', 'f2']
    :return missing_value_df: ['fea_name', 'missing_ratio']
    """
    percent_missing = data.isnull().sum() * 100 / len(data)
    missing_value_df = pd.DataFrame({'column_name': data.columns, 'percent_missing': percent_missing})
    missing_value_df['percent_missing'] = missing_value_df['percent_missing'].round(2)
    return missing_value_df


# table one info
def cal_tableone_info(data, group_name, categorical_all):
    """
    :param data: the import data used to calculate tableone information
    :para group_name: the group name like 'death_hosp'
    :param categorical_all: all categorical names of four databases
    :return overall_table: without considering the group label
    :return groupby_table: considering the group label
    """
    columns, categorical, groupby, nonnormal = [], [], [], []
    columns = data.columns.values.tolist()
    columns.remove('id')  # change to the needing columns' name
    categorical = list(set(data.columns.values.tolist()).intersection(set(categorical_all)))
    nonnormal = [x for x in columns if x != group_name]
    groupby = group_name
    overall_table = TableOne(data, columns, categorical, nonnormal=columns)
    group_table = TableOne(data, columns, categorical, groupby, nonnormal, pval=True)
    return overall_table, group_table


# string to int type
def older_string_int(data):
    """
    :param data: the import data containing character value's columns
    :return data: character value maps to int value
    """
    string_column_name = {'ethnicity':{'asian':0, 'white':3, 'hispanic':2, 'black':1, 'other':4, 'unknown':5},
                          'gender':{'F':0, 'M':1}
                          }

    for i in string_column_name:
        data[i] = data[i].map(string_column_name[i])

    return data


# add flag: indicate the measurement existing or not
def add_flag(data):
    """
    :param data: the import data containing columns needed to add the corresponding columns
    :return data_new: add the flag columns with zero or one value to indicate no-record or having record
    """    
    flag_columns = ['pao2fio2ratio_vent', 'pao2fio2ratio_novent', 'bilirubin_max'
                    , 'albumin_min', 'alp_max', 'alt_max', 'ast_max', 'baseexcess_min', 'fio2_max'
                    , 'lactate_max', 'lymphocytes_max', 'lymphocytes_min', 'neutrophils_min'
                    , 'paco2_max', 'pao2_min']
    data_flag = data[flag_columns]
    data_flag = data_flag.notnull().astype('int')
    flag_columns_new = []
    for i in data_flag.columns.tolist():
        flag_columns_new.append(i + '_flag')
    data_flag.columns = flag_columns_new
    data_new = pd.DataFrame()
    data_new = pd.concat([data, data_flag], axis=1)
    return data_new


# imputation
def imputation_methods(data, features_name, features_name_flag):
    """
    :param data: the import data containing empty values
    :param feature_name:
    :param features_name_flag:
    :return data_new: the imputed dataframe
    """
    # data_part1: median value for missing
    # data_part2: 0 for missing
    data_part1, data_part2 = pd.DataFrame(), pd.DataFrame()
    names_part1 = [] # no flag features name
    names_part1 = list(set(data.columns.tolist()).difference(set(features_name + features_name_flag)))
    data_part1 = data[names_part1]
    data_part1_new = pd.DataFrame()
    names_part1_new = list(set(names_part1).difference(['id']))
    data_part1_median = data_part1[names_part1_new].median()
    data_part1_new = data_part1[names_part1_new].fillna(data_part1_median)
    data_part1_new = pd.concat([data[['id']], data_part1_new], axis=1)
    data_part1_new.columns = ['id'] + names_part1_new


    data_part2 = data[features_name]
    data_part2.fillna(0, inplace=True)
    data_part2_new = pd.DataFrame()
    data_part2_new = pd.concat([data[features_name_flag], data_part2], axis=1)
    if 'fio2_max' in features_name:
        data_part2_new.loc[data_part2_new['fio2_max_flag'] == 0, 'fio2_max'] = 21

    del data_part1, data_part2

    data_imputation = pd.DataFrame()
    data_imputation = pd.concat([data_part1_new, data_part2_new], axis=1)

    del data_part1_new, data_part2_new

    return data_imputation


# merge the data process
def data_process_older_score(data_use, text_use_count, data_use_name, outlier_range_check, result_path):
    """
    :param data_use: initial data without processing
    :param text_use_count: patients with text existing count
    :para data_use_name: the group name like 'death_hosp'
    :param outlier_range_check: the initial name outlier like hearrate, resp_rate
    :param result_path: save path of results
    :return data_use_final: without considering the group label
    """    

    data_use_final = pd.DataFrame() # create to generate
    data_use = pd.merge(data_use, text_use_count.drop(['label'], axis=1), on=['id'])
    del text_use_count

    # [1]. drop outliers - Amplification the expression: add feature_max/min/mean
    outlier_range_check_new = pd.DataFrame()
    outlier_range_check_new = pd.concat([outlier_range_check, outlier_range_check, outlier_range_check, outlier_range_check], axis=0)
    outlier_range_check_new.rename(columns={'Unnamed: 0': 'index_name'}, inplace=True)
    outlier_range_check.rename(columns={'Unnamed: 0': 'index_name'}, inplace=True)
    # according to the statistic features' name to generate the outlier check columns
    add_fea = ['', '_min', '_max', '_mean']
    new_index = []
    for i in add_fea:
        for j in outlier_range_check['index_name'].tolist():
            new_index.append(j + i)
    outlier_range_check_new['index_name'] = new_index
    outlier_range_check_new.reset_index(drop=True, inplace=True)

    data_use = outlier_value_nan(data_use, outlier_range_check_new)

    # [2]. calculate missing ratio
    calculate_missing_ratio(data_use).to_csv(result_path + data_use_name + '_missing.csv', index=False)

    # get the impute data's tableone info to support to check data
    categorical_all = ['activity_bed', 'activity_eva_flag', 'activity_sit', 'activity_stand'
        , 'admission_type', 'agegroup', 'anchor_year_group', 'epinephrine'
        , 'code_status', 'code_status_eva_flag', 'death_hosp', 'delirium_eva_flag'
        , 'delirium_flag', 'dobutamine', 'dopamine', 'electivesurgery', 'ethnicity'
        , 'fall_risk', 'fall_risk_eva_flag', 'first_careunit', 'gender', 'norepinephrine'
        , 'region', 'teachingstatus', 'vent', 'weightgroup', 'heightgroup'
        , 'chief complaint', 'family history', 'history of present illness', 'medications on admission'
        , 'past medical history', 'physical exam', 'social history', 'vasopressor', 'rrt'
        , 'braden_score_cat', 'braden_flag']
    overall_table, group_table = cal_tableone_info(data_use, 'death_hosp', categorical_all)
    overall_table.to_excel(result_path + 'overall_' + data_use_name + '.xlsx')
    group_table.to_excel(result_path + 'group_' + data_use_name + '.xlsx')
    del overall_table, group_table

    # [3]. map string to int
    data_use = older_string_int(data_use)

    # [4] add flag with value
    data_use_final = pd.DataFrame()
    data_use_final = add_flag(data_use)

    # [5] drop features - without considering in the study
    drop_names = ['apache_iv', 'predictediculos_iv', 'predictedhospitallos_iv'
        , 'apache_iva', 'predictediculos_iva', 'predictedhospitallos_iva'
        , 'oasis', 'saps', 'sofa', 'apsiii'
        , 'oasis_prob', 'saps_prob', 'sofa_prob', 'apsiii_prob', 'apache_iv_prob'
        , 'first_careunit', 'los_hospital_day', 'deathtime_icu_hour'
        , 'anchor_year_group', 'hospitaldischargeyear', 'first_careunit'
        , 'teachingstatus', 'region', 'hospitalid'
        , 'agegroup', 'heightgroup'
        , 'los_icu_day', 'weightgroup'
        , 'troponin_max', 'fibrinogen_min', 'bnp_max'
        , 'chief complaint', 'family history', 'history of present illness'
        , 'medications on admission', 'past medical history', 'physical exam'
        , 'social history'] # 'ethnicity', 'electivesurgery' - will drop them latter

    # drop_names = drop_names + fl_lab_name
    data_use_final.drop(drop_names, axis=1, inplace=True, errors='ignore')

    # [6]. impute values (features with flag: will impute zero;
    #                        features without flag: will impute median value)
    # data [id, features_flag_name, features_no_flag_name]
    features_name = ['pao2fio2ratio_vent', 'pao2fio2ratio_novent', 'bilirubin_max'
        , 'albumin_min', 'alp_max', 'alt_max', 'ast_max', 'baseexcess_min', 'fio2_max'
        , 'lactate_max', 'lymphocytes_max', 'lymphocytes_min', 'neutrophils_min'
        , 'paco2_max', 'pao2_min']

    features_name_flag = [s + '_flag' for s in features_name]
    data_use_final = imputation_methods(data_use_final, features_name, features_name_flag)
    data_use_final.drop(['bmi'], axis=1, inplace=True, errors='ignore')

    # [7]. get BMI, shock index, bun_creatinine, egfr, GNRI info
    data_use_final['bmi'] = 10000*data_use_final['weight']/(data_use_final['height']**2)
    data_use_final['bmi'] = data_use_final['bmi'].round(2)

    data_use_final['shock_index'] = (data_use_final['heart_rate_mean']/data_use_final['sbp_mean']).round(2)
    data_use_final['bun_creatinine'] = (data_use_final['bun_max'] / data_use_final['creatinine_max']).round(2)

    # egfr: gender, creatinine_max, age, ethnicity
    egfr = pd.DataFrame()
    egfr = data_use_final[['id', 'gender', 'age', 'ethnicity', 'creatinine_max']]
    egfr['egfr'] = 186*(egfr['creatinine_max'].pow(-1.154))*(egfr['age'].pow(-0.203))
    egfr.loc[(egfr['gender'] == 0) & (egfr['ethnicity'] == 1), 'egfr'] = 0.742*1.210*egfr['egfr']
    egfr.loc[(egfr['gender'] == 0) & (egfr['ethnicity'] != 1), 'egfr'] = 0.742*egfr['egfr']
    egfr['egfr'] = egfr['egfr'].round(2)

    # ideal weight = height (cm) - 100 - ([height(cm) - 150]/4) for men
    # ideal weight = height (cm) - 100 - ([height(cm) - 150]/2.5) for women
    # GNRI = [1.489*albumin(g/L)] + [41.7*(weight/ideal weight)]
    gnri = pd.DataFrame()
    gnri = data_use_final[['id', 'gender', 'albumin_min', 'weight', 'height', 'albumin_min_flag']]
    gnri['gnri_flag'] = gnri['albumin_min_flag']
    
    gnri['ideal_weight'] = gnri['height'] - 100 - ((gnri['height'] - 150)/4)
    gnri.loc[gnri['gender'] == 0, 'ideal_weight'] = gnri['height'] - 100 - ((gnri['height'] - 150)/2.5)

    gnri['gnri'] = 0
    gnri.loc[gnri['albumin_min_flag'] > 0, 'gnri'] = (14.89*gnri['albumin_min']) + (41.7*(gnri['weight']/gnri['ideal_weight']))    
    gnri['gnri'] = gnri['gnri'].round(2)

    nlr = pd.DataFrame() # nlr: Neutrophil-to-Lymphocyte Ratio
    nlr = data_use_final[['id', 'neutrophils_min', 'lymphocytes_min', 'lymphocytes_min_flag']]
    nlr['nlr_flag'] = nlr['lymphocytes_min_flag']
    nlr['nlr'] = 0
    nlr.loc[nlr['lymphocytes_min_flag'] > 0, 'nlr'] = nlr['neutrophils_min']/nlr['lymphocytes_min']
    nlr['nlr'] = nlr['nlr'].round(2)

    # [8]. merge: main part + gnri + nlr
    data_use_final.drop(['ethnicity', 'electivesurgery'], axis=1, inplace=True, errors='ignore') # not be considered in the study
    data_use_final = pd.merge(data_use_final.drop(['height', 'weight', 'albumin_min', 'albumin_min_flag'], axis=1), gnri[['id', 'gnri', 'gnri_flag']], on='id')
    data_use_final = pd.merge(data_use_final, egfr[['id', 'egfr']], on='id')
    data_use_final = pd.merge(data_use_final, nlr[['id', 'nlr', 'nlr_flag']], on='id')
    data_use_final = pd.merge(data_use_final, data_use[['id', 'sofa', 'oasis', 'saps', 'apsiii', 'oasis_prob', 'saps_prob', 'sofa_prob', 'apsiii_prob']], on='id')

    # [9] drop fall_risk and  fall_risk_eva_flag - due to the very low percent records
    data_use_final.drop(['fall_risk', 'fall_risk_eva_flag'], axis=1, inplace=True)

    # [10] category features reset dtype to int
    columns_int_names = ['activity_bed', 'activity_sit', 'activity_stand', \
                        'code_status', 'death_hosp', 'vasopressor', 'gender', 'vent', 'rrt', 'braden_score_cat']
    columns_int_names = columns_int_names + [s for s in data_use_final.columns.values.tolist() if '_flag' in s]
    data_use_final[columns_int_names] = data_use_final[columns_int_names].astype(int)

    return data_use_final


# get the development, temporal, and external set
# since the ids of mimiciii and mimiciv without overlapping, we directly merge them without processing id
def data_development_temporal_external(data_initial_path, data_path, result_path):
    development_numeric, temporal_numeric, external_numeric = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    development_text, temporal_text, external_text = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # add extra information of numeric data
    numeric_mimiciii = pd.read_csv(data_path + 'numeric_mimiciii_older_use.csv')
    numeric_mimiciii_extra = pd.read_csv(data_initial_path + 'numeric_mimiciii_older.csv')[['icustay_id', 'first_careunit', 'ethnicity', 'anchor_year_group']]
    numeric_mimiciii_extra.rename(columns={'icustay_id': 'id'}, inplace=True)
    numeric_mimiciii = pd.merge(numeric_mimiciii, numeric_mimiciii_extra, on = 'id')
    numeric_mimiciv = pd.read_csv(data_path + 'numeric_mimiciv_older_use.csv')
    numeric_mimiciv_extra = pd.read_csv(data_initial_path + 'numeric_mimiciv_older.csv')[['stay_id', 'first_careunit', 'ethnicity', 'anchor_year_group']]
    numeric_mimiciv_extra.rename(columns={'stay_id': 'id'}, inplace=True)
    numeric_mimiciv = pd.merge(numeric_mimiciv, numeric_mimiciv_extra, on = 'id')
    numeric_eicu = pd.read_csv(data_path + 'numeric_eicu_older_use.csv')
    numeric_eicu_extra = pd.read_csv(data_initial_path + 'numeric_eicu_older.csv')[['patientunitstayid', 'first_careunit', 'ethnicity', 'anchor_year_group']]
    numeric_eicu_extra.rename(columns={'patientunitstayid': 'id'}, inplace=True)
    numeric_eicu = pd.merge(numeric_eicu, numeric_eicu_extra, on = 'id')
    del numeric_mimiciii_extra, numeric_mimiciv_extra, numeric_eicu_extra

    # # add vasopressor feature
    # numeric_mimiciii['vasopressor'] = 0
    # numeric_mimiciii.loc[numeric_mimiciii[['norepinephrine', 'epinephrine', 'dobutamine', 'dopamine']].sum(axis=1) > 0, 'vasopressor'] = 1
    # numeric_mimiciv['vasopressor'] = 0
    # numeric_mimiciv.loc[numeric_mimiciv[['norepinephrine', 'epinephrine', 'dobutamine', 'dopamine']].sum(axis=1) > 0, 'vasopressor'] = 1
    
    # load text data
    text_mimiciii = pd.read_csv(data_path + 'text_mimiciii_older_use.csv')
    text_mimiciv = pd.read_csv(data_path + 'text_mimiciv_older_use.csv')
    text_eicu = pd.read_csv(data_path + 'text_eicu_older_use.csv')

    # merge them
    numeric_data, text_data = pd.DataFrame(), pd.DataFrame()
    numeric_data = numeric_mimiciii.append(numeric_mimiciv).reset_index(drop=True)
    numeric_data.drop(['oasis_prob', 'saps_prob', 'sofa_prob', 'apsiii_prob'], axis=1, inplace=True) # no need
    numeric_eicu.drop(['oasis_prob', 'saps_prob', 'sofa_prob', 'apsiii_prob'], axis=1, inplace=True) # no need
    text_data = text_mimiciii.append(text_mimiciv).reset_index(drop=True)
    del numeric_mimiciii, numeric_mimiciv, text_mimiciii, text_mimiciv

    # get the needed data
    development_numeric =  numeric_data.loc[numeric_data['anchor_year_group'] != '2017 - 2019'].reset_index(drop=True)
    temporal_numeric =  numeric_data.loc[numeric_data['anchor_year_group'] == '2017 - 2019'].reset_index(drop=True)
    external_numeric = numeric_eicu.reset_index(drop=True)
    development_text = text_data[text_data['id'].isin(list(development_numeric['id']))].reset_index(drop=True)
    temporal_text = text_data[text_data['id'].isin(list(temporal_numeric['id']))].reset_index(drop=True)
    external_text = text_eicu[text_eicu['id'].isin(list(external_numeric['id']))].reset_index(drop=True)

    print('development numeric shape: ', development_numeric.shape)
    print('temporal numeric shape: ', temporal_numeric.shape)
    print('external numeric shape: ', external_numeric.shape)
    print('development text shape: ', development_text.shape)
    print('temporal text shape: ', temporal_text.shape)
    print('external text shape: ', external_text.shape)
    print('development set nums: ', len(list(development_numeric['id'].unique())))
    print('temporal set nums: ', len(list(temporal_numeric['id'].unique())))    
    print('external set nums: ', len(list(external_numeric['id'].unique())))

    # save to the result_path
    development_numeric.to_csv(result_path + 'development_num_older.csv', index=False)
    development_text.to_csv(result_path + 'development_text_older.csv', index=False)
    temporal_numeric.to_csv(result_path + 'temporal_num_older.csv', index=False)
    temporal_text.to_csv(result_path + 'temporal_text_older.csv', index=False)
    external_numeric.to_csv(result_path + 'external_num_older.csv', index=False)
    external_text.to_csv(result_path + 'external_text_older.csv', index=False)


def subtext_development_temporal_external(data_path, result_path):
    develop_num, temporal_num, external_num = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    develop_num = pd.read_csv(data_path + 'development_num_older.csv')
    temporal_num = pd.read_csv(data_path + 'temporal_num_older.csv')
    external_num = pd.read_csv(data_path + 'external_num_older.csv')
    element_list = ['chief complaint', 'history of present illness', 'medications on admission',
                'past medical history', 'physical exam'] #'family history', 'social history' 
    for tt in element_list:
        subtext_mimiciii, subtext_mimiciv, subtext_all, subtext_ext = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        develop_subtext, temporal_subtext, external_subtext = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        subtext_mimiciii = pd.read_csv(data_path + 'initial/' + 'text_mimiciii_older_use_' + tt.replace(' ', '_') + '.csv')
        subtext_mimiciv = pd.read_csv(data_path + 'initial/' + 'text_mimiciv_older_use_' + tt.replace(' ', '_') + '.csv')
        subtext_all = pd.concat([subtext_mimiciii, subtext_mimiciv], ignore_index=True)
        subtext_ext = pd.read_csv(data_path + 'initial/' + 'text_eicu_older_use_' + tt.replace(' ', '_') + '.csv')
        del subtext_mimiciii, subtext_mimiciv
        develop_subtext = subtext_all[subtext_all['id'].isin(develop_num['id'].tolist())].reset_index(drop=True)
        temporal_subtext = subtext_all[subtext_all['id'].isin(temporal_num['id'].tolist())].reset_index(drop=True)
        external_subtext = subtext_ext[subtext_ext['id'].isin(external_num['id'].tolist())].reset_index(drop=True)
        develop_subtext.to_csv(result_path + 'development_text_older_' + tt.replace(' ', '_') + '.csv', index=False)
        temporal_subtext.to_csv(result_path + 'temporal_text_older_' + tt.replace(' ', '_') + '.csv', index=False)
        external_subtext.to_csv(result_path + 'external_text_older_' + tt.replace(' ', '_') + '.csv', index=False)


def statistics_development_temporal_external(data_initial_path, data_path, result_path):
    data_extra_use_1, data_extra_iii, data_extra_iv, data_extra_eicu_1 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    data_extra_iii = pd.read_csv(data_initial_path + 'numeric_mimiciii_older.csv')[['icustay_id', 'los_icu_day', 'los_hospital_day']]
    data_extra_iv = pd.read_csv(data_initial_path + 'numeric_mimiciv_older.csv')[['stay_id', 'los_icu_day', 'los_hospital_day']]
    data_extra_eicu_1 = pd.read_csv(data_initial_path + 'numeric_eicu_older.csv')[['patientunitstayid', 'los_icu_day', 'los_hospital_day']]

    data_extra_iii.rename(columns={'icustay_id': 'id'}, inplace=True)
    data_extra_iv.rename(columns={'stay_id': 'id'}, inplace=True)
    data_extra_eicu_1.rename(columns={'patientunitstayid': 'id'}, inplace=True)
    data_extra_use_1 = data_extra_iii.append(data_extra_iv).reset_index(drop=True)
  
    data_extra_use_2, data_extra_iii, data_extra_iv, data_extra_eicu_2 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    data_extra_iii = pd.read_csv(data_initial_path + 'text_mimiciii_older_count_use.csv')
    data_extra_iv = pd.read_csv(data_initial_path + 'text_mimiciv_older_count_use.csv')
    data_extra_eicu_2 = pd.read_csv(data_initial_path + 'text_eicu_older_count_use.csv')

    data_extra_use_2 = data_extra_iii.append(data_extra_iv).reset_index(drop=True)
    data_extra_use_2.drop(['label'], axis=1, inplace=True)
    data_extra_eicu_2.drop(['label'], axis=1, inplace=True)

    data_extra_use, data_extra_eicu_use = pd.DataFrame(), pd.DataFrame()
    data_extra_use = pd.merge(data_extra_use_2, data_extra_use_1, on = 'id')
    data_extra_eicu_use = pd.merge(data_extra_eicu_2, data_extra_eicu_1, on = 'id')

    del data_extra_iii, data_extra_iv, data_extra_use_1, data_extra_use_2, data_extra_eicu_1, data_extra_eicu_2

    seq_info = {'seq_240': [240, 20], 'seq_120': [120, 10], 'seq_60': [60, 5]}
    for i in ['seq_240' , 'seq_120', 'seq_60']:
        categorical_all = ['activity_bed', 'activity_eva_flag', 'activity_sit', 'activity_stand'
            , 'admission_type', 'agegroup', 'anchor_year_group', 'epinephrine'
            , 'code_status', 'code_status_eva_flag', 'death_hosp', 'delirium_eva_flag'
            , 'delirium_flag', 'dobutamine', 'dopamine', 'electivesurgery', 'ethnicity'
            , 'fall_risk', 'fall_risk_eva_flag', 'first_careunit', 'gender', 'norepinephrine'
            , 'region', 'teachingstatus', 'vent', 'weightgroup', 'heightgroup', 'vasopressor'
            , 'chief complaint', 'family history', 'history of present illness', 'medications on admission'
            , 'past medical history', 'physical exam', 'social history', 'rrt'
            , 'braden_score_cat', 'braden_flag'
            ]
        for j in ['development', 'temporal', 'external']:
            data = pd.read_csv(data_path + i + '/' + j + '_num_older.csv')
            if j in ['development', 'temporal']:
                data = pd.merge(data, data_extra_use, on = 'id')
            else:
                data = pd.merge(data, data_extra_eicu_use, on = 'id')
            overall_table, group_table = cal_tableone_info(data, 'label', categorical_all)
            overall_table.to_excel(result_path + 'overall_' + j + '_' + i + '.xlsx')
            group_table.to_excel(result_path + 'group_' + j + '_' + i + '.xlsx')


def find_number_after_string(s, target_string):
    # Construct a regular expression pattern to match the number that immediately follows the target string.
    pattern = re.escape(target_string) + r'(\d+)'
    
    # Search for matching patterns.
    match = re.search(pattern, s)
    
    # If a match is found, return the numeric part; otherwise, return an empty string
    return match.group(1) if match else ""


def develop_temp_ext_missing_ratio(data_initial_path, data_path, result_path):
    mimiciii = pd.read_csv(data_initial_path + 'numeric_mimiciii_older.csv').rename(columns={'icustay_id': 'id'})
    mimiciv = pd.read_csv(data_initial_path + 'numeric_mimiciv_older.csv').rename(columns={'stay_id': 'id'})
    mimic = mimiciii.append(mimiciv).reset_index(drop=True)
    eicu = pd.read_csv(data_initial_path + 'numeric_eicu_older.csv').rename(columns={'patientunitstayid': 'id'})
    del mimiciii, mimiciv

    develop_id, temp_id, ext_id, develop_missing, temp_missing, ext_missing = [], [], [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    develop_id = list(pd.read_csv(data_path + 'development_num_older.csv')['id'])
    temp_id = list(pd.read_csv(data_path + 'temporal_num_older.csv')['id'])
    ext_id = list(pd.read_csv(data_path + 'external_num_older.csv')['id'])

    develop_missing = calculate_missing_ratio(mimic[mimic['id'].isin(develop_id)])
    develop_missing.rename(columns={'percent_missing': 'develop'}, inplace=True)
    temp_missing = calculate_missing_ratio(mimic[mimic['id'].isin(temp_id)])
    temp_missing.rename(columns={'percent_missing': 'temp'}, inplace=True)
    ext_missing = calculate_missing_ratio(eicu[eicu['id'].isin(ext_id)])
    ext_missing.rename(columns={'percent_missing': 'ext'}, inplace=True)

    missing_all = pd.DataFrame()
    missing_all = pd.merge(pd.merge(develop_missing, temp_missing, on = 'column_name', how = 'outer'), ext_missing, on = 'column_name', how = 'outer')
    missing_all.to_csv(result_path + 'develop_temporal_ext_set_missing_seq_' + find_number_after_string(data_path, "seq_") + '.csv', index=False)



#                          Part 3. plot AUROC and APPRC with 95%CI for studying models                        #

#  ---------------------------------------------------------------------------------------------------------- #
# get all needs metrics of performance
def model_performance_params(data, data_pred_proba, ts_use, ts_value):
    """
    data: the truth label of target [array]
    data_pred_proba: predict probability of target with one columns [array]
    ts_use: 'True' or 'False' (if true, will use ts_value, else will not use ts_value) [Bool]
    ts_value: float value (if ts_use = 'True', will use it - input the value needed, or not use it)

    """
    fpr, tpr, thresholds_ROC = roc_curve(data, data_pred_proba)
    precision, recall, thresholds = precision_recall_curve(data, data_pred_proba)
    average_precision = average_precision_score(data, data_pred_proba)
    brier_score = brier_score_loss(data, data_pred_proba, pos_label=1)
    roc_auc = auc(fpr, tpr)

    threshold_final = []
    if ts_use == 'False':
        optimal_idx = []
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds_ROC[optimal_idx]
        sensitivity = tpr[optimal_idx]
        specificity = 1 - fpr[optimal_idx]
        data_pred = np.zeros(len(data_pred_proba))
        data_pred[data_pred_proba >= optimal_threshold] = 1
        threshold_final = optimal_threshold
    else:
        optimal_idx = []
        optimal_idx = np.max(np.where(thresholds_ROC >= ts_value))
        sensitivity = tpr[optimal_idx]
        specificity = 1 - fpr[optimal_idx]
        data_pred = np.zeros(len(data_pred_proba))
        data_pred[data_pred_proba >= ts_value] = 1
        threshold_final = ts_value

    tn, fp, fn, tp = confusion_matrix(data, data_pred).ravel()
    accuracy = accuracy_score(data, data_pred)
    F1 = f1_score(data, data_pred)  # not consider the imbalance, using 'binary' 2tp/(2tp+fp+fn)
    precision_c = tp/(tp+fp)

    parameters = {'auc': roc_auc, 'sensitivity': sensitivity, 'specificity': specificity, 'accuracy': accuracy,
                  'F1': F1, 'precision': precision_c, 'ap':average_precision, 'brier_score': brier_score, 'threshold': threshold_final}
    roc_plot_data = {'fpr_data': fpr, 'tpr_data': tpr}
    return parameters, roc_plot_data


def model_performance_params_nobs(data, data_pred_proba, ts_use, ts_value):
    """
    data: the truth label of target [array]
    data_pred_proba: predict probability of target with one columns [array]
    ts_use: 'True' or 'False' (if true, will use ts_value, else will not use ts_value) [Bool]
    ts_value: float value (if ts_use = 'True', will use it - input the value needed, or not use it)

    """
    fpr, tpr, thresholds_ROC = roc_curve(data, data_pred_proba)
    precision, recall, thresholds = precision_recall_curve(data, data_pred_proba)
    average_precision = average_precision_score(data, data_pred_proba)
    roc_auc = auc(fpr, tpr)

    threshold_final = []
    if ts_use == 'False':
        optimal_idx = []
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds_ROC[optimal_idx]
        sensitivity = tpr[optimal_idx]
        specificity = 1 - fpr[optimal_idx]
        data_pred = np.zeros(len(data_pred_proba))
        data_pred[data_pred_proba >= optimal_threshold] = 1
        threshold_final = optimal_threshold
    else:
        optimal_idx = []
        optimal_idx = np.max(np.where(thresholds_ROC >= ts_value))
        sensitivity = tpr[optimal_idx]
        specificity = 1 - fpr[optimal_idx]
        data_pred = np.zeros(len(data_pred_proba))
        data_pred[data_pred_proba >= ts_value] = 1
        threshold_final = ts_value

    tn, fp, fn, tp = confusion_matrix(data, data_pred).ravel()
    accuracy = accuracy_score(data, data_pred)
    F1 = f1_score(data, data_pred)  # not consider the imbalance, using 'binary' 2tp/(2tp+fp+fn)
    precision_c = tp/(tp+fp)

    parameters = {'auc': roc_auc, 'sensitivity': sensitivity, 'specificity': specificity, 'accuracy': accuracy,
                  'F1': F1, 'precision': precision_c, 'ap':average_precision, 'threshold': threshold_final}
    roc_plot_data = {'fpr_data': fpr, 'tpr_data': tpr}
    return parameters, roc_plot_data


# get the index of all values using booststrap
# 'roc_auc', 'sensitivity', 'specificity', 'accuracy', 'F1', 'precision', 'npv', 'ap', 'brier_score'
def model_performance_params_bootstrap(data, data_ths, num_iterations):
    """
    input
    data: dataframe ('true_label', 'probability')
    data_ths: float -- threshold of using
    num_iterations: int -- the iteration time
    output
    stats: dataframe: 'roc_auc', 'sensitivity', 'specificity', 'accuracy', 'F1', 'precision', 'npv', 'ap', 'brier_score'
    """
    n_iterations = num_iterations
    n_size = int(data.shape[0] * 0.80)
    stats = list()
    for i in range(n_iterations):
        data_use = resample(data.values, n_samples=n_size)
        fpr, tpr, thresholds_ROC = roc_curve(data_use['true_label'], data_use['probability'])
        precision, recall, thresholds = precision_recall_curve(data_use['true_label'], data_use['probability'])
        average_precision = average_precision_score(data_use['true_label'], data_use['probability'])
        brier_score = brier_score_loss(data_use['true_label'], data_use['probability'], pos_label=1)
        roc_auc = auc(fpr, tpr)
        optimal_idx = np.max(np.where(thresholds_ROC >= data_ths))
        sensitivity = tpr[optimal_idx]
        specificity = 1 - fpr[optimal_idx]
        data_pred = np.zeros(len(data_use))
        data_pred[data_use['probability'] >= data_ths] = 1
        tn, fp, fn, tp = confusion_matrix(data_use['true_label'], data_pred).ravel()
        accuracy = accuracy_score(data_use['true_label'], data_pred)
        F1 = f1_score(data_use['true_label'], data_pred)
        precision_c = tp / (tp + fp)
        npv = (tn)/(tn+fn)
        score = []
        score = [roc_auc, sensitivity, specificity, accuracy, F1, precision_c, npv, average_precision]
        stats.append(score)
    stats = pd.DataFrame.from_records(stats)
    stats.columns = ['roc_auc', 'sensitivity', 'specificity', 'accuracy', 'F1', 'precision', 'npv', 'ap', 'brier_score']
    
    return stats


# get the index with 95% CI
# 'roc_auc', 'sensitivity', 'specificity', 'accuracy', 'F1', 'precision', 'npv', 'ap', 'brier'
def model_performance_params_bootstrap_95CI(data, data_ths, num_iterations):
    """
    input
    data: dataframe ('true_label', 'probability')
    data_ths: float -- threshold of using
    num_iterations: int -- the iteration time
    output
    stats_new: dataframe -- 95% CI: 'roc_auc', 'sensitivity', 'specificity', 'accuracy', 'F1', 'precision', 'npv', 'ap', 'brier_score'
    """
    n_iterations = num_iterations
    n_size = int(data.shape[0] * 0.90)
    stats = list()
    for i in range(n_iterations):
        data_use = resample(data.values, n_samples=n_size)
        data_use = pd.DataFrame(data_use, columns=['true_label', 'probability'])
        data_use['true_label'] = data_use['true_label'].astype(int)
        fpr, tpr, thresholds_ROC = roc_curve(data_use['true_label'], data_use['probability'])
        precision, recall, thresholds = precision_recall_curve(data_use['true_label'], data_use['probability'])
        average_precision = average_precision_score(data_use['true_label'], data_use['probability'])
        brier_score = brier_score_loss(data_use['true_label'], data_use['probability'], pos_label=1)
        roc_auc = auc(fpr, tpr)
        optimal_idx = np.max(np.where(thresholds_ROC >= data_ths))
        sensitivity = tpr[optimal_idx]
        specificity = 1 - fpr[optimal_idx]
        data_pred = np.zeros(len(data_use))
        data_pred[data_use['probability'] >= data_ths] = 1
        if sum(data_use['true_label']) == 0:
            precision_c, npv = np.nan, np.nan
        else:
            tn, fp, fn, tp = confusion_matrix(data_use['true_label'], data_pred).ravel()
            precision_c = tp / (tp + fp)
            npv = (tn) / (tn + fn)
        accuracy = accuracy_score(data_use['true_label'], data_pred)
        F1 = f1_score(data_use['true_label'], data_pred)
        score = []
        score = [roc_auc, sensitivity, specificity, accuracy, F1, precision_c, npv, average_precision, brier_score]
        stats.append(score)
    stats = pd.DataFrame.from_records(stats)
    stats.columns = ['roc_auc', 'sensitivity', 'specificity', 'accuracy', 'F1', 'precision', 'npv', 'ap', 'brier_score']
    stats = stats.dropna()
    # calculate 95% CI of each model
    alpha = 0.95
    p_l = ((1.0-alpha)/2.0) * 100
    p_u = (alpha+((1.0-alpha)/2.0)) * 100

    stats_new = list()
    for j in range(9): # auc, sen, spe, acc, f1, pre, npv, ap, brier_score
        lower = max(0.0, round(np.percentile(list(stats.iloc[:,j]), p_l),3))
        upper = min(1.0, round(np.percentile(list(stats.iloc[:,j]), p_u),3))
        val = round(np.percentile(list(stats.iloc[:,j]), 50),3)
        val = str(val) + ' (' + str(lower) + '-' + str(upper) + ')'
        stats_new.append(val)
    stats_new = pd.DataFrame.from_records([stats_new])
    stats_new.columns = ['roc_auc', 'sensitivity', 'specificity', 'accuracy', 'F1', 'precision', 'npv', 'ap', 'brier_score']
    
    return stats, stats_new


# get the index with 95% CI
# 'roc_auc', 'sensitivity', 'specificity', 'accuracy', 'F1', 'precision', 'npv', 'ap'
def model_performance_params_bootstrap_95CI_nobs(data, data_ths, num_iterations):
    """
    input
    data: dataframe ('true_label', 'probability')
    data_ths: float -- threshold of using
    num_iterations: int -- the iteration time
    output
    stats_new: dataframe -- 95% CI: 'roc_auc', 'sensitivity', 'specificity', 'accuracy', 'F1', 'precision', 'npv', 'ap'
    """
    n_iterations = num_iterations
    n_size = int(data.shape[0] * 0.90)
    stats = list()
    for i in range(n_iterations):
        data_use = resample(data.values, n_samples=n_size)
        data_use = pd.DataFrame(data_use, columns=['true_label', 'probability'])
        data_use['true_label'] = data_use['true_label'].astype(int)
        fpr, tpr, thresholds_ROC = roc_curve(data_use['true_label'], data_use['probability'])
        precision, recall, thresholds = precision_recall_curve(data_use['true_label'], data_use['probability'])
        average_precision = average_precision_score(data_use['true_label'], data_use['probability'])
        roc_auc = auc(fpr, tpr)
        optimal_idx = np.max(np.where(thresholds_ROC >= data_ths))
        sensitivity = tpr[optimal_idx]
        specificity = 1 - fpr[optimal_idx]
        data_pred = np.zeros(len(data_use))
        data_pred[data_use['probability'] >= data_ths] = 1
        if sum(data_use['true_label']) == 0:
            precision_c, npv = np.nan, np.nan
        else:
            tn, fp, fn, tp = confusion_matrix(data_use['true_label'], data_pred).ravel()
            precision_c = tp / (tp + fp)
            npv = (tn) / (tn + fn)
        accuracy = accuracy_score(data_use['true_label'], data_pred)
        F1 = f1_score(data_use['true_label'], data_pred)
        score = []
        score = [roc_auc, sensitivity, specificity, accuracy, F1, precision_c, npv, average_precision]
        stats.append(score)
    stats = pd.DataFrame.from_records(stats)
    stats.columns = ['roc_auc', 'sensitivity', 'specificity', 'accuracy', 'F1', 'precision', 'npv', 'ap']
    stats = stats.dropna()
    # calculate 95% CI of each model
    alpha = 0.95
    p_l = ((1.0-alpha)/2.0) * 100
    p_u = (alpha+((1.0-alpha)/2.0)) * 100

    stats_new = list()
    for j in range(8): # auc, sen, spe, acc, f1, pre, npv, ap
        lower = max(0.0, round(np.percentile(list(stats.iloc[:,j]), p_l),3))
        upper = min(1.0, round(np.percentile(list(stats.iloc[:,j]), p_u),3))
        val = round(np.percentile(list(stats.iloc[:,j]), 50),3)
        val = str(val) + ' (' + str(lower) + '-' + str(upper) + ')'
        stats_new.append(val)
    stats_new = pd.DataFrame.from_records([stats_new])
    stats_new.columns = ['roc_auc', 'sensitivity', 'specificity', 'accuracy', 'F1', 'precision', 'npv', 'ap']
    
    return stats, stats_new


# get a file all result need - metrics
# model, seq_len, lr, db_type
def all_metric_95ci(data_path, num_iterations, result_path):
    result_need = pd.DataFrame()
    for m_name in ['clinical_longformer']: # 'clinicalbert', 'clinical_longformer', 'clinical_bigbird'
        for s_l in ['seq_240']: # 'seq_120', 
            for lr in ['1e-05', '2e-05', '3e-05', '4e-05']:
                data_all = pd.DataFrame()
                data_all = pd.read_csv(data_path + m_name + '/' + s_l + '/max_len-512_bs-16_epoch-2_lr-' + lr + '/' + 'text_all.csv')
                for i in ['train', 'val', 'test', 'temp', 'ext']:
                    data, para_score, stats_new = pd.DataFrame(), {}, pd.DataFrame()
                    data = data_all.loc[data_all['db_type'] == i].reset_index(drop=True)
                    para_score, _ = model_performance_params(data['all_labels'], data['probs_1'], 'False', 0)
                    _, stats_new = model_performance_params_bootstrap_95CI(data[['all_labels', 'probs_1']], para_score['threshold'], num_iterations)
                    stats_new[['db_type', 'epoch', 'learning_rate', 'seq_len', 'model']] = [i, 2, lr, s_l, m_name]
                    result_need = result_need.append(stats_new)
            for ep in [1, 2, 3, 4]:
                data_all = pd.DataFrame()
                data_all = pd.read_csv(data_path + m_name + '/' + s_l + '/max_len-512_bs-16_epoch-' + str(ep) + '_lr-1e-05' + '/' + 'text_all.csv')
                for i in ['train', 'val', 'test', 'temp', 'ext']:
                    data, para_score, stats_new = pd.DataFrame(), {}, pd.DataFrame()
                    data = data_all.loc[data_all['db_type'] == i].reset_index(drop=True)
                    para_score, _ = model_performance_params(data['all_labels'], data['probs_1'], 'False', 0)
                    _, stats_new = model_performance_params_bootstrap_95CI(data[['all_labels', 'probs_1']], para_score['threshold'], num_iterations)
                    stats_new[['db_type', 'epoch', 'learning_rate', 'seq_len', 'model']] = [i, ep, '1e-05', s_l, m_name]
                    result_need = result_need.append(stats_new)                    
    
    result_need.reset_index(drop=True, inplace=True)
    result_need.to_csv(result_path + 'all_metric_95ci.csv',index=False)


# plot the needed - axis: x = learning rate, y = auroc
def plot_confidence_interval(data, color, name, horizontal_line_width=0.25):

    for lr in [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]:
        mean = data.loc[data['learning_rate'] == lr, 'auc'].values[0]
        top = data.loc[data['learning_rate'] == lr, 'auc_upper'].values[0]
        bottom = data.loc[data['learning_rate'] == lr, 'auc_lower'].values[0]
        x = int(lr*(1e+5))
        left = x - horizontal_line_width / 2
        right = x + horizontal_line_width / 2
        plt.plot([x, x], [top, bottom], color=color)
        plt.plot([left, right], [top, top], color=color)
        plt.plot([left, right], [bottom, bottom], color=color)
        plt.plot(x, mean, 'o-', color=color)

    x = [1,2,3,4,5]
    y = list(data.sort_values(by=['learning_rate'])['auc'])
    plt.plot(x, y, color=color, label=name)
    plt.xticks([1, 2, 3, 4, 5], ['1e-5', '2e-5', '3e-5', '4e-5', '5e-5'])


def all_model_seq_len_lr_compare(data, path_save):
    for db in ['test', 'temp']:
        for s_l in ['seq_120', 'seq_240']:
            color_m = {'clinicalbert': 'blue', 'clinical_longformer': 'red', 'clinical_bigbird': 'green'}
            name_map = {'clinicalbert': 'Bio-ClinicalBert', 'clinical_longformer': 'Clinical-Longformer', 'clinical_bigbird': 'Clinical-BigBird'}
            plt.rcParams['figure.dpi'] = 600
            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams['figure.figsize'] = (5, 4.3)
            for m_name in ['clinicalbert', 'clinical_longformer', 'clinical_bigbird']:
                data_each = pd.DataFrame()
                data_each = data.loc[(data['seq_len'] == s_l) & (data['db_type'] == db) & (data['model'] == m_name)].reset_index(drop=True)
                data_each[['auc', 'auc_range']] = data_each['roc_auc'].str.split(' ', 1, expand=True)
                data_each[['auc_lower', 'auc_upper']] = data_each['auc_range'].str.split('-', 1, expand=True)
                data_each[['auc_lower', 'auc_upper']] = data_each[['auc_lower', 'auc_upper']].replace('\(|\)','',regex=True)
                data_each[['auc', 'auc_lower', 'auc_upper']] = data_each[['auc', 'auc_lower', 'auc_upper']].astype(float)
                plot_confidence_interval(data=data_each, color=color_m[m_name], name=name_map[m_name])
            plt.legend(loc="upper right")
            plt.xlabel('Learning rate', fontsize=12)
            plt.ylabel('AUROC', fontsize=12)
            plt.legend(fontsize=13)
            plt.savefig(path_save + db + '_' + s_l + '_auc.png')
            plt.show()


def all_selected_models_95ci(data_path, num_iterations, result_path):
    result_need = pd.DataFrame()
    for db_type in ['test', 'temp', 'ext']:
        data_each = pd.DataFrame()
        data_each = pd.read_csv(data_path + 'roc_' + db_type + '_set.csv', index_col=[0])
        data_each.rename(columns={'preICU_risk_score_raw': 'preICU_risk_score'}, inplace=True)
        # # sofa, saps - map to probability
        # # 1 / (1 + exp(- (-3.3890 + 0.2439*(sf.sofa) ))) as sofa_prob
        # # 1 / (1 + exp(- (-7.7631 + 0.0737*(sapsii) + 0.9971*(ln(sapsii + 1))) )) as sapsii_prob
        # data_each['sofa_pb'] = 1/(1 + np.exp(-(-3.3890 + 0.2439*data_each['sofa'])))
        # data_each['saps_pb'] = 1/(1+np.exp(-(-7.7631 + 0.0737*(data_each['saps']) + 0.9971*(np.log(data_each['saps'] + 1)))))
        # data_each = data_each.round({'sofa_pb': 5, 'saps_pb': 5})
        # data_each.loc[data_each['sofa_pb'] > 1] = 1
        # data_each.loc[data_each['sofa_pb'] < 0] = 0
        # data_each.loc[data_each['saps_pb'] > 1] = 1
        # data_each.loc[data_each['saps_pb'] < 0] = 0
        # data_each.drop(['sofa', 'saps'], axis=1, inplace=True)
        # data_each.rename(columns={'sofa_pb': 'sofa', 'saps_pb': 'saps'}, inplace=True)        
        for m_name in ['pred_f_num_text', 'preICU_risk_score', 'pred_f_num', 'sofa', 'saps']:
            para_score, stats_new = {}, pd.DataFrame()
            # print(db_type, m_name)
            para_score, _ = model_performance_params_nobs(data_each['label'], data_each[m_name], 'False', 0)
            _, stats_new = model_performance_params_bootstrap_95CI_nobs(data_each[['label', m_name]], para_score['threshold'], num_iterations)
            stats_new[['m_name', 'db_type']] = [m_name, db_type]
            result_need = result_need.append(stats_new)
    
    result_need.reset_index(drop=True, inplace=True)
    result_need.to_csv(result_path + 'all_selected_models_95ci.csv',index=False)


# calculate nomogram
# https://www.dounaite.com/article/625e4871f86aba5c787d998b.html
def nomogram_score(ls_right_value, ls_left_value, ls_or, ls_xvar):
    # Provide the values on the right and left sides of the nomogram. For categorical variables, the values are 1 and 0. For multicategorical variables, the values are multiple 1s and 0s.
    ls_beta=[np.log(x) for x in ls_or]
    ls_beta_abs=[np.abs(x) for x in ls_beta]
    ls_distance_abs=[np.abs(a-b) for a,b in zip(ls_right_value,ls_left_value)]# The difference between the right and left values on each scale
    ls_pi_pre=[a*b for a,b in zip(ls_beta_abs,ls_distance_abs)]
    ls_max_score=[] # Calculate the maximum score for each variable
    for pi_pre in ls_pi_pre:        
        max_score=np.divide(pi_pre,np.max(ls_pi_pre))*100
        ls_max_score.append(max_score)
    ls_unit_score=[a/b for a,b in zip(ls_max_score,ls_distance_abs)] # Calculate the score for each scale unit of each variable
    ls_actual_distance=[a-b for a,b in zip(ls_xvar, ls_left_value)] # Calculate the actual total score
    ls_actual_distance_abs=map(np.abs,ls_actual_distance)
    ls_score=[a*b for a,b in zip(ls_unit_score,ls_actual_distance_abs)]
    total_score=0
    for i,val in enumerate(ls_score):
        total_score +=ls_score[i]
    # ls_unit_score_use = [] # Calculate the score for each scale unit of each variable
    # for a, b in zip(ls_beta, ls_unit_score):
    #     if a < 0:
    #         ls_unit_score_use.append(-round(b,4))
    #     else:
    #         ls_unit_score_use.append(round(b,4))
    return ls_unit_score, ls_score, total_score


def split_notes_lr_models_95ci(data_path, num_iterations, result_path):
    result_need = pd.DataFrame()
    
    for db_type in ['test', 'temp', 'ext']:      
        for t_name in ['', 'chief_complaint', 'history_of_present_illness', 
        'medications_on_admission', 'past_medical_history', 'physical_exam']: 
            data_each = pd.DataFrame()
            if t_name == '':
                data_each = pd.read_csv(data_path + 'roc_' + db_type + t_name + '_set.csv', index_col=[0]) 
            else:
                data_each = pd.read_csv(data_path + 'roc_' + db_type + '_' + t_name + '_set.csv', index_col=[0]) 
            para_score, stats_new = {}, pd.DataFrame()
            para_score, _ = model_performance_params_nobs(data_each['label'], data_each['pred_f_num_text'], 'False', 0)
            _, stats_new = model_performance_params_bootstrap_95CI_nobs(data_each[['label', 'pred_f_num_text']], para_score['threshold'], num_iterations)
            if t_name == '':
                stats_new[['note_name', 'db_type']] = ['all', db_type]
            else:
                stats_new[['note_name', 'db_type']] = [t_name.replace("_", " "), db_type]
            result_need = result_need.append(stats_new)
    
    result_need.reset_index(drop=True, inplace=True)
    result_need.to_csv(result_path + 'split_notes_lr_models_95ci.csv',index=False)


# change the results to tableone format to directly calculate the p values (models & scores)
def auc_ap_model_score_vs_pvalue(data, test_set_name, path_save):
    for pvalue_name in ['roc_auc', 'ap']:
        data_pvalue_use = []
        model_name_list = ['preICU_risk_score', 'pred_f_num', 'saps', 'sofa', 'eldericu', 'rf', 'xgb']
        for name in model_name_list:
            data_pvalue = pd.DataFrame()
            data_pvalue = data.loc[data['model_name'] == name, pvalue_name].values.reshape(-1,1)
            if len(data_pvalue_use) == 0:
                data_pvalue_use = data_pvalue
            else:
                data_pvalue_use = np.hstack((data_pvalue_use, data_pvalue))
        data_pvalue_use = pd.DataFrame(data_pvalue_use, columns=model_name_list)
        data_pvalue_use['label'] = 0
        # append gemini-icu results (base)
        data_pvalue_base = pd.DataFrame()
        data_pvalue_base = data.loc[data['model_name'] == 'pred_f_num_text', pvalue_name].reset_index(drop=True)
        data_pvalue_base = pd.concat([data_pvalue_base] * (len(model_name_list)), axis=1, ignore_index=True)
        data_pvalue_base.columns = model_name_list
        data_pvalue_base['label'] = 1
        data_pvalue_use = data_pvalue_use.append(data_pvalue_base).reset_index(drop=True)
        groupby, columns = '', []
        groupby = 'label'
        columns = data_pvalue_use.columns.values.tolist()
        nonnormal = [x for x in columns if x != 'label']
        categorical = []
        group_table = TableOne(data_pvalue_use, columns=columns, categorical=categorical, \
                               groupby=groupby, nonnormal=nonnormal, decimals=3, pval=True)
        group_table.to_excel(path_save + test_set_name + '_' + pvalue_name + '_pvalue.xlsx')


def split_notes_auroc(data_path, result_path):
    data = pd.read_csv(data_path + 'all_subtext_performance_95ci.csv')
    data['roc_auc_mean'] = data['roc_auc'].str.split(expand=True).loc[:,0].astype(float)
    data = data[['roc_auc_mean', 'text_name', 'db_type']]

    for j in ['test', 'temp', 'ext']:
        data_new = pd.DataFrame()
        for i in ['all', 'physical exam', 'history of present illness', 
            'chief complaint', 'past medical history', 'medications on admission']:
            data_each = []
            data_each = data.loc[(data['text_name'] == i.replace(" ", "_")) & (data['db_type'] == j)]
            data_new = data_new.append(data_each)

        data_new.reset_index(drop=True, inplace=True)
        data_new = data_new.loc[::-1].reset_index(drop=True)
        text_name = data_new['text_name']
        roc_auc_mean = data_new['roc_auc_mean']

        plt.rcParams['figure.dpi'] = 600
        plt.rcParams["font.family"] = "Times New Roman"
        
        # Figure Size
        fig, ax = plt.subplots(figsize =(14, 9))
        
        # Horizontal Bar Plot
        ax.barh(text_name, roc_auc_mean, color='dodgerblue')
        
        # Show top values
        # ax.invert_yaxis()
        ax.set_xlim(0.6, 0.92)

        # Show Plot
        plt.savefig(result_path + 'split_note_auroc_' + j + '.svg')
        # plt.show()



#                       Part 4. ELDER-ICU data prepare and evaluate                        #

#  --------------------------------------------------------------------------------------- #
def data_process_eldericu(data_use, data_use_name, outlier_range_check, result_path):
    """
    :param data_use: initial data without processing
    :para data_use_name: the group name like 'death_hosp'
    :param outlier_range_check: the initial name outlier like hearrate, resp_rate
    :param result_path: save path of results
    :return data_use: remove outlier
    """    

    data_use_final = pd.DataFrame() # create to generate

    # [1]. drop outliers - Amplification the expression: add feature_max/min/mean
    outlier_range_check_new = pd.DataFrame()
    outlier_range_check_new = pd.concat([outlier_range_check, outlier_range_check, outlier_range_check, outlier_range_check], axis=0)
    outlier_range_check_new.rename(columns={'Unnamed: 0': 'index_name'}, inplace=True)
    outlier_range_check.rename(columns={'Unnamed: 0': 'index_name'}, inplace=True)
    # according to the statistic features' name to generate the outlier check columns
    add_fea = ['', '_min', '_max', '_mean']
    new_index = []
    for i in add_fea:
        for j in outlier_range_check['index_name'].tolist():
            new_index.append(j + i)
    outlier_range_check_new['index_name'] = new_index
    outlier_range_check_new.reset_index(drop=True, inplace=True)

    data_use = outlier_value_nan(data_use, outlier_range_check_new)

    # [2]. calculate missing ratio
    calculate_missing_ratio(data_use).to_csv(result_path + data_use_name + '_missing.csv', index=False)

    # get the impute data's tableone info to support to check data
    categorical_all = ['activity_bed', 'activity_eva_flag', 'activity_sit', 'activity_stand'
        , 'admission_type', 'agegroup', 'anchor_year_group', 'epinephrine'
        , 'code_status', 'code_status_eva_flag', 'death_hosp'
        , 'dobutamine', 'dopamine', 'electivesurgery', 'ethnicity'
        , 'first_careunit', 'gender', 'norepinephrine'
        , 'region', 'teachingstatus', 'vent', 'weightgroup', 'heightgroup']
    
    overall_table, group_table = cal_tableone_info(
        data_use.drop(['uniquepid', 'patienthealthsystemstayid', 'subject_id', 'hadm_id'], errors='ignore', axis=1), 'death_hosp', categorical_all)
    overall_table.to_excel(result_path + 'overall_' + data_use_name + '.xlsx')
    group_table.to_excel(result_path + 'group_' + data_use_name + '.xlsx')
    del overall_table, group_table

    # [3]. map string to int
    data_use = older_string_int(data_use)
    data_use['admission_type'] = data_use['admission_type'].map({'EMERGENCY':1, 'URGENT':1, 'OBSERVATION': 0})
    
    # [4] remove no need variables
    drop_names = ['agegroup', 'anchor_year_group', 'apache_iva', 'apache_iva_prob', 'deathtime_icu_hour',
        'first_careunit', 'hadm_id', 'heightgroup', 'hospitalid', 'los_hospital_day', 'los_icu_day', 'patienthealthsystemstayid',
        'patientid', 'predictedhospitallos_iv', 'predictedhospitallos_iva', 'predictediculos_iv', 'predictediculos_iva',
        'region', 'subject_id', 'teachingstatus', 'uniquepid', 'weightgroup', 'bmi',
        'troponin_max', 'fibrinogen_min', 'bnp_max', 'apache_iv', 'apache_iv_prob',
        'oasis', 'oasis_prob', 'saps', 'saps_prob', 'sofa', 'sofa_prob', 'apsiii', 'apsiii_prob'
        ]
    data_use.drop(drop_names, axis=1, inplace=True, errors='ignore')

    return data_use


def generate_data_imputation_miceforest(data_all, data_split_info, data_save_path):
    """
    :param data_all: all datasets' dictionary {'train', 'val', 'test', 'ext'}
    :param data_split_info: train, val, test, temp, ext id info
    :param data_save_path: imputed data save path
    :return data_use_final: directly save the imputed data
    """ 
    # [1] split to impute and non-impute, save
    data_use = {}
    for i in ['train', 'val', 'test', 'temp', 'ext']:
        data_impute, data_non_impute = pd.DataFrame(), pd.DataFrame()

        list_nonimpute_name = [
        'code_status', 'code_status_eva_flag', 'death_hosp', 'activity_eva_flag',
        'activity_bed', 'activity_sit', 'activity_stand', 'admission_type', 
        'dobutamine', 'dopamine', 'electivesurgery', 'epinephrine', 
        'ethnicity', 'gender', 'norepinephrine', 'vent', 'id']
        data_non_impute = data_all[i][list_nonimpute_name]

        list_impute_name = list(set(data_all[i].columns.to_list()) - set(list_nonimpute_name))
        data_impute = data_all[i][list_impute_name + ['id']]
        data_impute[list_impute_name] = data_impute[list_impute_name].astype(float)
        data_impute = data_impute.reindex(sorted(data_impute.columns), axis=1)        
        data_use[i] = {'impute_before':data_impute, 'nonimpute':data_non_impute}

    # [2] imputation using miceforest
    # Create kernel and imputation.
    kernel = mf.ImputationKernel(data_use['train']['impute_before'], datasets=4, save_all_iterations=True, save_models=1, random_state=2)
    kernel.mice(iterations=3, n_jobs=-1)
    data_use['train']['imputed'] = {}
    for m in range(4):
        data_use['train']['imputed'][m] = kernel.complete_data(m)
    for j in ['val', 'test', 'temp', 'ext']:
        kernel_new = kernel.impute_new_data(data_use[j]['impute_before'])
        data_use[j]['imputed'] = {}
        for m in range(4):
            data_use[j]['imputed'][m] = kernel_new.complete_data(m)

    # [3] change type and merge label
    for i in ['train', 'val', 'test', 'temp', 'ext']:
        data_average = pd.DataFrame()
        for m in range(4):
            data_each = pd.DataFrame()
            data_each = data_use[i]['imputed'][m]
            # notice several values were still nan and were imputed as median values
            aa = pd.Series(dtype=object)
            aa = data_each[data_each.columns.to_list()].median()
            data_each = data_each[data_each.columns.to_list()].fillna(aa)
            del aa
            data_each[['cci_score', 'fio2_max', 'gcs_min']] = data_each[['cci_score', 'fio2_max', 'gcs_min']].round(0).astype(int)
            data_average = data_average.append(data_each)
        data_average = data_average.groupby(['id']).mean().reset_index()
        data_average.loc[data_average['fio2_max'] > 100, 'fio2_max'] = 100
        data_average.loc[data_average['fio2_max'] < 21, 'fio2_max'] = 21
        data_average.loc[data_average['gcs_min'] > 15, 'gcs_min'] = 15
        data_average.loc[data_average['gcs_min'] < 3, 'gcs_min'] = 3
        data_average = pd.merge(data_average, data_use[i]['nonimpute'].astype('Int64'), on='id')

        # [4] get BMI, shock index, bun_creatinine, egfr, GNRI, nlr info
        data_average['bmi'] = 10000*data_average['weight']/(data_average['height']**2)
        data_average['bmi'] = data_average['bmi'].round(2)

        data_average['shock_index'] = (data_average['heart_rate_mean']/data_average['sbp_mean']).round(2)
        data_average['bun_creatinine'] = (data_average['bun_max'] / data_average['creatinine_max']).round(2)

        # egfr: gender, creatinine_max, age, ethnicity
        egfr = pd.DataFrame()
        egfr = data_average[['id', 'gender', 'age', 'ethnicity', 'creatinine_max']]
        egfr['egfr'] = 186*(egfr['creatinine_max'].pow(-1.154))*(egfr['age'].pow(-0.203))
        egfr.loc[(egfr['gender'] == 0) & (egfr['ethnicity'] == 1), 'egfr'] = 0.742*1.210*egfr['egfr']
        egfr.loc[(egfr['gender'] == 0) & (egfr['ethnicity'] != 1), 'egfr'] = 0.742*egfr['egfr']
        egfr['egfr'] = egfr['egfr'].round(2)

        # ideal weight = height (cm) - 100 - ([height(cm) - 150]/4) for men
        # ideal weight = height (cm) - 100 - ([height(cm) - 150]/2.5) for women
        # GNRI = [14.89*albumin(g/dL)] + [41.7*(weight/ideal weight)]
        gnri = pd.DataFrame()
        gnri = data_average[['id', 'gender', 'albumin_min', 'weight', 'height']]
        gnri['ideal_weight'] = gnri['height'] - 100 - ((gnri['height'] - 150)/4)
        gnri.loc[gnri['gender'] == 0, 'ideal_weight'] = gnri['height'] - 100 - ((gnri['height'] - 150)/2.5)
        gnri['gnri'] = (14.89*gnri['albumin_min']) + (41.7*(gnri['weight']/gnri['ideal_weight']))
        gnri['gnri'] = gnri['gnri'].round(2)

        nlr = pd.DataFrame() # nlr: Neutrophil-to-Lymphocyte Ratio
        nlr = data_average[['id', 'neutrophils_min', 'lymphocytes_min']]
        nlr['nlr'] = 0
        nlr.loc[nlr['lymphocytes_min'] > 0, 'nlr'] = nlr['neutrophils_min']/nlr['lymphocytes_min']
        nlr['nlr'] = nlr['nlr'].round(2)

        data_average.drop(['ethnicity', 'electivesurgery', 'height', 'weight'], axis=1, inplace=True, errors='ignore') # not be considered in the study
        data_average = pd.merge(data_average, gnri[['id', 'gnri']], on='id')
        data_average = pd.merge(data_average, egfr[['id', 'egfr']], on='id')
        data_average = pd.merge(data_average, nlr[['id', 'nlr']], on='id')

        # [5] category features reset dtype to int
        columns_int_names = ['activity_bed', 'activity_sit', 'activity_stand', 'admission_type', \
                            'code_status', 'death_hosp', 'dobutamine', 'dopamine', 'epinephrine', \
                            'gender', 'norepinephrine', 'vent', 'activity_eva_flag', 'code_status_eva_flag',\
                            'cci_score', 'fio2_max', 'gcs_min']
        data_average[columns_int_names] = data_average[columns_int_names].astype(int)

        # [6] no need study columns - drop again
        data_average.drop(['albumin_min'], axis=1, inplace=True)

        # [7] save the imputation data
        data_average.to_csv(data_save_path + i + '_imputation.csv', index=False)


def older_eldericu_model_eva(data, model_result_info):
    """
    :param data: data_type2 dict
    :para model_result_info: dict of {'model_use':{'existing': , 'params':}, 'model_path_full_info':, 'ths_use':, 'ths_value':, 'cal_yn': , 
    #                          'shap_info':{'shap_yn':, 'image_full_info': , 'image_full_info_bar': , 'fea_table_full_info': },
    #                          'no_cal_model_full_info': , 'cal_model_full_info': }
    # like: '.../xgb_cal.dat', '.../features_ranking.png', '.../features_ranking.csv', '.../no_cal.dat', '.../cal.dat'
    :return data: without considering the group label
    """    
    # XGBoost
    # data = {'test', 'temp', 'ext'}
    with open(model_result_info['result_path'] + 'all_eledericu_evaluate_info.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['datatype_name', 'roc_auc', 'sensitivity', 'specificity', 'accuracy', 'F1', 'precision', 'ap', 'brier_score', 'threshold'])

    # 1. load model                                                                        
    clf_XG_bs = pickle.load(open(model_result_info['model_path_full_info'], "rb"))
    model_use = clf_XG_bs

    # 2. get performance results
    for i in ['test', 'temp', 'ext']:
        X_test_type2, y_test_type2 = pd.DataFrame(), pd.Series([])
        X_test_type2 = data[i].drop(['id','death_hosp'], axis=1)
        X_test_type2 = X_test_type2.reindex(sorted(X_test_type2.columns), axis=1)
        y_test_type2 = data[i]['death_hosp']
        predicted_XG = model_use.predict(X_test_type2)
        probas_XG = model_use.predict_proba(X_test_type2)
        para_XG_bs, roc_plot_XG = model_performance_params(y_test_type2, probas_XG[:, 1], model_result_info['ths_use'], model_result_info['ths_value'])        
        result_each = []
        result_each = [i, round(para_XG_bs['auc'],3),
                    round(para_XG_bs['sensitivity'],3), round(para_XG_bs['specificity'],3), round(para_XG_bs['accuracy'],3),
                    round(para_XG_bs['F1'],3), round(para_XG_bs['precision'],3), round(para_XG_bs['ap'],3), 
                    round(para_XG_bs['brier_score'],3), para_XG_bs['threshold']]

        roc_result_need = pd.DataFrame()
        roc_result_need['id'] = data[i]['id']
        roc_result_need['true_label'] = y_test_type2
        roc_result_need['eldericu'] = probas_XG[:, 1]

        # print and save all needed results
        roc_result_need.to_csv(model_result_info['result_path'] + i + '_eledericu_evaluate' + '.csv', index=False)
        print(result_each)
        result_all = []
        result_all = open(model_result_info['result_path'] + 'all_eledericu_evaluate_info.csv', 'a', newline='')
        writer = csv.writer(result_all)
        writer.writerow(result_each)
        result_all.close()



#                               Part 5. compare with xgboost, rf models                                  #

#  ----------------------------------------------------------------------------------------------------- #
def older_xgb_model(data, model_result_info):
    """
    :param data: data_type2 dict
    :para model_result_info: dict of {'model_use':{'existing': , 'params':}, 'model_path_full_info':, 'ths_use':, 'ths_value':, 'cal_yn': , 
    #                          'shap_info':{'shap_yn':, 'image_full_info': , 'image_full_info_bar': , 'fea_table_full_info': },
    #                          'no_cal_model_full_info': , 'cal_model_full_info': }
    # like: '.../xgb_cal.dat', '.../features_ranking.png', '.../features_ranking.csv', '.../no_cal.dat', '.../cal.dat'
    :return data: without considering the group label
    """    
    # XGBoost
    # data = {'train', 'val', 'test', 'temp', 'ext'}
    with open(model_result_info['result_path'] + 'all_xgb_evaluate_info.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['datatype_name', 'roc_auc', 'sensitivity', 'specificity', 'accuracy', 'F1', 'precision', 'ap', 'brier_score', 'threshold'])

    # 0. define the variables
    predicted_XG, probas_XG, para_XG, roc_plot_XG, parameters = [], [], {}, {}, {}
    fpr_XG, tpr_XG, threshold_XG = [], [], []
    X_train_t_XG, X_test_t_XG, y_train_t_XG, y_test_t_XG = \
    pd.DataFrame(), pd.DataFrame(), pd.Series([]), pd.Series([])

    # 1. load data (keep the same type of the eldericu project)
    X_train_t_XG = data['train'].drop(['id', 'label'], axis=1)
    X_test_t_XG = data['val'].drop(['id', 'label'], axis=1)
    y_train_t_XG = data['train']['label']
    y_test_t_XG = data['val']['label']

    # 2. train a model                                                                        
    params = [] # acquire according to using bayesian optimization 
    params = {
        'base_score': 0.5, 'booster': 'gbtree', 'colsample_bylevel': 1, 'colsample_bynode': 1,
        'colsample_bytree': 1, 'gamma': 0, 'learning_rate': 0.05, 'max_delta_step': 0,
        'max_depth': 5, 'min_child_weight': 4.0, 'n_estimators': 860,
        'n_jobs': -1, 'nthread': None, 'objective': 'binary:logistic', 'random_state': 0,
        'reg_alpha': 0, 'reg_lambda': 1, 'scale_pos_weight': 1,
        'silent': None, 'subsample': 0.8500000000000001, 'verbosity': 1
    }
    clf_XG_bs = xgb.XGBClassifier(**params)
    clf_XG_bs.fit(
        X_train_t_XG, y_train_t_XG, 
        early_stopping_rounds=80, eval_metric="auc",
        eval_set=[(X_test_t_XG, y_test_t_XG)])
    model_use = clf_XG_bs
    # save model
    pickle.dump(model_use, open(model_result_info['no_cal_model_full_info'], "wb"))

    # 3. get performance results
    for i in ['test', 'temp', 'ext']:
        X_test_type2, y_test_type2 = pd.DataFrame(), pd.Series([])
        X_test_type2 = data[i].drop(['id', 'label'], axis=1)
        y_test_type2 = data[i]['label']
        predicted_XG = model_use.predict(X_test_type2)
        probas_XG = model_use.predict_proba(X_test_type2)
        para_XG_bs, roc_plot_XG = model_performance_params(y_test_type2, probas_XG[:, 1], model_result_info['ths_use'], model_result_info['ths_value'])        
        result_each = []
        result_each = [i, round(para_XG_bs['auc'],3),
                    round(para_XG_bs['sensitivity'],3), round(para_XG_bs['specificity'],3), round(para_XG_bs['accuracy'],3),
                    round(para_XG_bs['F1'],3), round(para_XG_bs['precision'],3), round(para_XG_bs['ap'],3), 
                    round(para_XG_bs['brier_score'],3), para_XG_bs['threshold']]
        print(result_each)
        result_all = []
        result_all = open(model_result_info['result_path'] + 'all_xgb_evaluate_info.csv', 'a', newline='')
        writer = csv.writer(result_all)
        writer.writerow(result_each)
        result_all.close()

        roc_result_need = pd.DataFrame()
        roc_result_need['id'] = data[i]['id']
        roc_result_need['true_label'] = y_test_type2
        roc_result_need['xgb'] = probas_XG[:, 1]

        # save all needed results
        roc_result_need.to_csv(model_result_info['result_path'] + i + '_xgb_evaluate' + '.csv', index=False)


def older_rf_model(data, model_result_info):
    # Random forest
    # data = {'train', 'val', 'test', 'temp', 'ext'}
    with open(model_result_info['result_path'] + 'all_rf_evaluate_info.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['datatype_name', 'roc_auc', 'sensitivity', 'specificity', 'accuracy', 'F1', 'precision', 'ap', 'brier_score', 'threshold'])

    # 0. define the variables
    predicted_RF, probas_RF, para_RF, roc_plot_RF, parameters = [], [], {}, {}, {}
    fpr_RF, tpr_RF, threshold_RF = [], [], []
    X_train_type2, X_cal_type2, y_train_type2, y_cal_type2 = \
    pd.DataFrame(), pd.DataFrame(), pd.Series([]), pd.Series([])

    # 1. load data (keep the same type of the eldericu project)
    X_train_type2 = data['train'].drop(['id', 'label'], axis=1)
    X_cal_type2 = data['val'].drop(['id', 'label'], axis=1) # no need
    y_train_type2 = data['train']['label']
    y_cal_type2 = data['val']['label'] # no need

    # 2. train a model                                                                        
    clf_RF_bs = RandomForestClassifier(
        max_features=8, n_jobs=-1, oob_score=True, random_state=0,
        max_depth=8, n_estimators=770
    ) # Bayesian tuning acquired
    clf_RF_bs = clf_RF_bs.fit(X_train_type2, y_train_type2)
    model_use = clf_RF_bs
    # save model
    pickle.dump(model_use, open(model_result_info['no_cal_model_full_info'], "wb"))

    # 3. get performance results
    for i in ['test', 'temp', 'ext']:
        X_test_type2, y_test_type2 = pd.DataFrame(), pd.Series([])
        X_test_type2 = data[i].drop(['id', 'label'], axis=1)
        y_test_type2 = data[i]['label']
        predicted_RF = model_use.predict(X_test_type2)
        probas_RF = model_use.predict_proba(X_test_type2)
        para_RF_bs, roc_plot_RF = model_performance_params(y_test_type2, probas_RF[:, 1], model_result_info['ths_use'], model_result_info['ths_value'])        
        result_each = []
        result_each = [i, round(para_RF_bs['auc'],3),
                    round(para_RF_bs['sensitivity'],3), round(para_RF_bs['specificity'],3), round(para_RF_bs['accuracy'],3),
                    round(para_RF_bs['F1'],3), round(para_RF_bs['precision'],3), round(para_RF_bs['ap'],3), 
                    round(para_RF_bs['brier_score'],3), para_RF_bs['threshold']]
        print(result_each)
        result_all = []
        result_all = open(model_result_info['result_path'] + 'all_rf_evaluate_info.csv', 'a', newline='')
        writer = csv.writer(result_all)
        writer.writerow(result_each)
        result_all.close()

        roc_result_need = pd.DataFrame()
        roc_result_need['id'] = data[i]['id']
        roc_result_need['true_label'] = y_test_type2
        roc_result_need['rf'] = probas_RF[:, 1]

        # save all needed results
        roc_result_need.to_csv(model_result_info['result_path'] + i + '_rf_evaluate' + '.csv', index=False)