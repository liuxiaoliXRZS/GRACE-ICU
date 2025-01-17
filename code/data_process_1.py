from xml.dom.minidom import Element
import numpy as np
import pandas as pd
import re
import os
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import general_utils as gu

project_path = '.../gemini_icu/' # here is the project path



#                   STEP1. get the needed text and numeric data of mimiciii, mimiciv, eicu                 #
#
#    -------------------------------------------------------------------------------------------------     #
data_path, result_path = '', ''
data_path = project_path + 'data/'
outlier_range_check = pd.read_csv(data_path + 'outlier_range_check.csv') # outlier info for numeric data
no_meaning_contents = pd.read_csv(data_path + 'no_meaning_contents.csv') # no meaning contents for notes
result_path = project_path + 'result/initial_data_analysis/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
seq_info = {'seq_240': [240, 20], 'seq_120': [120, 10], 'seq_60': [60, 5]} # seq_len, slide_len

for db_name in ['mimiciii','mimiciv', 'eicu']:
    text_info, data_numeric = pd.DataFrame(), pd.DataFrame()
    if db_name == 'mimiciv':
        text_info = pd.read_csv(data_path + 'db_generation/' + 'text_mimiciv_older.csv')
        # special process the chief complain: ___ Complaint:
        for mm in range(text_info.shape[0]):
            data_each_mm = ''
            data_each_mm = str(text_info.loc[mm, 'text'])
            data_each_mm = data_each_mm.replace('___ Complaint:', 'chief complaint:')
            text_info.loc[mm, 'text'] = data_each_mm
        del mm, data_each_mm
        text_info.rename(columns={'stay_id': 'id'}, inplace=True)
        data_numeric = pd.read_csv(data_path + 'db_generation/' + 'numeric_mimiciv_older.csv')
        data_numeric.rename(columns={'stay_id': 'id'}, inplace=True)
    elif db_name == 'mimiciii':
        text_info = pd.read_csv(data_path + 'db_generation/' + 'text_mimiciii_older.csv')
        text_info.rename(columns={'icustay_id': 'id'}, inplace=True)
        data_numeric = pd.read_csv(data_path + 'db_generation/' + 'numeric_mimiciii_older.csv')
        data_numeric.rename(columns={'icustay_id': 'id'}, inplace=True)
    else:
        text_info = pd.read_csv(data_path + 'db_generation/' + 'text_eicu_older.csv')
        text_info.rename(columns={'patientunitstayid': 'id'}, inplace=True)
        data_numeric = pd.read_csv(data_path + 'db_generation/' + 'numeric_eicu_older.csv')
        data_numeric.rename(columns={'patientunitstayid': 'id'}, inplace=True)


    text_info = text_info.merge(data_numeric[['id', 'death_hosp']], how='left').reset_index(drop=True)
    if db_name in ['mimiciii','mimiciv']:
        text_info.dropna(subset=['text'], inplace=True)
    else:
        rows_to_drop = text_info.drop('id', axis=1).isnull().all(axis=1)
        text_info = text_info[~rows_to_drop]
        del rows_to_drop
    text_info.reset_index(drop=True, inplace=True)

    text_info_new = pd.DataFrame()
    if db_name in ['mimiciii','mimiciv']:
        text_info_new = gu.get_need_type_text(text_info)[['id', 'text_need', 'death_hosp']]
    else:
        text_info_new = gu.get_need_type_text_spec(text_info)[['id', 'text_need', 'death_hosp']]
    text_info_new['text_need_len'] = text_info_new['text_need'].str.len()
    text_info_new = text_info_new.loc[text_info_new['text_need_len']>0].reset_index(drop=True)
    text_info_new.rename(columns={'text_need': 'text', 'death_hosp': 'label'}, inplace=True)
    text_info_new = gu.preprocessing(text_info_new)
    # split separate note types
    text_info_new.dropna(subset=['text'], inplace=True)
    text_info_new.reset_index(drop=True, inplace=True)
    text_info_new = gu.name_the_same(text_info_new)
    text_new_sep, text_new_sep_count, text_new_merge = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    text_new_sep, text_new_merge = gu.notes_split_process(text_info_new, no_meaning_contents['content_string'].tolist())
    text_new_sep_count = pd.get_dummies(text_new_sep[['id', 'label', 'text_name']],
                                        columns=['text_name']).groupby(['id', 'label'], as_index=False).sum()
    text_new_sep_count.columns = [sub.replace('text_name_', '') for sub in text_new_sep_count.columns.to_list()]
    text_new_sep.to_csv(data_path + 'db_generation/' + 'text_' + db_name + '_older_sep_use.csv', index=False)
    text_new_sep_count.to_csv(data_path + 'db_generation/' + 'text_' + db_name + '_older_count_use.csv', index=False)
    text_new_merge.to_csv(data_path + 'db_generation/' + 'text_' + db_name + '_older_merge_use.csv', index=False)
    del text_info_new, text_new_sep

    for i in ['seq_240', 'seq_120', 'seq_60']:
        text_info_final = pd.DataFrame()
        text_info_final = gu.text_after_split(text_new_merge, seq_info[i][0], seq_info[i][1])
        text_info_final['text_len'] = text_info_final['text'].map(len)
        text_info_final = text_info_final.loc[text_info_final['text_len']>5].reset_index(drop=True) # remove data with phrases shorter than 5 words.
        text_info_final = text_info_final[['id', 'text', 'label']]

        # imputation
        data_numeric_new = pd.DataFrame()
        data_numeric_new = data_numeric[data_numeric['id'].isin(list(text_info_final['id'].unique()))].reset_index(drop=True)
        data_numeric_new = gu.data_process_older_score(
            data_numeric_new, text_new_sep_count, db_name, outlier_range_check, result_path)

        # id, label type set to int
        columns_int_names = ['id', 'label']
        data_numeric_new.rename(columns={'death_hosp': 'label'}, inplace=True)
        data_numeric_new[columns_int_names] = data_numeric_new[columns_int_names].astype(int)
        text_info_final[columns_int_names] = text_info_final[columns_int_names].astype(int)

        # check whether including discharge and hospital information
        check_info = pd.DataFrame()
        check_info = gu.check_no_error_note(text_info_final)
        # double-check the special error in eicu and confirm is right
        # conjunctivae is discharge/ purulent discharge/
        print('error num check:', max(check_info['error_num']))
        print('total error num:', check_info[check_info['error_num']>0].shape[0])
        if not os.path.exists(data_path + 'use/' + i + '/initial/'):
            os.makedirs(data_path + 'use/' + i + '/initial/')
        text_info_final.to_csv(data_path + 'use/' + i + '/initial/' + 'text_' + db_name + '_older_use.csv', index=False)
        data_numeric_new.to_csv(data_path + 'use/' + i + '/initial/' + 'numeric_' + db_name + '_older_use.csv', index=False)

        print("{} with {} of patients {}".format(db_name, i, data_numeric_new.shape[0]))



#                   STEP2. prepare different text type data by sliding window                #
#
#    -----------------------------------------------------------------------------------     #
data_path, result_path = '', ''
data_path = project_path + 'data/'
seq_info = {'seq_240': [240, 20], 'seq_120': [120, 10], 'seq_60': [60, 5]} # seq_len, slide_len
element_list = ['chief complaint', 'history of present illness', 'medications on admission',
                'past medical history', 'physical exam'] # , 'family history', 'social history'

for db_name in ['mimiciii', 'mimiciv', 'eicu']:
    text_info = pd.DataFrame()
    text_info = pd.read_csv(data_path + 'db_generation/' + 'text_' + db_name + '_older_sep_use.csv')
    text_info['text'] = text_info['text_name'] + ': ' + text_info['sub_text'] # add subtext info name

    for tt in element_list:

        for i in ['seq_240', 'seq_120', 'seq_60']:

            num_info = pd.DataFrame()
            num_info = pd.read_csv(data_path + 'use/' + i + '/initial/' + 'numeric_' + db_name + '_older_use.csv')

            text_info_each = pd.DataFrame()
            text_info_each = text_info.loc[text_info['text_name'] == tt]
            text_info_each.reset_index(drop=True, inplace=True)
            text_info_each = num_info.merge(text_info_each[['id', 'text']], on='id', how='left')
            text_info_each['text'] = text_info_each['text'].fillna(tt + ': ' + 'noncontributory.')
            text_info_final = pd.DataFrame()
            text_info_final = gu.text_after_split(text_info_each, seq_info[i][0], seq_info[i][1])
            text_info_final['text_len'] = text_info_final['text'].map(len)
            # if tt not in ['family history', 'social history']: # short records
            #     text_info_final = text_info_final.loc[text_info_final['text_len']>5].reset_index(drop=True) # remove data with phrases shorter than 5 words.
            text_info_final = text_info_final[['id', 'text', 'label']]

            # id, label type set to int
            columns_int_names = ['id', 'label']
            text_info_final[columns_int_names] = text_info_final[columns_int_names].astype(int)

            # check whether including discharge and hospital information
            check_info = pd.DataFrame()
            check_info = gu.check_no_error_note(text_info_final)
            print('error num check:', max(check_info['error_num']))
            print('total error num:', check_info[check_info['error_num']>0].shape[0])

            text_info_final.to_csv(
                data_path + 'use/' + i + '/initial/' + 'text_' + db_name + '_older_use_' + tt.replace(' ', '_') + '.csv', index=False)

            print("{} with {} of {} text with {} rows".format(db_name, i, tt, text_info_final.shape[0]))



#                   STEP3. split to the development set and temporal set                 #
#
#    -------------------------------------------------------------------------------     #
# development set: 2001~2016 of mimic
# temporal set: 2017~2019 of mimic
# external set: eicu
for i in ['seq_240', 'seq_120', 'seq_60']:
    data_initial_path = project_path + 'data/db_generation/'
    data_path = project_path + 'data/use/' + i + '/initial/'
    result_path = project_path + 'data/use/' + i + '/'
    gu.data_development_temporal_external(data_initial_path, data_path, result_path)

    data_path = project_path + 'data/use/' + i + '/'
    result_path = data_path + 'subtext/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    gu.subtext_development_temporal_external(data_path, result_path)
del data_initial_path, data_path, result_path

# get development, temporal, and external statistic informations
data_initial_path = project_path + 'data/db_generation/'
data_path = project_path + 'data/use/'
result_path = project_path + 'result/tableone/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
gu.statistics_development_temporal_external(data_initial_path, data_path, result_path)
del data_initial_path, data_path, result_path

# get the missing ratio of all sets
for i in ['seq_240', 'seq_120', 'seq_60']:
    data_initial_path = project_path + 'data/db_generation/'
    data_path = project_path + 'data/use/' + i + '/'
    result_path = project_path + 'result/tableone/'
    gu.develop_temp_ext_missing_ratio(data_initial_path, data_path, result_path)
del data_initial_path, data_path, result_path