# score to level mapping
'''
<400 -> 1
>= 400 & < 500 -> 2
>= 500 & < 600 -> 3
>= 600 -> 4
'''

import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns

import sys
import random
import time
sys.setrecursionlimit(100000) #Increase the recursion limit of the OS
from sklearn.impute import SimpleImputer
from impyute.imputation.cs import fast_knn, mice
from hyperimpute.plugins.imputers import Imputers

class DataLoader():
    def __init__(self, dataset_path, dataset_dict_path):

        #Read datadict
        datadict = pd.read_excel(dataset_dict_path, sheet_name='Data Dictionary', header=2)
        datadict = datadict.iloc[20:561]

        self.all_features = datadict['Variable name'].tolist()

        important_features_df = datadict[datadict['Important\xa0feature'] == 'Yes']
        self.important_features = important_features_df['Variable name'].tolist()

        sensitive_features_df = datadict[datadict['Sensitive attribute'] == 'Yes']
        self.sensitive_features = sensitive_features_df['Variable name'].tolist()

        categorical_features_df = datadict[datadict['Type'] == 'Categorical']
        self.categorical_features = categorical_features_df['Variable name'].tolist()
        
        continues_features_df = datadict[datadict['Type'] == 'Continuous']
        self.continues_features = continues_features_df['Variable name'].tolist()

        self.non_ordered_categorical_features = ['a1', 'a3a', 'a3b', 'living_with_father_mother', 'a3c', 'a3d', 'a3et', 'a3f', 'a24',
                                                 'country_iso_cnac', 'country_iso_nac', 'd1', 'd30a', 'd30b', 'd30c', 'd30d', 'd30e', 'd30f', 'd32a', 'd33a',
                                                 'd301', 'd302', 'd303', 'd304', 'd305', 'd306', 'd307', 'd308', 'island', 'capital_island', 'public_private', 
                                                 'f0', 'f4a', 'f4b', 'f5a', 'f5b', 'f5n', 'inmigrant', 'inmigrant2', 'inmigrant_second_gen', 'f7', 'f24a', 'f24b', 'f31', 'single_parent_household',
                                                 'f33a', 'f33b', 'f33c', 'f33d', 'f33e', 'f33f', 'f33g', 'f33h',
                                                 'p2', 'p5', 'p9a', 'p9b', 'p9c', 'p9d', 'p9e', 'p9f', 'p15a', 'p15b', 'p15c', 'p15d', 'p15e', 'p15f', 'p15g', 'p15h', 'p15i',
                                                 'p18a', 'p18b', 'p18c', 'p18d', 'p18e', 'p18f', 'p18g', 'p18h', 'p18i', 'p25', 'p26', 'p141', 'pfc', 'rep']
        self.cor_features = ['a3a', 'a3b', 'a4', 'island', 'f3a', 'f3b', 'f5n', 'f8ta', 'f8tm', 'f11', 'f24a', 'f24b', 'f31', 'f34',
                             'inmigrant2', 'inmigrant_second_gen', 'country_iso_cnac', 'country_iso_nac']
        self.label_feature_score = ['score_MAT', 'score_LEN', 'score_ING']
        self.label_feature_level = ['level_MAT', 'level_LEN', 'level_ING']

        #Read dataset
        dataset = pd.read_csv(dataset_path, low_memory=False)
        self.datasets = dataset

        #params
        self.labels = None
        self.features = None
        self.sensitives = None

        #split to different grades and years
        self.dataset_3_16 = dataset[(dataset['id_grade'] == 3) & (dataset['id_year'] == 2016)].reset_index(drop=True)
        self.dataset_6_16 = dataset[(dataset['id_grade'] == 6) & (dataset['id_year'] == 2016)].reset_index(drop=True)

        self.dataset_3_17 = dataset[(dataset['id_grade'] == 3) & (dataset['id_year'] == 2017)].reset_index(drop=True)
        self.dataset_6_17 = dataset[(dataset['id_grade'] == 6) & (dataset['id_year'] == 2017)].reset_index(drop=True)
        self.dataset_4_17 = dataset[(dataset['id_grade'] == 4) & (dataset['id_year'] == 2017)].reset_index(drop=True)

        self.dataset_3_18 = dataset[(dataset['id_grade'] == 3) & (dataset['id_year'] == 2018)].reset_index(drop=True)
        self.dataset_6_18 = dataset[(dataset['id_grade'] == 6) & (dataset['id_year'] == 2018)].reset_index(drop=True)
        self.dataset_4_18 = dataset[(dataset['id_grade'] == 4) & (dataset['id_year'] == 2018)].reset_index(drop=True)

        self.dataset_6_19 = dataset[(dataset['id_grade'] == 6) & (dataset['id_year'] == 2019)].reset_index(drop=True)
        self.dataset_4_19 = dataset[(dataset['id_grade'] == 4) & (dataset['id_year'] == 2019)].reset_index(drop=True)

        #get past and future
        self.predict_for_future = dataset[~dataset['id_student_16_19'].isna()]
        self.predict_for_future = self.get_predict_future()
    
    def get_predict_future(self):

        dataset = self.predict_for_future
        past = dataset[dataset['id_grade'] == 3]
        future = dataset[(dataset['id_grade'] == 6) & (dataset['id_year'] == 2019)]

        merge_column = ['id_student_16_19'] + self.label_feature_score + self.label_feature_level
        merged_df = pd.merge(past, future[merge_column], on='id_student_16_19', how='left', suffixes=('', '_future'))

        return merged_df
        
    #******************
    #Pre-processing
    #******************

    #1.SPLIT
    def get_split(self, dataset, is_level=False, feature_filter=False):

        #Labels
        if is_level:
            if dataset['id_grade'][0] == 3:
                labels = [a for a in self.label_feature_level if 'ING' not in a]
            else:
                labels = self.label_feature_level
        else:
            if dataset['id_grade'][0] == 3:
                labels = [a for a in self.label_feature_score if 'ING' not in a]
            else:
                labels = self.label_feature_score
        
        flag_future = False
        for column in dataset.columns:
            if '_future' in column:
                flag_future = True
                break;
        
        if flag_future:
            if is_level:
                labels = ['level_MAT_future', 'level_LEN_future', 'level_ING_future']
            else:
                labels = ['score_MAT_future', 'score_LEN_future', 'score_ING_future']

        #Features
        if feature_filter:
            feature_columns_to_keep = self.important_features
        else:
            feature_columns_to_keep = self.all_features
        feature_columns_to_keep = [column for column in feature_columns_to_keep if column not in self.cor_features]
        feature_columns_to_keep = [column for column in feature_columns_to_keep if column not in self.label_feature_score]
        feature_columns_to_keep = [column for column in feature_columns_to_keep if column not in self.label_feature_level]

        features = feature_columns_to_keep
        sensitives = [column for column in features if column in self.sensitive_features]
        ####
        self.labels = labels
        self.features = features
        self.sensitives = sensitives

        #Combine features with each label
        data_selected = dataset[features]
        all_datasets = []
        for label in labels:
            dataset_one = pd.concat([dataset[label], data_selected.reset_index(drop=True)], axis=1)
            dataset_one = dataset_one.dropna(subset=[label])
            all_datasets.append((label, dataset_one.reset_index(drop=True)))

        return all_datasets


    #2.IMPUTATION
    def get_imputated(self, dataset_origin, strategy='default'):
        #
        dataset = dataset_origin.copy()
        
        #Pre drop
        drop_column = []
        for col in dataset.columns:
            rate= dataset[col].isnull().sum() / float(dataset.shape[0])
            if rate > 0.5:
                drop_column.append(col)
        dataset = dataset.drop(drop_column, axis=1)

        #Pre imputation
        columns = dataset.columns
        #f23
        if 'f23' in columns:
            dataset.loc[dataset['f23'] == 9, 'f23'] = np.nan
        #f34
        if 'f34' in columns:
            dataset.loc[dataset['f34'] == 10, 'f34'] = np.nan
        #ESCS
        if 'ESCS' in columns:
            mean_escs = dataset['ESCS'].mean()
            dataset['ESCS'] = dataset['ESCS'].map(lambda x: 0 if x > mean_escs else 1)
        #OBJ data type
        #single_parent_household
        if 'single_parent_household' in columns:
            single_parent_household = {'Single parent household': 1, 'Non single parent household': 2}
            dataset['single_parent_household'] = dataset['single_parent_household'].map(single_parent_household).astype(float)
        
        #Impyute
        columns_with_nan = dataset.columns[dataset.isna().any()].tolist()

        if strategy == 'default':
            #no imputation
            pass
        if strategy == 'simple_most_frequent':
            imputer = SimpleImputer(strategy='most_frequent')
            imputer.fit(dataset[columns_with_nan])
            dataset[columns_with_nan] = imputer.transform(dataset[columns_with_nan])
        if strategy == 'knn':
            imputed = fast_knn(dataset[columns_with_nan].values.astype(float), k=30)
            dataset[columns_with_nan] = imputed
        if strategy == 'mice':
            imputed = mice(dataset[columns_with_nan].values)
            dataset[columns_with_nan] = imputed
        if strategy == 'miwae':
            method = "miwae"
            plugin = Imputers().get(method)
            output = plugin.fit_transform(dataset[columns_with_nan].copy())
            dataset[columns_with_nan] = output.values

        return dataset
    
    #3.ENCODE & SCALE (if using numerically sensitive models)
    def get_encoded(self, dataset):
  
        columns = dataset.columns
        scaler = MinMaxScaler()

        non_order_categorical_features = [column for column in columns if column in self.non_ordered_categorical_features]
        other_features = [column for column in columns[1:] if column not in non_order_categorical_features]

        dataset = pd.get_dummies(dataset, columns=non_order_categorical_features)
        dataset[other_features] = scaler.fit_transform(dataset[other_features])
        for column in dataset.columns[1:]:
            if column not in other_features:
                dataset[column] = dataset[column].astype('int')

        return dataset
    
    #4.SAMPLING
    # def get_sampled(self, dataset):
    #     X = dataset.drop(columns=self.labels)
    #     y = dataset[self.labels]

    #     seed_value = int(time.time())
    #     sampler = SMOTEENN(random_state=seed_value)

    #     all_pairs = []
    #     for label in self.labels:
    #         X_resampled, y_resampled = sampler.fit_resample(X, y[label])
    #         all_pairs.append((X_resampled, y_resampled))


    def plot_score_distribution(self, data):
        fig, axes = plt.subplots(nrows=1, ncols=data.shape[1], figsize=(12, 5))
        for i, column in enumerate(data.columns):
            sns.histplot(data[column], kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {column}')

        plt.tight_layout()
        plt.show()



