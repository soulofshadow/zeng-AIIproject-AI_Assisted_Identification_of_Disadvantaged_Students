
import pandas as pd
import itertools
import random
import numpy as np

class MakeFair():
    def __init__(self, target, sensitive):
        self.target = target
        self.sensitive = sensitive

    def binary_target(self, x):

        if x > self.mean_value:
            return 1
        else:
            return 0

    def balance(self, df):

        #fix_protected_attributes
        df = self.fix_protected_attributes(df, self.sensitive)

        #target binary
        self.mean_value = df[self.target].mean()
        self.max_value = df[self.target].max()
        self.min_value = df[self.target].min()

        df = df.assign(
            score_binary = df[self.target].map(self.binary_target)
        )

        #combination_attributes
        attr_freq = {}
        for protected in self.sensitive:
            combination_attributes = []
            combination_attributes.append(protected)
            combination_attributes.append('score_binary')
            combination_list = list(map(list, itertools.product([0, 1], repeat=len(combination_attributes))))

            combination_frequency = []
        
            for combination in combination_list:
                combination_frequency.append(len(df.query(self.return_query_for_dataframe(combination, combination_attributes))))
            
            attr_freq[protected] = (combination_list, combination_attributes, combination_frequency)

        #sample balance dataset
        print('Balancing the sensitive attributes...')
        df_new = pd.DataFrame()
        
        for protected in self.sensitive:
            combination_list, combination_attributes, combination_frequency = attr_freq[protected]
            combination_frequency_target = max(combination_frequency)

            #Display
            print(f'View Progress: {self.sensitive.index(protected) + 1} of {len(self.sensitive)}')
            print('Sensitive attribute:', protected)
            print('Combination frequency:', combination_frequency)
            print('Combination frequency target:', combination_frequency_target)
            print('\n')

            for index in range(0, len(combination_list)):
                if combination_frequency[index] == combination_frequency_target:
                    continue
                else:
                    combination = combination_list[index]
                    temp_dataset = df.query(self.return_query_for_dataframe(combination, combination_attributes))

                    adds_needed = combination_frequency_target - combination_frequency[index]
                    added = 0

                    # while added < adds_needed:
                        #1. original method : too slow
                        # new_row = {}
                        # for (attr, value) in zip(combination_attributes, combination):
                        #     new_row[attr] = value
                        # for attribute in df.columns:
                        #     if attribute not in combination_attributes:
                        #         if self.is_variable_discrete(temp_dataset, attribute):
                        #             new_row[attribute] = random.randint(temp_dataset[attribute].min(), temp_dataset[attribute].max())
                        #         else:
                        #             new_row[attribute] = random.uniform(temp_dataset[attribute].min(), temp_dataset[attribute].max())
                    
                        # df.loc[len(df)] = new_row

                        #2. new method
                        # if combination_frequency_target - combination_frequency[index] >= combination_frequency[index]:
                        #     frac = 1
                        # else:
                        #     frac = 0.5
                        # sampled_df = temp_dataset.sample(frac=frac)
                        # for column in sampled_df.columns:
                        #     if column not in combination_attributes:
                        #         if temp_dataset[column].min() == temp_dataset[column].max():
                        #             continue
                        #         if self.is_variable_discrete(sampled_df, column):
                        #             sampled_df[column] = np.random.randint(low=temp_dataset[column].min(), high=temp_dataset[column].max(), size=len(sampled_df))
                        #         else:
                        #             sampled_df[column] = np.random.uniform(low=temp_dataset[column].min(), high=temp_dataset[column].max(), size=len(sampled_df))
                        
                        # if len(sampled_df) < adds_needed - added:
                        #     df_new = pd.concat([df_new, sampled_df]).reset_index(drop=True)
                        #     added += len(sampled_df)
                        # else:
                        #     df_new = pd.concat([df_new, sampled_df.head(adds_needed - added)]).reset_index(drop=True)
                        #     added = adds_needed

                    #3. directly random
                    sample_data = temp_dataset.iloc[0:1] #take one row
                    repeated_df = pd.concat([sample_data]*(adds_needed), ignore_index=True)
                    for column in repeated_df.columns:
                        if column not in combination_attributes:
                            if temp_dataset[column].min() == temp_dataset[column].max():
                                continue
                            if self.is_variable_discrete(repeated_df, column):
                                repeated_df[column] = np.random.randint(low=temp_dataset[column].min(), high=temp_dataset[column].max(), size=len(repeated_df))
                            else:
                                repeated_df[column] = np.random.uniform(low=temp_dataset[column].min(), high=temp_dataset[column].max(), size=len(repeated_df))
                    
                    df_new = pd.concat([df_new, repeated_df]).reset_index(drop=True)
        
        df = pd.concat([df, df_new]).reset_index(drop=True)               
        return df


    def return_query_for_dataframe(self, combination_list: list, combination_attributes: list) -> str:
        query_str = ""
        for (value, attr) in zip(combination_list, combination_attributes):
            query_str += str(attr) + " == " + str(value) + " and "
            
        return query_str[:-5]

    def fix_protected_attributes(self, dataset: pd.DataFrame, protected_attributes: list) -> pd.DataFrame:
        """
        This method fixes the protected attributes in the dataset converting them into binary ones
        :param dataset: the dataset on which the protected attributes have to be fixed
        :param protected_attributes: the protected attributes that have to be fixed
        :return: returns the dataset with the protected attributes fixed
        """
        for protected_attribute in protected_attributes:
            protected_attribute_value = []
            if len(dataset[protected_attribute].values) == 2:
                max_value = dataset[protected_attribute].max()
                for index, row in dataset.iterrows():
                    if row[protected_attribute] == max_value:
                        protected_attribute_value.append(1)
                    else:
                        protected_attribute_value.append(0)
            else:
                threshold = (dataset[protected_attribute].max() + dataset[protected_attribute].min()) / 2
                for index, row in dataset.iterrows():
                    if row[protected_attribute] > threshold:
                        protected_attribute_value.append(1)
                    else:
                        protected_attribute_value.append(0)

            dataset[protected_attribute] = pd.Series(protected_attribute_value)

        return dataset
    
    def is_variable_continuous(self, dataset: pd.DataFrame, attribute: str) -> bool:
        """
            This method checks if the attribute is continuous
            Args:
                dataset: the working dataset
                attribute: the attribute to check

            Returns:
                True if the variable is continuous, False otherwise
            """
        for attr in dataset[attribute]:
            if isinstance(attr, float):
                continue
            else:
                return False
        return True

    def is_variable_discrete(self, dataset: pd.DataFrame, attribute: str) -> bool:
        """
        This method checks if the attribute is discrete
        Args:
            dataset: the working dataset
            attribute: the attribute to check

        Returns:
            True if the variable is discrete, False otherwise
        """
        for attr in dataset[attribute]:
            if isinstance(attr, int):
                continue
            else:
                return False
        return True

