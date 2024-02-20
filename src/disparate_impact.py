import pandas as pd
import numpy as np


def evaluation_report(dataset: pd.DataFrame, protected_attributes: list, output_column: str) -> pd.DataFrame:
    """_summary_

    Args:
        dataset (pd.DataFrame): the dataset on which search for biases
        protected_attributes (list): the protected attributes on which compute the disparate impact computation
        output_column (str): the output column
        output_column_values (list): the possible output column values

    Returns:
        pd.DataFrame: a report that represents the attribute with the relative disparate impact value
    """

    output_column_values = dataset[output_column].unique()
    return return_disparate_impact(dataset, protected_attributes, output_column, output_column_values)


def return_disparate_impact(dataset: pd.DataFrame, protected_attributes: list,
                            output_column: str, output_column_values: list) -> pd.DataFrame:
    """
    This method returns the disparate impact for the protected attributes inì the specified dataset
    Args:
        output_column: the output column on which compute the disparate impact value
        protected_attributes: the list of protected attributes
        dataset: the working dataset
        output_column_values: the values for the output column
    """
    attribute_series = pd.Series(protected_attributes)
    attributes = []
    disparate_impact_array = []
    for output_value in output_column_values:
        for attribute in protected_attributes:
            unprivileged_probability = compute_disparate_impact(dataset, attribute, 0,
                                                                output_column, output_value)

            privileged_probability = compute_disparate_impact(dataset, attribute, 1,
                                                              output_column, output_value)

            if privileged_probability == 0:
                continue
            else:
                disparate_impact = unprivileged_probability / privileged_probability
                attributes.append(attribute + ', y=' + str(output_value))
                disparate_impact_array.append(disparate_impact)

    attribute_series = pd.Series(attributes)
    disparate_impact_series = pd.Series(np.array(disparate_impact_array))
    disparate_impact_dataframe = pd.DataFrame(
        {"Attribute": attribute_series, "Disparate Impact": disparate_impact_series})

    return disparate_impact_dataframe


def compute_disparate_impact(dataset: pd.DataFrame, protected_attribute, protected_attribute_value,
                             output_column, output_value) -> float:
    """
    This method computes the disparate impact value starting from the parameters
    :param dataset: the dataset needed to perform the computation
    :param protected_attribute: the protected attribute on which compute the disparate impact
    :param protected_attribute_value: the value of the protected attribute
    :param output_column: the output of interest
    :param output_value: the value of the output of interest
    :return:
    """
    attribute_columns_data = dataset[dataset[protected_attribute] == protected_attribute_value]
    return len(attribute_columns_data[attribute_columns_data[output_column] == output_value]) / len(
        attribute_columns_data)
