
import pandas as pd
import numpy as np


# load csv content
def load_csv(path: str):
    return pd.read_csv(path)

# create dataset with single variable
def create_single_variable_dataset(pd_dataframe, variable, label, csv_path):
    labels = pd_dataframe[label].tolist()
    variables = pd_dataframe[variable].tolist()

    data = {variable: variables, label: labels}
    new_dataframe = pd.DataFrame(data, columns=[variable, label])
    
    new_dataframe.to_csv(csv_path)
    

if __name__ == "__main__":
    training_set = load_csv("dataset/houston_housing/initial_dataset/train.csv")
    testing_set = load_csv("dataset/houston_housing/initial_dataset/test.csv")

    create_single_variable_dataset(training_set, "GrLivArea", "SalePrice", "dataset/houston_housing/single_variable_dataset/train.csv")
