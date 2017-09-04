#!/usr/bin/python
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

#_____________________Setups________________________________
# Common imports
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def plot(data):
    data.hist(bins=50, figsize=(20,15))
    plt.show()
    
#_________________________________Lab Starts_________________________
import os
import pandas as pd
HOUSING_PATH = ""

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def give_a_quick_view(data):
    print ("__________housing's head____________")
    print (data.head())
    print ("_________ table information________")
    print (data.info())
    print ("__________Data's descrpition_______")
    print (data.describe())

    #data.hist(bins=50, figsuze=(20, 15))
    #plt.show()

def generating_test_set(data):
    train_set, test_set = train_test_split(data)
    return train_set, test_set

def visualization(housing):
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,s=housing["population"]/100, label="population",c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
    plt.legend()
    plt.show()

def correlation(data, label):
    return data.corr()[label].sort_values(ascending=False)

def data_cleaning(data):
    data_num =data.drop("ocean_proximity", axis =1) #because this is not a median feature
    imputer = Imputer(strategy="median")
    imputer.fit(data_num)
    X = imputer.transform(data_num)
    return pd.DataFrame(X, columns = data_num.columns)
    
def main():
    housing = load_housing_data()
    #give_a_quick_view(housing)
    train_set, test_set = generating_test_set(housing)
    #visualization(train_set)
    print(correlation(housing, "median_house_value"))
    
if __name__ == "__main__":
    main()
