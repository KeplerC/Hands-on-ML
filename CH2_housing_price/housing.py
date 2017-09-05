#!/usr/bin/python
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

#_____________________Setups________________________________
# Common imports
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

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
    
#_____________________________Lab Starts_________________________
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelBinarizer

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

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
    
def pipeline(data):
    num_attribs = list(data)
    cat_attribs = ["ocean_proximity"]
    num_pipeline = Pipeline([('imputer', Imputer(strategy="median")),('std_scaler', StandardScaler()),])
    cat_pipeline = Pipeline([('selector', DataFrameSelector(cat_attribs)),('label_binarizer', LabelBinarizer()),])
    full_pipeline = FeatureUnion(transformer_list=[("num_pipeline", num_pipeline),("cat_pipeline", cat_pipeline),])
    return num_pipeline.fit_transform(data)

from sklearn.metrics import mean_squared_error
def error(model, test_set, labels):
    predictions = lin_reg.predict(test_set)
    lin_mse = mean_squared_error(labels, predictions)
    return np.sqrt(lin_mse)

def main():
    housing = load_housing_data()
    #give_a_quick_view(housing)
    train_set, test_set = generating_test_set(housing)
    #visualization(train_set)
    #print(correlation(housing, "median_house_value"))
    data_cleaning(housing)
    #housing = pipeline(housing)
    lin_reg = LinearRegression()
    #lin_reg.fit(housing, housing["median_house_value"])
    
if __name__ == "__main__":
    main()
