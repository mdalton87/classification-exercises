import pandas as pd
import numpy as np
import os

# acquire
from env import host, user, password
from pydataset import data
from acquire import get_connection, new_iris_data, get_iris_data

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer



## Clean IRIS DATABASE ##

def clean_iris():
    '''
    clean_iris will take a dataframe acquired as df and remove columns:
        species_id: species_name has same info but more descriptive
        measurement_id: redundant to the index, so no statistical value
    rename species_name to species,
    and add dummy values for the species
    return: single cleaned dataframe
    '''
    df = get_iris_data()
    df['species'] = df.species_name
    dropcols = ['species_id', 'measurement_id', 'species_name']
    df.drop(columns=dropcols, inplace=True)
    dummies = pd.get_dummies(df[['species']], drop_first=True)
    return pd.concat([df, dummies], axis=1)

### Machine Learning ###

df = clean_iris()

train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.species)
train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train.species)

def impute_mode():
    '''
    impute mode for species
    '''
    imputer = SimpleImputer(strategy='most_frequent')
    train[['species']] = imputer.fit_transform(train[['species']])
    validate[['species']] = imputer.transform(validate[['species']])
    test[['species']] = imputer.transform(test[['species']])
    return train, validate, test


def prep_iris_data():
    df = clean_iris()
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.species)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.species)
    train, validate, test = impute_mode()
    return train, validate, test