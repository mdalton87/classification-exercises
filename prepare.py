import pandas as pd
import numpy as np
import os

# acquire
from env import host, user, password
from pydataset import data
from acquire import get_connection, new_iris_data, get_iris_data


## Prepare IRIS DATABASE ##

def prep_iris():
    '''
    prep_iris will take a dataframe acquired as df and remove columns:
        species_id: species_name has same info but more descriptive
        measurement_id: contains no statistical value
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