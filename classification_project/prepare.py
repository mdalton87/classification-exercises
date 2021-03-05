import pandas as pd
import numpy as np
import os

# acquire
from env import host, user, password
from pydataset import data
from acquire import get_connection, new_telco_data, get_telco_data

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer



## Clean IRIS DATABASE ##

def clean_telco():
    '''
    clean_telco will take in the telco_churn data as df, replace "No service" string with "No", 
    convert categorical variable into dummy/indicator variables, drop columns with duplicate information as 
    well as statistically invalid columns, normalize the column titles, and finally rename some titles for legibility
    '''
    df = get_telco_data()
    df.replace('No phone service', 'No', inplace=True)
    df.replace('No internet service', 'No', inplace=True)
    dummy_df = pd.get_dummies(df[['gender']], drop_first=True)
    dummy2_df = pd.get_dummies(df[['contract_type','internet_service_type','payment_type']], drop_first=False)
    dropcols = [
            'payment_type_id',
            'internet_service_type_id',
            'contract_type_id',
            'customer_id',
            'contract_type',
            'internet_service_type',
            'payment_type',
            'internet_service_type_none'
               ]
    df = pd.concat([df, dummy_df,dummy2_df], axis=1)
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('(','').str.replace(')','')
    df.drop(columns=dropcols, inplace=True)
    df.columns = [
        'gender','senior_citizen','partner','dependents','tenure','phone_service','multiple_lines',
        'online_security','online_backup','device_protection','tech_support','streaming_tv','streaming_movies',
        'paperless_billing','monthly_charges','total_charges','churn','gender_male','month-to-month_contract', 
        'one_year_contract','two_year_contract', 'dsl_internet','fiber_optic_internet','payment_type_bank_transfer',
        'payment_type_credit_card', 'payment_type_e_check','payment_type_mailed_check'
                ]
    return df


def impute_mode():
    '''
    impute mode for churn
    '''
    imputer = SimpleImputer(strategy='most_frequent')
    train[['churn']] = imputer.fit_transform(train[['churn']])
    validate[['churn']] = imputer.transform(validate[['churn']])
    test[['churn']] = imputer.transform(test[['churn']])
    return train, validate, test



def train_validate_test_split(df, seed=123):
    train_and_validate, test = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df.churn
    )
    train, validate = train_test_split(
        train_and_validate,
        test_size=0.3,
        random_state=seed,
        stratify=train_and_validate.churn,
    )
    return train, validate, test



def prep_telco_data():
    df = clean_telco()
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.churn)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.churn)
    train, validate, test = impute_mode()
    return train, validate, test




