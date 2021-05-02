import pandas as pd
import numpy as np
import os

# acquire
from env import host, user, password



def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'    
       

# Telco Database        
        
def new_telco_data():
    '''
    This function reads in the tecl_churn data from the Codeup db
    and returns a pandas DataFrame with all columns.
    '''
    sql_query = '''
    SELECT *
    FROM customers
    JOIN contract_types USING(`contract_type_id`)
    JOIN internet_service_types USING(`internet_service_type_id`)
    JOIN payment_types USING(payment_type_id);
    '''
    
    return pd.read_sql(sql_query, get_connection('telco_churn'))
        
        
        
def get_telco_data(cached=False):
    '''
    This function reads in telco_churn data from Codeup database and writes data to a csv file if cached == False or if cached == True reads in titanic df from a csv file, returns df.
    '''
    if cached == False or os.path.isfile('telco_churn.csv') == False:
        
        # Read fresh data from db into a DataFrame.
        df = new_telco_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('telco_churn.csv')
        
    else:
        
        # If csv file exists or cached == True, read in data from csv.
        df = pd.read_csv('telco_churn.csv', index_col=0)
        
    return df
