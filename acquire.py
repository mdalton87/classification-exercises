import pandas as pd
import numpy as np
import os

# visualize
import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(11, 9))
plt.rc('font', size=13)

# acquire
from env import host, user, password
from pydataset import data
import acquire

# data = sns.load_dataset('iris')
# df_iris = pd.DataFrame(data)
# 
# df_iris.head(3)
# df_iris.shape
# df_iris.columns
# df_iris.info()
# df_iris.describe()
# 
# # 2.
# 
# df_excel = pd.read_excel('/Users/matthewdalton/codeup-data-science/spreadsheets/mytable_customer_details.xlsx', sheet_name='Table1_CustDetails')
# df_excel_sample = df_excel.head(100)
# df_excel.shape[0]
# df_excel.columns[:5]
# 
# obj_lst = list(df_excel.columns) # alicia's
# 
# for obj in list(df_excel.columns):
#     if df_excel[obj].dtype == 'object':
#         print(obj)
#         
# for obj in obj_lst:
#     if df_excel[obj].dtype == 'float64':
#         print(obj,"range:", df_excel[obj].max() - df_excel[obj].min())
#         
#         
# sheet_url = 'https://docs.google.com/spreadsheets/d/1Uhtml8KY19LILuZsrDtlsHHDC9wuDGUSe8LTEwvdI5g/edit#gid=341089357'
# csv_export_url = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
# df_google = pd.read_csv(csv_export_url)
# 
# df_google.head(3)
# df_google.shape
# df_google.columns
# df_google.info()
# df_google.describe(include=[np.number])
# 
# df_google.PassengerId.unique()
# df_google.Survived.unique()
# df_google.Pclass.unique()
# df_google.Name.unique()
# df_google.Sex.unique()
# df_google.Age.unique()
# df_google.SibSp.unique()
# df_google.Parch.unique()
# df_google.Ticket.unique()
# df_google.Fare.unique()
# df_google.Cabin.unique()
# df_google.Embarked.unique()


def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'    
       

## TITANIC DATABASE        
        
def new_titanic_data():
    '''
    This function reads in the titanic data from the Codeup db
    and returns a pandas DataFrame with all columns.
    '''
    sql_query = 'SELECT * FROM passengers'
    return pd.read_sql(sql_query, get_connection('titanic_db'))


def get_titanic_data(cached=False):
    '''
    This function reads in titanic data from Codeup database and writes data to
    a csv file if cached == False or if cached == True reads in titanic df from
    a csv file, returns df.
    '''
    if cached == False or os.path.isfile('titanic_df.csv') == False:
        
        # Read fresh data from db into a DataFrame.
        df = new_titanic_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('titanic_df.csv')
        
    else:
        
        # If csv file exists or cached == True, read in data from csv.
        df = pd.read_csv('titanic_df.csv', index_col=0)
        
    return df

## IRIS DATABASE

def get_iris_data(cached=False):
    '''
    This function reads in iris data from Codeup database and writes data to
    a csv file if cached == False or if cached == True reads in titanic df from
    a csv file, returns df.
    '''
    if cached == False or os.path.isfile('iris_df.csv') == False:
        
        # Read fresh data from db into a DataFrame.
        df = sns.load_dataset('iris')
        
        # Write DataFrame to a csv file.
        df.to_csv('iris_df.csv')
        
    else:
        
        # If csv file exists or cached == True, read in data from csv.
        df = pd.read_csv('iris_df.csv', index_col=0)
        
    return df    