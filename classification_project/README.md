# Telco™ - readme.md
![Telco](https://icon-library.net/images/telecommunications-icon/telecommunications-icon-3.jpg)

***

## Project Description:
- The purpose of this project is to identify various drivers of churn within Telco using machine learning methodologies.

## Goals

The goals of the project are to answer the questions and deliver the following:

- A Complete Jupyter Notebook Report showing the step-by-step process and statistical analysis of my driver(s) of customer churn.
- a README.md file containing: 
    - project description with goals
    - a data dictionary
    - project planning (lay out your process through the data science pipeline)
    - explanation of how someone else can recreate your project and findings
    - key findings and takeaways from your project.
- a CSV file with customer_id, probability of churn, and prediction of churn.
- individual modules, .py files, that hold your functions to acquire and prepare your data.
- Clearly state your starting hypotheses (and add the testing of these to your task list).

***
## Data Dictionary

---
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
|customer\_id|Alpha-numeric ID that identifies each customer| object |
online_security|Indicates if a customer has online security add-on| int64 |
online_backup|Indicates if a customer has online backups add-on| int64 |
device_protection|Indicates if a customer has a protection plan for Telco™ devices| int64 |
tech_support|Indicates whether a customer has technical support add-on| int64 |
streaming_tv|Indicates if a customer uses internet to stream tv| int64 |
streaming_movies|Indicates if a customer uses internet to stream movies| int64 |
|internet\_service\_type|Indicates the type of internet service a customer has| object |
|dsl_internet|Indicates if the customer has dsl internet or not| uint8 |
|fiber_optic_internet|Indicates if the customer has fiber optic internet or not| uint8 |
| n_services | Sum of online_security, online_backup, device_protection, tech_support, streaming_tv, and streaming_movies | int64 |

| Target | Definition | Data Type |
| ----- | ----- | ----- |
|churn|Indicates whether a customer has terminated service| object |
***
## Data 
### Acqisition
- acquire.py

| Function Name | Definition |
| ----- | ----- |
| get_telco_data(cached=False) | This function reads in telco_churn data from Codeup database and writes data to a csv file if cached == False or if cached == True reads in titanic df from a csv file, returns df. |


### Preparation
- prepare.py 
| Function Name | Definition |
| ----- | ----- |
| clean_telco() | clean_telco will take in the telco_churn data as df, replace "No service" string with "No", convert categorical variable into dummy/indicator variables, drop columns with duplicate  information as well as statistically invalid columns, normalize the column titles, rename some titles for  legibility, convert total_charges to dtype='float64' and fill null values in total_charges with 0 since the customers tenure was 0 and have yet to pay their first bill. Finally,this function converts 'Yes' to 1 and 'No' to 0. |
| train_validate_test_split(df, seed=123) | This funstion splits the data for modeling |

***
## Data Exploration:
- explore.py 
| Function Name | Definition |
| ----- | ----- |
| explore_univariate(train, cat_vars, quant_vars) | This function takes in the train df, categorical and quantitative variables and returns barplots for categorical, histplots and boxplots for quantitative variables, and descriptive stats of each variable. |
| explore_bivariate(train, target, cat_vars, quant_vars) | This function takes in the train df, the target variable (churn), categorical and quantitative variables and returns barplots for categorical and histplots and boxplots for quantitative variables where the hue indicates churn, and descriptive stats of each variable. |
| explore_bivariate_categorical(train, target, cat_var) | Takes in categorical variable and binary target variable, returns a crosstab of frequencies runs a chi-square test for the proportions and creates a barplot, adding a horizontal line of the overall rate of the target. |

### Univariate:
Established variables for quant_vars, cat_vars and target for future exploratory use.
```json
{
quant_vars = ['tenure','monthly_charges','total_charges']
cat_vars = list((df.columns).drop(quant_vars))
target = 'churn'
}
```
```json
{
explore.explore_univariate(train, cat_vars, quant_vars)
}
```
#### Observations:
- There are significantly more non-senior citizens than senior citizens
- There are a lot more customers with dependents
- Significantly more customers with phone service than without
- Less have online security, online backup, device protection, and tech support
- A lot more people churn than stay
- More customers are Month-to-month than in contracts
- Electronis check is the most popular payment method

#### Questions:
- Customers with phone service that have multiple lines?
- Customers with internet that have online services (i.e. online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies)

### Bivariate:
Initially ran the explore_bivariate function and noticed a pattern. I decided to act limit the categorical variables and re-run the function.
```json
{
cat_vars = ['online_security','online_backup','device_protection','tech_support','streaming_tv','streaming_movies']
}
```
```json
{
explore.explore_bivariate(train, target, cat_vars, quant_vars)
}
```
#### Observations:
- Variables with very low p-values (when scientific notation is used)
    - senior, partner, dependents, online_security, tech_support, paperless_billing, ***month-to_month***, fiber_optic_internet, one_year, two_year
- Low p-values (well below 0.05 but able to read without scientific notation)
    - online_backup, device_protection, streaming_tv, streaming_movies, 
- Barely passes 95% confidence (very close to alpha - 0.05)
    - multiple lines
- Does not pass
    - gender, phone_service
- Vast majority of churn happens before 30 months
- higher monthly bill increases churn

#### Questions:
- Do people with all online services churn more than customers without all of the online services?
- Are the really low p-values a good starting point?

***
## Question:

### Does the amount of online services affect churn rates of our customers with internet service?
#### Online services are:
   - online security
   - online backup
   - device protection
   - tech supprt
   - streaming tv
   - streaming movies

***   
## Answering the question:
### Cleaning train, validate and test. 
- Only need customers with internet service 
```json
{
train = train[train.internet_service_type != 'None']
validate = validate[validate.internet_service_type != 'None']
test = test[test.internet_service_type != 'None']
}
```
- Need to remove unecessary information for statistics and modeling
```json
{
dropcols = ['internet_service_type','senior_citizen','partner','dependents','tenure','phone_service','multiple_lines','paperless_billing','monthly_charges','total_charges','contract_type','payment_type','gender_male','one_year_contract','two_year_contract','credit_card_payment','e_check_payment','mailed_check_payment']
train = train.drop(columns=dropcols)
validate = validate.drop(columns=dropcols)
test = test.drop(columns=dropcols)
}
```
- Set the index to 'customer_id'
```json
{
train.set_index('customer_id')
validate.set_index('customer_id')
test.set_index('customer_id')
}
```
- Add a column that adds the number of online services that our internet customers have.
```json
{
train = train.assign(n_services = train[train.columns[1:7]].sum(axis=1))
validate = validate.assign(n_services = validate[validate.columns[1:7]].sum(axis=1))
test = test.assign(n_services = test[test.columns[1:7]].sum(axis=1))
}
```

***
## Visualize the question:
Below is the code I used to create better visualizations than the explore_bivariate function could produce.
```json
{
features = ['online_security', 'online_backup', 'device_protection']
_, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 6), sharey=True)
for i, feature in enumerate(features):
    sns.barplot(feature, 'churn', data=train, ax=ax[i])
    ax[i].set_xlabel('')
    ax[i].set_ylabel('Churn Rate')
    ax[i].set_title(feature)
    ax[i].axhline(train.churn.mean(), ls='--', color='grey')
}
```
```json
{
features = ['tech_support', 'streaming_tv', 'streaming_movies']
_, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 6), sharey=True)
for i, feature in enumerate(features):
    sns.barplot(feature, 'churn', data=train, ax=ax[i])
    ax[i].set_xlabel('')
    ax[i].set_ylabel('Churn Rate')
    ax[i].set_title(feature)
    ax[i].axhline(train.churn.mean(), ls='--', color='grey')
}
```
```json
{
feature = 'n_services'
target = 'churn'
sns.barplot(x=feature, y=target, data=train)
plt.xlabel('Number of Online Services per Customer with Internet')
plt.ylabel('Churn Rate')
plt.title("Relationship of Churn to Number of Online Services")
plt.show()
}
```

***
## Statistical Analysis

### χ<sup>2</sup> test
 - Testing for independence between 2 categorical values.
 - Churn is categorical (i.e. can be 1 or 0)
 - n_services is categorical (i.e. can be integers from 0-6)

- #### Hypothesis:
    - The **null hypothesis** = Churn is independent of the number of online services per internet customer.
    - the **alternate hypothesis** = We assume that there is an association between churn and the number of online services.

- #### Confidence level and alpha value:
    - I established a 95% confidence level
    - alpha = 1 - confidence, therefore alpha is 0.05

- #### Results:
    - χ<sup>2</sup> = 242.75
    - p-value = 1.45 x 10<sup>-49</sup>
    - degrees of freedom = 6


We reject the null hypothesis and move forward with the alternative hypothesis: We assume that there is an association between churn and the number of online services

### T-test

- I am running a T-test to verify that there is a difference between having no additional online services vs. having any extra online service

- #### Hypothesis:
    - The **null hypothesis** = There is no difference between in the means of customers without any additional online services and  customers with any number of online services.
    - the **alternate hypothesis** = There is a difference in the means of customers with online services and those without online services.

- #### Confidence level and alpha value:
    - I established a 95% confidence level
    - alpha = 1 - confidence, therefore alpha is 0.05

- #### Results:
    - t-score = -2.17
    - p-value = 0.044

***
## Modeling:

### Baseline
- Began by importing my machine learning models
```json
{
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
}
```
- Then I established a baseline:
    - First is to run a value_counts() on the column 'churn'
    - Then add a most_frequent column to the train dataframe and set all values set to the highest value_count(). (in this case there were 2122 - 0's and 983 - 1's)
    - Finally run the baseline_accuracy:
```json
{
train["most_frequent"] = 0
baseline_accuracy = (train.churn == train.most_frequent).mean()
print(f'My baseline prediction is survived = 0')
print(f'My baseline accuracy is: {baseline_accuracy:.2%}')
}
```
    - The goal here is to create a model that beats the baseline:
        - My baseline accuracy is: 68.34%
        
### Make: X_train, X_validate, X_test, y_train, y_validate, and y_test
```json 
{
X_train = train.drop(columns=['churn','most_frequent','customer_id'])
y_train = train.churn

X_validate = validate.drop(columns=['churn','customer_id'])
y_validate = validate.churn

X_test = test.drop(columns=['churn','customer_id'])
y_test = test.churn
}
```

### kNN:
```json
{
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
}
```
- #### Results:
    - Accuracy of KNN classifier on training set n_neighbors set to 5: 0.71
    - Accuracy of KNN classifier on validate set with n_neighbors set to 5: 0.70

### Random Forest:
```json
{
rf = RandomForestClassifier(bootstrap=True, 
                            class_weight=None, 
                            criterion='gini',
                            min_samples_leaf=5,
                            max_depth=50, 
                            random_state=42)
}
```
- #### Results
    - Accuracy of random forest classifier on training set: 0.73
    - Accuracy of random forest classifier on the validate set: 0.72

### Decision Tree:
```json
{
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
}
```
- #### Results
    - Accuracy of Decision Tree classifier on training set: 0.73
    - Accuracy of Decision Tree classifier on validate set: 0.72

### Logistic Regression:
```json
{
logit = LogisticRegression(penalty='l2', C=1, random_state=42, solver='lbfgs')    
}
```
- #### Results
    - Accuracy of on training set: 0.72
    - Accuracy out-of-sample set: 0.73
    - Accuracy of on test set: 0.72
    
***    
## Predictions CSV
Make a new dataframe
```json
{
    new_df = df
}
```
Limit the features to the features being tested
```json 
{
features = ['online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'dsl_internet', 'fiber_optic_internet', 'n_services']
new_df_trimmed = new_df[new_df.internet_service_type != 'None']
new_df_trimmed = new_df_trimmed.drop(columns=dropcols)
new_df_trimmed = new_df_trimmed.assign(n_services = new_df_trimmed[new_df_trimmed.columns[1:7]].sum(axis=1))
}
```
Remove customers without internet service into prediction dataframe
```json
{
prediction_df = new_df[new_df.internet_service_type != "None"] 
}
```
Run predictions on the Logistic Regression model
```json
{
prediction_df['prediction'] = logit.predict(new_df_trimmed[features])   
}
``` 

```json
{
predictions = prediction_df[['customer_id','prediction']]
predictions.to_csv('predictions.csv')
}
```

***
## Key Takaways

- It is clear that fewer people churn when they have more online services. 

- What can we do now?

    - Promote online services into bundled packages:
        - For instance: the "Security package" will contain: online_security, online_backup, device_protection and tech_support
        - Along with "Streaming package" that will contain: streaming_tv, streaming_movies, and tech_support as well.

- With additional time dedicated to this project:

    - Investigate fiber optic customers in greater detail and look at possible combinations of factors that might be driving churn within that group.
    - Investigate our pricing structure of all internet service types and online services.
    - Make improvements to this report with more comments, markdown cells, and summary tables.
    - Make improvements to the cooresponding readme.md for this github containing project description with a more in depth explanation of how someone else can recreate this project and findings, and key takeaways from this project.
    - Add models to test varying hyperparameters and features to improve model performance.