# Project Description and Goals:
- The purpose of this project is to identify various drivers of churn within Telco.
- My goal is to identify 

# The Plan:
- 
-
-

# Data Dictionary

## Data acqisition
- connect to SQL server using env file 
- wrote SQL query to acquire the correct and complete database, telco_churn.

## Data Cleaning
- replace "No service" string with "No"
- convert categorical variables into dummy/indicator variables
    - gender, contract_type, internet_service_type, and payment_type
- drop columns with duplicate information as well as statistically invalid columns
    - gender, payment_type_id, internet_service_type_id, contract_type_id, customer_id, internet_service_type_none
- normalize and rename the column titles for ease-of-use and legibility
- convert total_charges to dtype='float64' and fill null values in total_charges with 0
    - since the customers tenure was 0 and have yet to pay their first bill. 
- converts the srings 'Yes' and 'No to 1 and 0, respectively.
  
## Takeaways from exploration:
### Univariate:
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
#### Observations:
- REALLY LOW p-values
    - senior, partner, dependents, online_security, tech_support, paperless_billing, ***month-to_month***, fiber_optic_internet, one_year, two_year
- Real low p-values
    - online_backup, device_protection, streaming_tv, streaming_movies, 
- barely passes 95% confidence
    - multiple lines
- Not low
    - gender, phone_service
- Vast majority of churn happens before 30 months
- higher monthly bill increases churn

#### Questions:
- Do people with all online services churn more than customers without all of the online services?
- Are the really low p-values a good starting point?