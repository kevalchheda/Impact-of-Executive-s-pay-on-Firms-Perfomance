#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[88]:


# Reading Data file
df = pd.read_csv(r"C:\Users\keval\Desktop\Python\Data Research Lab\lab_assignment.csv")


# In[529]:


#Dimension of the dataframe
print("The Dimension of the Dataframe:" , df.shape)

#Summary of the Dataframe
df.info()


# In[ ]:


#To know the number of unique TICKER Entries
df.TICKER.nunique()


# In[ ]:


# To know if the Salary column have any null values
df.SALARY.isnull().any()


# In[598]:


df


# In[620]:


modeldf = df
type(modeldf)
modeldf = modeldf[["EXECRANKANN", "SALARY", "TDC1"]]
modeldf = modeldf.dropna()
modeldf.reset_index()


# In[621]:


# PLotting Salary vs EXECRANKANN to determine the relationship between the 2 variables
sns.scatterplot(data = modeldf, x = "TDC1", y = "SALARY")
plt.show()

# It is evident from the scatter plot that there is no relation betwwen Salary and Total Compensation of the Executive.

# PLotting Salary vs EXECRANKANN to determine the relationship between the 2 variables
sns.scatterplot(data = modeldf, x = "EXECRANKANN", y = "SALARY")
plt.show()

# We can fit a curve line in the scatter plot to identify the trend between Salary and Executive Rank
# It can be seen that with increase in executive rank the Salary tends to decrease

# As the range of data is large, we get many data points which may decrease the model accuracy. 
# Hence we take the average of Salary w.r.t Executive rank in order to get a clear overview of the relationship between the variables.

# As there are many data points, we will take the average salary based on the executive rank. 
# Using Pivot_Table to get the mean salary in order of Executive Rank. 
df10 = df.pivot_table(index = "EXECRANKANN", values = ["SALARY"])
sns.scatterplot(data = df10, x = "EXECRANKANN", y = "SALARY")
plt.show()

df10 = df10.reset_index()
df10.columns = ["EXECRANKANN", "SALARY"]


# In[ ]:


# Adding Number of Years Worked Column in the dummy dataframe
df4 = df.fillna(0)
for i in range(0,len(df4.CO_PER_ROL)):
    filt = df4.loc[:,"CO_PER_ROL"] == df4.loc[i,"CO_PER_ROL"]
    df5 = df4[filt]
    df4.loc[i,"Num_of_years_worked"] = df5.loc[i,"YEAR"] - df5["YEAR"].min()


# In[ ]:


# Plotting Salary vs Num_of_years_worked
plt.scatter(x = df4.SALARY, y = df4.Num_of_years_worked)
# We can clearly identify that there is no relationship between Salary of the Employee and the Number of Years it has been associated with the company


# In[ ]:


# Variables Considered to Build the model

# Target Variable - Salary
# Independent Variable - EXECRANKANN 

# Assumptions- 
# 1. Executive's Gender has no relationship with Executive's Salary

# Reason for non - consideration of other variables/columns
# EXEC_FULLNAME - Name of the Executive does not have any impact on the Salary.
# CO_PER_ROL - Unique ID for the Company-Executive combination have no impact on Salary.
# CONAME - Company Name has no impact on Salary 
# AGE - Age of the Executive has no correlation with variable Salary
# TDC1 - TDC1 = Salary + variable incentives. Thus, it does not have any impact on Salary. It is further justified by the scatter plot
# TDC1_PCT - Change in TDC1 does not impact Salary 
# GVKEY - Unique Company Id has no impact on Salary
# EXECID - Unique Executive Id has no impact on Salary
# YEAR - Year has no impact on Salary
# GENDER - It is assumed that Gender has no impact on Salary
# TICKER - Ticker represents a company, hence it has no impact on the variable Salary. 


# In[622]:



# Using OLS regression model regression to describe the relationship between the independent variable (Executive Rank) and dependent variable (Salary).
x = df10["EXECRANKANN"]
y = df10.SALARY
x = sm.add_constant(x)
model = sm.OLS(y,x)
results = model.fit()
print(results.params)
print(results.summary())

# R2 Value of the model is 0.839 which states that 83.9% of independent variable is explained when there's a change in dependent variable


# In[5]:


# Writing this for loop to get tickers which have "." in their ticker name.
for tick in df.TICKER :
    for letter in tick:
        if letter == ".":
            print(tick)

# Changing ticker names
# Changing BRK.B to BRK-B
filt = df[df.TICKER == "BRK.B"]
ind = list(filt.index)
print(ind)
df.loc[ind, "TICKER"] = "BRK-B"
df.iloc[1616, :]


# Changing BRK.B to BRK-B
filt = df[df.TICKER == "BF.B"]
ind = list(filt.index)
print(ind)
df.loc[ind, "TICKER"] = "BF-B"

all_tickers = list(df.TICKER.unique())


# In[28]:


#!pip install yfinance 
get_ipython().system('pip install pandas-datareader')
import yfinance as yf
from pandas_datareader import data as pdr


# In[115]:


# For loop to pair each row in the data with the company’s corresponding high stock price during the specified year. 
a = 0
for year in range(2020,2021):
    for tick in all_tickers:
        try:
            data = pdr.get_data_yahoo(f"{tick}", start = f"{year}-01-01", end = f"{year}-12-30")
        except:
            a += 1
        else:
            max_price = data.High.max()
            comp = df[df.TICKER == tick]
            comp = comp[comp.YEAR == year]
            ind = list(comp.index)
            df.loc[ind,"Stock_High_Price"] = max_price
        


# In[143]:


# Creating dummy Variable
final_df = df


# In[522]:


# Rounding Stock Price
final_df.Stock_High_Price = round(final_df.Stock_High_Price, 2)


# In[283]:


# Plotting High Stock Price vs Total Compensation
sns.scatterplot(data = df, x = "Stock_High_Price", y = "TDC1")


# In[ ]:


# using Normalization technique and adding new columns to the data frame
df[[ "Norm_TDC1", "Norm_Comp_without_Salary", "Norm_Stock_Price"]] = 0
for tick in all_tickers:
    new_data = df[df["TICKER"] == tick]["TDC1"]
    ind = list(new_data.index)
    df.loc[ind,"Norm_TDC1"] = (new_data-new_data.mean())/new_data.std()
    
    new_data = df[df["TICKER"] == tick]["Comp_without_Salary"]
    ind = list(new_data.index)
    df.loc[ind,"Norm_Comp_without_Salary"] = (new_data-new_data.mean())/new_data.std()
    
    new_data = df[df["TICKER"] == tick]["Stock_High_Price"]
    ind = list(new_data.index)
    df.loc[ind,"Norm_Stock_Price"] = (new_data-new_data.mean())/new_data.std()


# In[414]:


# Plotting stock Price vs TDC1 without Salary (TDC1 - Salary)
sns.scatterplot(data = df, x = "Norm_Stock_Price", y = "Norm_Comp_without_Salary")


# In[507]:


normalized_data = df[["Norm_Comp_without_Salary", "Norm_Stock_Price"]].dropna()
x = normalized_data["Norm_Comp_without_Salary"].values.reshape(-1,1)
y = normalized_data.Norm_Stock_Price.values.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

from sklearn.neighbors import KNeighborsRegressor
classifier = KNeighborsRegressor(n_neighbors = 1)
classifier.fit(X_train, y_train)
pred_y = classifier.predict(X_test)

score=classifier.score(x,y)
print("The score of the KNN model = ", score)


# In[547]:


# Adding Industry column to the dataframe and mapping each row in the data with the company’s corresponding industry

for tick in all_tickers:
    gettick = yf.Ticker(tick)
    s = gettick.info
    s = s.get('sector')
    comp = df[df.TICKER == tick]
    ind = list(comp.index)
    df.loc[ind,"INDUSTRY"] = s
    
    


# In[ ]:


# Finding Unique industries to which industries in the data belong to.
df["INDUSTRY"].unique()


# In[570]:


# Plotting scatter plots to identify compensation trends in different industries
for industry in df["INDUSTRY"].unique():
    sns.scatterplot(data = df[df["INDUSTRY"] == industry], x = "Stock_High_Price", y = "Comp_without_Salary").set(title= industry)
    plt.show()


# In[579]:


# Plotting Normalized columns to rectify the trends in different industries
for industry in df["INDUSTRY"].unique():
    sns.scatterplot(data = df[df["INDUSTRY"] == industry], x = "Norm_Stock_Price", y = "Norm_Comp_without_Salary").set(title= industry)
    plt.show()


# In[ ]:


# Creating machine learning model
# We are using K nearest neighbor model as the trend is uncertain and KNN can fit regression as well as classification problems
normalized_data = df[["Norm_Comp_without_Salary", "Norm_Stock_Price"]].dropna()
x = normalized_data["Norm_Comp_without_Salary"].values.reshape(-1,1)
y = normalized_data.Norm_Stock_Price.values.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

from sklearn.neighbors import KNeighborsRegressor
classifier = KNeighborsRegressor(n_neighbors = 1)
classifier.fit(X_train, y_train)
pred_y = classifier.predict(X_test)

score=classifier.score(x,y)
print(score)


# In[ ]:


# Creating a csv file of the newly made data frame with additional columns
df.to_csv(r'C:\Users\keval\Desktop\Python\Data Research Lab\final_df1.csv')


# In[3]:


print("The score of the KNN model = 0.6")

