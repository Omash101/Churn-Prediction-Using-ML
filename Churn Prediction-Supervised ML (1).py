#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install xgboost


# In[2]:


get_ipython().system('pip install delayed')


# In[ ]:





# # Context
#  ConnectTel is a leading telecommunications company at the forefront of innovation and connectivity solutions.
# it has strong presence in the global market, with a well established trusted provider of reliable voice, data, and internet services. Offering a comprehensive range of Telecommunications solutions, including mobile networks, broadband connections, and enterprise solutions. It caters to both individual and corporate customers, they are commited to providing exceptional customer service and cutting-edge technology.
# 
#  ConnectTel ensures seamless communoication experiences for millions of users worldwide through strategic parnerships and a customer-centric approach, ConnectTel continue to revolutionize the telecom industry, empowering individuals and businesses rto stay connected and thrive in the digital age

# # Objective
# 
# To address customer churn,using supervised learning Machine which poses a significant threat to ConnectTel business sustainability and growth. The Company current customer retention strategies lack precision and effectiveness, resulting in the loss of valuable customers to competitors. To overcome this challenge, there is the need to develop a robust customer churn prediction system by leveraging on advance analytics and Machine learning techniqueson available customer data to accurately predict customer churn and implement targeted retrention initiatives.This approach will help to reduce customer attrition and bring back customer confidence, and maintain a competitive edge in the highly dynamic and competitive telecommunictions industry.

# # Data Description
# 
# - The data provided is of various customers
# 
# Data Dictionary
# - customerID: A unique Identifier assigned to each Telecom customer, enabling tracking and identification of individual customer.
# - Gender: The gender of the customer, which can be categorized as male, or female. This information helps to analyzing gender- based trends in customer churn.
# - SeniorCitizen: A binary indicator that identifies whether the customer is a senior Citizen or not.This attributes help in understanding if there are any specific churn patterns among the senior customers.
# - Partner: Indicates whether the customer has a partner or not. This attribute helps in evaluating the impact of having a partner on churn behavior.
# - Dependents:Indicates whether the customer has dependants or not.This attribute help in assessing the influence of having dependents oncustomer churn.
# - Tenure: The duration for which the customer has been subscribed to the telecom service. I t represents the loyalty of longevity of the customer's relationship with the company and it is a significant predictor of churn.
# - PhoneService: Indicates whether the customer has a phone service or not.This attribute helps in understanding the impact of phone service on churn
# - MultipleLines: Indicates whether the customer has multiple lines or not.This attribute helps in analyzing the effect of having multiple lines on customer churn.
# - InternetService: Indicates the type of internet service subscribed by the customer, such as DSL, fibre optics or no internet service. It helps in evaluating the relationship between internet service and churn.
# - OnlineSecurity: Indicates whether the customer has online security services or not. This attribute helps in analyzing the impact of online security on customer churn.
# - OnlineBackup: Indicates whether the customer has online backup services or not. This service helps in evaluating theimpact of online backup on churn behavior.
# - DeviceProtection: This indicates whether the customer has device protection services or not. The attribute helps in understanding the influence of device protection on churn.
# - TechSupport: Indicates whether the customer has technical support services or not.This attribute helps in assessing the impact of tech support on churn behavior.
# - StreamingTV: Indicates whether the customer has streaming Tv services or not.The attribute helps in evaluating the impact of streaming Tv on customer churn.
# - StreamingMovies:Indicates whether the customer has streaming Movies services or not. This attribute helps in understanding the influence of streaming movies on churn behavior.
# - Contract:Indicates the type of contract the customer has, such as a month-to-month, one-year, or two year contract. It is a crucial factor in predicting churn as different contract length may have varying impacts on customer loyalty.
# - PaperlessBilling: Indicates whether the customer has opted for paperless billing or not.This attribute helps in analyzing the effect of paperless billing on customer churn.
# - PaymentMethod: Indicates the payment method used by the customer, such as electronic checks, mailed checks, bank transfers, or credit cards. This attribute helps in evaluating the impact of payment method on churn.
# - MonthlyCharges: The total amount charged to the customer over the entire tenure. It represents the cumulative revenue generated from the customer and may have an impact on the churn.
# - Churn: The target variable indicates whether the customer has churn(cancel the service) or not. It is the main variable to predict in Telecom customer churn analysis

# In[3]:


get_ipython().system('pip install imbalanced-learn==0.8.0')


# In[4]:


get_ipython().system('pip install delayed')


# ### APPROACH
# - PREPARE DATA (DATA PROCESSING)
# - EDA
# - BUILD THE MODELS
# - MODEL EVALUATION
# - MODEL OPTIMIZATION
# - REPORT THE BEST PERFORMING MODEL

# In[5]:


# import necessary Libraries

# for data Analysis
import pandas as pd
import numpy as np


# for Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Data Pre-processing, to tune model, get different metric score and split data
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import pickle

import scipy.stats as stats


from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline, make_pipeline



# Classifier Libraries to help with model building
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# !pip install xgboost
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix


# to suppress scientic notations
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import warnings
warnings.filterwarnings('ignore')


# In[6]:


# Load the dataset
data = pd.read_csv(r"C:\Users\omats\Downloads\Customer-Churn.csv")
data.head()


# In[7]:


# view bottom
data.tail()


# In[8]:


data.shape


# ### Observation
# - There are 7043 rows and 21 columns

# In[ ]:





# In[9]:


# Data verification - Data type, number of features and rows, missing data, etc
data.info()


# In[10]:


# checking for duplicated data
data.duplicated().sum()


# ### Observation
# - There are no duplicate data

# In[11]:


# Explore missing data

data.isnull().sum()


# ### Observation
# - There are no missing data

# In[12]:


# checking the number of unique values in each column
data.nunique()


# ### Observation
# - This is an indication that there are no duplicate data and no missing number

# In[13]:


# Statistical analysis of the Numerical dataset
data.describe().T


# In[ ]:





# ### Observations
# - Data looks legitimate as all the statistics seem reasonable
# - The SeniorCitizen is right-skewed there are younger citizen than senior citizen.
# - The tenure ranges from 0 to 72 and it seems to be evenly distributed with a close value of mean and median
# - The MonthlyCharges is right skewed
# - The average monthlycharges per customer is 70.35 while the maximum is 118.75

# In[14]:


data.drop(['customerID'], axis=1, inplace=True)


# In[15]:


data.describe(exclude='int64')


# In[16]:


{x: len(data[x].unique()) for x in data.columns}


# In[17]:


for i in data.describe(include=['object']).columns:
    print('unique values in' ,i, 'are :')
    print(data[i].value_counts())
    print('*'*50)


# ### OBSERVATIONS
# - The records are for existing customers.
# - Most of the customers are male.
# - There are more non Partners than partners.
# - 70% of the customers are non dependent
# - 90% of the customers use phoneservice
# - 42% of the customers have multiplelines while 48% don't have multiplelines
# - 43.96% of the customers use fiber optic for internet service more than those that uses DSL
# - 49.67% does not use online security more than those that uses online backup.
# - From records 21.6% of the customer does not have interner services. 
# - 44% of the customer does not use online backup
# - 44% use device protection.
# - 29% of the customers use TechSupport less than those who don't use.
# - StreamingTV users are 1% less than non streamingTV users
# - Streaming Movies users are slightly less than non streamingMovies use
# - 55% of the customers are on Month-to Month contract.
# - 59% uses paperlessbilling
# - 34% prefered payment method is Electronic check
# - out of 7043 customers 1869 churn service which is 27%, which is quite high.

# ### There is need for data pre-processing after a closed look at the unique value of the categorical variables

# In[18]:


data['Churn'].unique()


# In[19]:


data['Partner'].unique()


# In[20]:


data['MultipleLines'].unique()


# In[21]:


data['Dependents'].unique()


# In[22]:


data['Contract'].unique()


# In[23]:


data['gender'].unique()


# In[24]:


data['InternetService'].unique()


# In[25]:


data['PaperlessBilling'].unique()


# In[ ]:





# ### Converting the dtype 'object' to categorical variables

# In[27]:


data["gender"]=data["gender"].astype("category")
data["Partner"]=data["Partner"].astype("category")


# In[28]:


data.info()


# ### Exploratory Data Analysis

# ### Univariate Analysis

# In[29]:


# While doing univariate analysis of numerical variables we want to study their central tendency and dispersion.
# Let us write a function that will help us create boxplot and histogram for any input numerical variables.
# This function takes the numerical column as the input and retrurns the boxplots and histograms for the variable.
# Let see if this help us write faster and clearer code.

def histogram_boxplot(feature, figsize=(25,15), bins = None):
    """ Boxplot and histogram combined
    feature: 1-d feature array
    figsize: size of fig (default (20,12))
    bins: number of bins (default None / auto)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(nrows = 2, # Number of rows of the subplot grids=2
                                          sharex = True, # x-axis will be shared among all subplots
                                          gridspec_kw = {"height_ratios": (.25, .75)},
                                          figsize = figsize
                                          )# creating the 2 subplots
    sns.boxplot(x=feature, ax=ax_box2, showmeans=True, color='violet') # boxplot will be created and a star will indicate the mean of the column
    sns.histplot(x=feature, kde=F, ax=ax_hist2, bins=bins, palette="winter") if bins else sns.histplot(x=feature, kde=False, ax=ax_hist2)# for histogram
    ax_hist2.axvline(np.mean(feature), color='green', linestyle='--')# Add mean to the histogram
    ax_hist2.axvline(np.median(feature), color='black', linestyle='-') #Add median to the histogram


# In[30]:


data.columns


# In[31]:


# check for outliers
histogram_boxplot(data['tenure'])


# In[32]:


data['tenure'].describe()


# ### Observation
# - The tenure is uniformly distributed
# - The range of tenure is 0 and 72
# - The average tenure is 32
# - The mean and the median, 32 and 29 respectively are not far apart meaning there is no outliers

# In[33]:


# check for outliers
histogram_boxplot(data['MonthlyCharges'])


# ### Observation
# - The customer MonthlyCharges is uniformly distributed but slightly skewed to the right
# - The range of MonthlyCharges is between 18.25 to 118.75
# - The average MonthlyCharge is 64.762
# - There are no outlier because the mean and the median are not far apart.
# - The maximum Monthly Charge is 118.750 is could be customer using all three of the service.
# 

# In[34]:


data['MonthlyCharges'].describe()


# In[ ]:





# In[35]:


histogram_boxplot(data['SeniorCitizen'])


# In[36]:


data['SeniorCitizen'].describe()


# ### Observation
# - The company has about 1200 seniorCitizen as customer while the balance 5,843 are young adult.
# - There are outlier because the 75 percentile i9s far from the max value and skewed to the right.

# In[37]:


# countplot for some of the variables
fig, axs = plt.subplots(2,2, figsize = (20,7.5))
plt1 = sns.countplot(x=data['gender'], ax = axs[0,0])
plt2 = sns.countplot(x=data['SeniorCitizen'], ax = axs[0,1])
plt3 = sns.countplot(x=data['Churn'], ax = axs[1,0])
plt4 = sns.countplot(x=data['Dependents'], ax = axs[1,1])   


# In[ ]:





# In[38]:


data['gender'].describe()


# In[39]:


data['SeniorCitizen'].describe()


# In[40]:


data['Dependents'].describe()


# In[41]:


data['Churn'].describe()


# ### Observation
# - There are more male customer than female
# - 2110 out of 7043 are dependents
# - 27% of the customer chruned the company services

# In[42]:


fig, ax = plt.subplots(figsize=(5,5))
count = Counter(data['PaymentMethod'])
ax.pie(count.values(), labels=count.keys(), autopct=lambda p:f'{p:.2f}%')
plt.show()


# ### Observation 
# - Customers prefered mode of payment is Electronic Check, which is 33.58% followed by Mailed check

# In[43]:


data['PaymentMethod'].describe()


# ### Observation
# - The customers prefer paying with eletronic check foloowed by Bank Transfer

# In[44]:


fig, ax = plt.subplots(figsize=(5,5))
count = Counter(data["StreamingTV"])
ax.pie(count.values(), labels=count.keys(), autopct=lambda p:f'{p:.2f}%')
plt.show()


# ### Observation
# - 38.44% of the customer v iew there program through StreamingTV this is a lot

# In[45]:


fig, ax = plt.subplots(figsize=(5,5))
count = Counter(data["StreamingMovies"])
ax.pie(count.values(), labels=count.keys(), autopct=lambda p:f'{p:.2f}%')
plt.show()


# ### Observation
# - 38.79% of their customer stream movies

# In[46]:


fig, ax = plt.subplots(figsize=(5,5))
count = Counter(data["InternetService"])
ax.pie(count.values(), labels=count.keys(), autopct=lambda p:f'{p:.2f}%')
plt.show()


# ### Observation
# - 43.96% of the customers use fiber optic

# In[ ]:





# In[47]:


fig, ax = plt.subplots(figsize=(5,5))
count = Counter(data["OnlineSecurity"])
ax.pie(count.values(), labels=count.keys(), autopct=lambda p:f'{p:.2f}%')
plt.show()


# ### Observation
# - 28.67% use OnlineSecurity

# In[48]:


fig, ax = plt.subplots(figsize=(5,5))
count = Counter(data["Contract"])
ax.pie(count.values(), labels=count.keys(), autopct=lambda p:f'{p:.2f}%')
plt.show()


# ### Observation
# - Most of the customers prefer to Month-to- month payment

# ## Bivariate

# In[49]:


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(25,20))

gender_PhoneService = data.groupby("gender")["PhoneService"].count().reset_index()
sns.barplot(x="gender", y="PhoneService",  data=gender_PhoneService, ax=axs[0,0])
axs[0,0].set_title("gender vs PhoneService");

gender_tenure = data.groupby("gender")["tenure"].count().reset_index()
sns.barplot(x="gender", y="tenure", data=gender_tenure, ax=axs[0,1])
axs[0,1].set_title("gender vs tenure");

gender_Contract = data.groupby("gender")["Contract"].count().reset_index()
sns.boxplot(x="gender", data=gender_Contract, y="Contract", ax=axs[1,0])
axs[1,0].set_title("gender vs Contract");

Contract_PaymentMethod = data.groupby("Contract")["PaymentMethod"].count().reset_index()
sns.barplot(x="Contract", data=Contract_PaymentMethod, y="PaymentMethod", ax=axs[1,1])
axs[1,1].set_title("Contract vs PaymentMethod");


# ### Observation
# - Male uses PhoneService than their female counterpart
# - The male customers have spent more time using the company's services than the female customer
# - The male prefer the highest subscription than the female.
# - Most customers prefer to Month-to-month subscription

# In[50]:


plt.figure(figsize=(15,5))
sns.heatmap(data.corr(), annot=True, vmin=-1, vmax=1)
plt.show()


# ### Observation
# - There is correlation between all the variables and they are positive but not high.
# - There correlation between tenure and monthlyCharges is higher than the others.

# In[51]:


sns.pairplot(data)  #pairplot
plt.show()


# ### Observation
# - The correlation between tenure and monthly charges is evenly distributed but the longer the tenure the higher the charges

# ### MULTIVARIATE

# In[52]:


plt.figure(figsize=(20,5))
sns.barplot(x='PaperlessBilling', y='MonthlyCharges', data=data, hue='gender')


# ### Observation
# - Female using PaperlessBilling are charged more on Monthly bases than their male counterpart using PaperlessBilling an indication that female uses voice, data and internet services more.
# - The Female and MAle without PaperlessBilling are charged the same.

# In[53]:


plt.figure(figsize=(20,5))
sns.barplot(x='Contract', y='MonthlyCharges', data=data, hue='gender')


# ### Observation
# - Most of the customers prefer Month-to-month subcription  followed by one year.

# In[54]:


##plt.figure(figsize=(20,5))
##sns.barplot(x='Contract', y='SeniorCitizen', data=data, hue='MonthlyCharges')


# In[55]:


### function to plot boxplot
def boxplot(x):
    plt.figure(figsize=(10,7))
    sns.boxplot(data=data, x="Churn",y=data[x], palette="PuBu")


# In[56]:


data.columns


# In[57]:


boxplot('tenure')


# ### Observation
# - Churn customers have only tenures between 3 to 26 while the existing customers have spent longer time.
# - There are outliers for those churn customers and indication of their unsual behavior to leave the company.
# - The mean for the customer no churning is higher than those churn customer. The mean is an indicatrion of their stay in thr company.

# In[58]:


boxplot('MonthlyCharges')


# ### Observation
# - Churn customers are those who are charged between 60 and 90 with an average monthly charge of 80

# In[59]:


boxplot('SeniorCitizen')


# In[ ]:





# In[60]:


plt.figure(figsize=(20,5))
sns.barplot(x='gender', y='MonthlyCharges', data=data, hue='PaperlessBilling')


# ### Observation
# - Female are charged more an indication that they use the services more.

# In[61]:


plt.figure(figsize=(20,5))
sns.barplot(x='gender', y='MonthlyCharges', data=data, hue='Churn')


# ### Observation
# - Female customers churn the company services more than the male customers because they are charged more monthly due to the fact they spend more on the company services affecting their financies.In spite of the fact they churn more the female customers using the services are still more.

# In[62]:


### function to plot stacked bar charts for categorical columns
def stacked_plot(x):
    sns.set(palette='nipy_spectral')
    tab1 = pd.crosstab(x,data['Churn'], margins=True)
    print(tab1)
    print('-'*120)
    tab = pd.crosstab(x,data['Churn'],normalize='index')
    tab.plot(kind='bar', stacked=True, figsize=(10,5))
    plt.legend(loc='lower left', frameon=False)
    plt.legend(loc="upper left", bbox_to_anchor=(1,1))
    plt.show()


# In[63]:


stacked_plot(data['gender'])


# ### Observations
# - Female customers are slightly more to churn the company's services than male customers

# In[64]:


stacked_plot(data['Dependents'])


# ### Observation
# - Customers with dependents are less likely to churn than those without dependents

# In[65]:


stacked_plot(data['MultipleLines'])


# ### Obeservation
# - Those with multipleLines are more likely to churn ConnetTel Services than those without multiple lines and no Phone service

# In[66]:


stacked_plot(data['StreamingTV'])


# ### Observation
# - Customers with no StreamingTV are slight likely to churn than those with StreamingTV, while those with no internewt service are more likely not to churn propably they are only using the PhoneService.

# In[67]:


stacked_plot(data["OnlineSecurity"])


# ### Observation
# - It is clear that trhose with no OnlineSecurity service are more likely to churn than those with OnlineSecurity and no internet services respectively.

# In[68]:


stacked_plot(data['Contract' ])


# ### Observation 
# - It is very clear that the contract terms affect customers use of the company's service. The longer the contract the more likelyhood that the customer will not churn the services.

# In[69]:


stacked_plot(data["TechSupport"])


# ### Observation
# - Likewise as the case with OnlineSecurity so as with TechSupport. That those with no TechSupport Are more likely to churn the company's services than those with TechSupport and those with no internet service respectively

# In[70]:


stacked_plot(data["StreamingMovies"])


# ### Observation
# - The difference in churning the company between customers with No StreamingMovies and customers with StreamingMovies is very small. Customer with no StreamingMovies are slightly likely to churn than those with StreamingMovies.

# In[71]:


stacked_plot(data["Partner"])


# ### Observation
# - Customers without Partners are more likely to churn the company services than customers with Partner

# In[72]:


stacked_plot(data["DeviceProtection"])


# ### Observation
# - Customers without DeviceProtection are more likely to churn the company's services than those with DeviceProtection

# ### Let's deal with outliers, in each column of the data, using IQR

# In[73]:


Q1 = data.quantile(0.25) #To find the 25th percenttile and 75th percentile
Q3 = data.quantile(0.75)

IQR = Q3 - Q1             # Inter Quantile Range (75th percentile - 25th percentile)

lower=Q1-1.5*IQR          # Finding lower and upper bounds for all values. All values outside these bounds are outliers
upper=Q3+1.5*IQR


# In[74]:


((data.select_dtypes(include=['float64', 'int64'])<lower) | (data.select_dtypes(include=['float64', 'int64'])>upper)).sum()/len(data)*100


# - After identifying outliers, we can decide whether to remove/treat them or not.It depends on one's approach,in this case we are not going to treat them as there will be outliers in real case scenerios. Let the model learn the underlying pattern for such customers.

# ## DATA PREPROCESSING

# In[75]:


data1 = data.copy()


# In[76]:


# dropping off some redundant features
data.drop(['PaymentMethod', 'MonthlyCharges','TotalCharges'], axis=1, inplace=True)


# In[77]:


# one hot encoding all the categorical features in the data
data.replace(to_replace = ["No", "Yes"], value = [0,1], inplace = True)
data.replace(to_replace= ["Male", "Female"], value = [0,1], inplace = True)


# In[78]:


data.replace(to_replace =["Month-to-month", "One Year","Two Year"], value = [0,1,2], inplace = True)


# In[79]:


data.replace(to_replace =["Fiber Optic", "DSL","No InternetService"], value = [0,1,2], inplace = True)


# ### Encoding all the categorical features in the dataset

# In[80]:


#encoding all the categorical features in the dataset
data = pd.get_dummies(data,drop_first=True)


# In[81]:


data


# In[82]:


# Data segmentation (Churn as the target data)

y = data.pop('Churn')


# In[93]:


#data


# In[89]:


scaler = StandardScaler()
scaled_data= pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
#scaled_data


# ### Identifying key features from the dataset

# In[95]:


# Identifying key features from the data set
# plotting a feature importance chart
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data,y)

model = RandomForestClassifier()
#fit the model
model.fit(data_scaled,y)
feature_names = list(data.columns)
importances = model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10,7))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color='lightgreen', align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# ### Observation
# - The tenure is the most important features meaning the more the tenure the higher the monthlyCharges and the totalChargesand the likely customer might churn the company's services
# - Fiber optic is another feature that the customer might consider to churn the company

# ### PHARE 2
# - SPLITTING DATA INTO TRAINING AND EVALUATION SPLIT
# - IMPLEMENTING /BUILDING MODEL
# - CREATING A PREDICTION

# ### The classification problem has imbalance distribution of the target classes, therefore it is good to use startified sampling to ensure that the relative class frequencies in both the train and test data.

# ### Split the data into train and test sets

# In[98]:


# splitting data into training and test set:
X_train, X_test, y_train, y_test = train_test_split(scaled_data,y,test_size=0.3, random_state=1, stratify=y)
print(X_train.shape, X_test.shape)


# In[99]:


imputer = KNNImputer(n_neighbors=9) # ensuring there are to leaking data


# In[100]:


#Fit and transform the train data
X_train = pd.DataFrame(imputer.fit_transform(X_train),columns=X_train.columns)

# Transform the test data
X_test=pd.DataFrame(imputer.transform(X_test),columns=X_test.columns)


# ### Encoding categorical variables

# In[101]:


X_train = pd.get_dummies(X_train,drop_first=True)
X_test = pd.get_dummies(X_test,drop_first=True)
print(X_train.shape, X_test.shape)


# ### After encoding there are 25 columns each on the test and train data. The train data has 4930 while the test data has 2113.

# In[102]:


y.value_counts(1)


# In[103]:


y_test.value_counts(1)


# ### Model Building

# ### Model evaluation criterion:

# ### Model can make wrong predictions as:
#  1. Predicting a customer will Churn the use of the company service and the customer might not churn.
#  2. Predicting a customer will not Churn the use of the Company's services and the customer will eventually churn

# ### Looking at the most important
# - predicting 2 above might lead to eventual loss of a valuable customer

# ### How to reduce this loss i.e need to reduce false Negatives?
# ##### ConnectTel wants recall to be maximized, greater the recall the lesser the chances of false negatives

# In[104]:


# Function to calculate different metric scores of the model - Accuracy, recall and Precision
def get_metrics_score(model,train,test,train_y,test_y,flag=True):
    '''
    model: classifier to predict values of X
    
    '''
    # defining an empty list to store train and test results
    score_list=[]
    
    pred_train = model.predict(train)
    pred_test = model.predict(test)
    
    train_acc = model.score(train,train_y)
    test_acc = model.score(test,test_y)
    
    train_recall = metrics.recall_score(train_y,pred_train)
    test_recall = metrics.recall_score(test_y,pred_test)
    
    train_precision = metrics.precision_score(train_y,pred_train)
    test_precision = metrics.precision_score(test_y,pred_test)
    
    score_list.extend((train_acc,test_acc,train_recall,test_recall,train_precision,test_precision))
    
    # If the flag is set to be True then only the following print statements will be displayed. The default value is set to True.
    if flag ==True:
        print("Accuracy on training set: ",model.score(train,train_y))
        print("Accuracy on test set :", model.score(test,test_y))
        print("Recall on training set :", metrics.recall_score(train_y,pred_train))
        print("Recall on test set :",metrics.recall_score(test_y,pred_test))
        print("Precision on training set:",metrics.precision_score(train_y,pred_train))
        print("Precision on test set :",metrics.precision_score(test_y,pred_test))
        
    return score_list #$ returning the list with train and test scores


# In[105]:


def make_confusion_matrix(model,y_actual,labels=[1,0]):
    '''
    model: classifier to predict values of X
    y_actual : ground truth
    
    '''
    y_predict = model.predict(X_test)
    cm=metrics.confusion_matrix(y_actual, y_predict, labels=[0,1])
    df_cm = pd.DataFrame(cm,index = [i for i in ["Actual - No", "Actual - Yes"]],
                        columns = [i for i in ['Predicted - No', 'Predicted - Yes']])
    group_counts = ["{0:0.0F}".format(value) for value in
                    cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                     cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}" for v1, v2 in
              zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=labels, fmt='')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# ### Logistic Regression

# In[106]:


lr = LogisticRegression(random_state=1)
lr.fit(X_train,y_train)


# In[139]:


g_boost = GradientBoostingClassifier()


# In[142]:


g_boost.fit(X_train,y_train)


# In[143]:


# create a prediction file
g_pred =g_boost.predict(X_test)
print('Gradient Boosting Model accuracy score :', format(g_boost.score(X_test,y_test)))


# ## Using KFold and Cross_val_score to evaluate the model performance

# ## Stratified K-Fold cross - validatrion provides dataset indices to split data into train/validation sets. Split dataset into k consecutive stratified folds
# (without shuffling by default). Each fold is then use once as validation while the k-1 remaining fold form the training set.

# In[108]:


scoring='recall'
kfold=StratifiedKFold(n_splits=5, shuffle=True,random_state=1) # Setting number of splits equal to 5
cv_result_bfr=cross_val_score(estimator=lr, X=X_train, y=y_train, scoring=scoring, cv=kfold)
# Plotting boxplots for CV\ scores of model defined above
plt.boxplot(cv_result_bfr)
plt.show


# ### Performance of the training set varies between 0.50 and 0.56 recall 

# ### Checking the performance on the test data

# In[109]:


# Calculating with different metrics
scores_LR = get_metrics_score(lr, X_train, X_test, y_train, y_test)

#Creating confusion matrix
make_confusion_matrix(lr, y_test)


# - Logistic Regression has given a generalized performance on training and test set
# - Recall is low, we can try oversampling (increase training data) to see if the model performance can be improved.
# - True positive shows that 14.43% churn

# In[ ]:





# In[ ]:





# ### Oversampling train data using SMOTE (Synthetric Minority Over Sampling Techniques)

# In[112]:


from imblearn.over_sampling import SMOTE


# In[113]:


print("Before UpSampling, counts of label 'Yes': {}".format(sum(y_train==1)))
print("Before UpSampling, counts of label 'No': {} \n". format(sum(y_train==0)))

sm = SMOTE(sampling_strategy = 1, k_neighbors = 5, random_state=1) #Synthetic Minority over sampling Techniques
X_train_over, y_train_over = sm.fit_resample(X_train, y_train)

print("After UpSampling, counts of label 'Yes': {}".format(sum(y_train_over==1)))
print("After UpSampling, counts of label 'No': {}\n".format(sum(y_train_over==0)))

print('After UpSampling, the shape of train_X: {}'.format(X_train_over.shape))
print('After UpSampling, the shape of train_y: {} \n'.format(y_train_over.shape))


# ### Logistic Regression on over sampled data

# 
# 

# In[114]:


log_reg_over = LogisticRegression(random_state =1)

# Training the basic logistic regression model with the training set
log_reg_over.fit(X_train_over, y_train_over)


# ### Evaluate the model performance by using KFOLD and cross_val_score

# - K- Folds cross-validation provides dataset indices to split data into train/validation sets. Split dataset into k consecutive stratified folds (without shuffling by default). Each fold is then used once as validation while the k-1 remaining folds form the training set.

# In[115]:


scoring='recall'
kfold=StratifiedKFold(n_splits=5, shuffle=True,random_state=1) # Setting number of splits equal to 5
cv_result_over=cross_val_score(estimator=log_reg_over, X=X_train_over, y=y_train_over, scoring=scoring, cv=kfold)
# Plotting boxplots for CV scores of model defined above
plt.boxplot(cv_result_over)
plt.show


# - Performance of the model on training set varies between 0.78 to 0.83, which is an improvement from the previous model.
# - The variability in the model performance has also decreased.
# - Let's check the performance on the test set.

# ### Checking the performance on the test set

# In[116]:


# Calculating different metrics
get_metrics_score(log_reg_over,X_train_over,X_test,y_train_over,y_test)

# creating confusion matrix
make_confusion_matrix(log_reg_over,y_test)


# - Model has given a generalized performance on training and test set and the performance has improved
# - The prediction has increased shows 20.73% of the customers churn
# - Performance has increased
# - The Recall has improved
# - a) Regularization to see if overfitting in precision can reduced
# - b) Undersampling the train dataset to handle the imbalance between classes and check the model performance

# ### UnderSampling train data using SMOTE

# In[117]:


from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state = 1)
X_train_un, y_train_un = rus.fit_resample(X_train, y_train)


# In[118]:


print("Before Under Sampling, counts of label 'Yes': {}".format(sum(y_train==1)))
print("Before Under Sampling, counts of label 'No': {} \n". format(sum(y_train==0)))


print("After Under Sampling, counts of label 'Yes': {}".format(sum(y_train_un==1)))
print("After Under Sampling, counts of label 'No': {}\n".format(sum(y_train_un==0)))

print('After Under Sampling, the shape of train_X: {}'.format(X_train_un.shape))
print('After Under Sampling, the shape of train_y: {} \n'.format(y_train_un.shape))


# ### Logistic Regression on underSampled data

# In[119]:


log_reg_under = LogisticRegression(random_state = 1)
log_reg_under.fit(X_train_un,y_train_un)


# ### Evaluating the model performance by using KFold and Cross_val_score

# ### K-Folds cross=validation provides dataset indices to splt data into train/validation sets.Split dataset into k consecutive stratified folds (without shuffling by default). Each fold is then used as validation while the k-1 remaining folds form the training set.

# In[120]:


scoring='recall'
kfold=StratifiedKFold(n_splits=5, shuffle=True,random_state=1) # Setting number of splits equal to 5
cv_result_under=cross_val_score(estimator=log_reg_under, X=X_train_un, y=y_train_un, scoring=scoring, cv=kfold)
# Plotting boxplots for CV scores of model defined above
plt.boxplot(cv_result_under)
plt.show()


# ### performance of the model on training set varies between 0.76 to 0.81, which is a slight fall from the upsampling model.

# ### Checking the performance on the test set

# In[121]:


# Calculating different metrics
get_metrics_score(log_reg_under,X_train_un,X_test,y_train_un,y_test)

# Creating confusion matrix
make_confusion_matrix(log_reg_under,y_test)


# - Model performance of the undersampling is slight lower than the upsampling model and now it is much better to differentiate between positive and negative classes
# - the model is still okay with true positive showing prediction of 20.63%
# - No need to regularize
# - Model performance improved when upsampling and downsampling techniques were used with logistic regression. There is no overfitting or underfitting on the result of both, so there is no need for regularization

# In[122]:





# ### Bagging and Boosting

# ### 1. Decision Tree Classifier

# In[123]:


# fitting the model
d_tree = DecisionTreeClassifier(random_state=1)
d_tree.fit(X_train,y_train)


# calculating different metrics
#get_metrics_score(d_tree)
get_metrics_score(d_tree,X_train,X_test,y_train,y_test)

#creating confusion matrix
make_confusion_matrix(d_tree,y_test)


# - The Recall is high, it seems the model is overfitting the training data as training recall/precision is much is much higher than the test recall/precision
# - The prediction dropped

# In[ ]:


### Evaluate the model


# In[148]:


from sklearn.metrics import classification_report


# In[ ]:





# ### Random Forest Classifier

# In[124]:


# fitting the model
rf_estimator = RandomForestClassifier(random_state=1)
rf_estimator.fit(X_train,y_train)

# Calculating different metrics
get_metrics_score(rf_estimator,X_train,X_test,y_train,y_test)

#Creating confusion matrix
make_confusion_matrix(rf_estimator,y_test)


# - Random Forest is better than Decision Tree in terms of precision test but low in recall
# - The model is overfitting the training data
# -  The pre

# 

# ### AdaBoost Classifier

# In[125]:


# fitting the model
ab_classifier = AdaBoostClassifier(random_state=1)
ab_classifier.fit(X_train,y_train)

#Calculating different metrics
get_metrics_score(ab_classifier,X_train,X_test,y_train,y_test)

# Creating confusion matrix
make_confusion_matrix(ab_classifier,y_test)


# - The model is not under but giving a generalized performance of training and test set.

# ### Gradient Boosting Classifier

# In[126]:


# Fitting the model
gb_classifier = GradientBoostingClassifier(random_state=1)
gb_classifier.fit(X_train,y_train)

# Calculating different metrics
get_metrics_score(gb_classifier,X_train,X_test,y_train,y_test)

#Creating confusion metrics
make_confusion_matrix(gb_classifier,y_test)


# - The model still looks like giving a generalized performance of the train and test set, loks underfitting

# ### XGBOOST Classifier

# In[127]:


# fitting the model
xgb_classifier = XGBClassifier(random_state=1)
xgb_classifier.fit(X_train,y_train)

# Calculating differnt metrics
get_metrics_score(xgb_classifier,X_train,X_test,y_train,y_test)

#Creating confusion metrix
make_confusion_matrix(xgb_classifier,y_test)


# - The model is able to generalized better on training data, the performance is not quite good
# - The True positive is 13.82% which is very okay because of the good spread on the Training, test and Recall data

# ### Comparing all models (Logistic Regression, Bagging and Boosting models)

# In[128]:


get_ipython().run_cell_magic('time', '', '# defining list of model\nmodels = [lr, log_reg_over,log_reg_under,d_tree,rf_estimator,xgb_classifier,ab_classifier,gb_classifier]\n\n# defining emptry lists to add train and test results\nacc_train = []\nacc_test = []\nrecall_train = []\nrecall_test = []\nprecision_train = []\nprecision_test = []\n\n# Looping through all the models to get the metrics score - Accuracy, recall and precision\nfor model in models:\n    \n    j = get_metrics_score(model,X_train,X_test,y_train,y_test,False)\n    acc_train.append(j[0])\n    acc_test.append(j[1])\n    recall_train.append(j[2])\n    recall_test.append(j[3])\n    precision_train.append(j[4])\n    precision_test.append(j[5])\n')


# In[129]:


comparison_frame = pd.DataFrame(
    {
        "Model":[
            'XGBoost', 
            'Logistic Regression with over sampling', 
            'Logistic Regression with under sampling', 
            'Decision Tree',                           
            'Gradient Boosting Classifier',
            'Logistic Regression',
            'AdaBoost Classifier',
            'Random Forest'],
        
            "Train_Accuracy":acc_train,
            "Test_Accuracy":acc_test,
            "Train_Recall":recall_train,
            "Test_Recall":recall_test,
            "Train_Precision":precision_train,
            "Test_Precision":precision_test,})
            
# sorting models in decreasing order of test recall
comparison_frame.sort_values(by='Test_Recall',ascending=False)


# ### The best three model
# 1. Gradient Boosting classifier
# 2. Decision Tree
# 3. Logistic Regression
# 4. The True positives of the above is higher than the rest.

# 1. The models as mentioned above have given a generalized performance of both the training and test set.
# 2. They have the best recall scores and are not over fitting the training set

# ### Tuning the model using GridSearCh, Random Search as well as Pipelines in hyperparameter tuning.

# ## Hyperparameter Tuning

# ### Using pipelines with StandardScaler and Xgboost,AdaBoost model &Gradient boost and tune the models using GridSearchCV and RandomizedSearchCV.After comparing the performances and time taken by these two methods- GridSearch and Randomized Search.

# ### Using the make_pieline function to create a pipeline.This is a shorthand for the pipeline constructor, it does require and also permit, naming the estimators.Instead, their names will be set to the lowercase of their tuypes automatically.

# ### Creating a function two functions to calculate different metrics and confusion matrix so that we don't have to use the same code repeatedly for each model

# ### Hyperparameter tuning 

# In[134]:


from sklearn.metrics import make_scorer, r2_score


# In[144]:


params ={
    'learning_rate':[0.1,0.5,1.0],
    'n_estimators':[50,100,150]
}

# to find the best parameters for the model
score = make_scorer(r2_score)
grid = GridSearchCV(g_boost,params,scoring=score,cv=3, return_train_score=True)
grid.fit(X_train,y_train)
print(grid.best_params_, "\n")


# In[145]:


best_model = grid.best_estimator_
new_pred = best_model.predict(X_test)
new_pred


# In[149]:


print(classification_report(y_test,new_pred))


# In[154]:


# instantiates mod
from sklearn.linear_model import LogisticRegression


# In[156]:


log_reg = LogisticRegression()


# In[158]:


## PRODUCTIONIZE MODEL
import joblib

filename ='./log_reg.pkl'
joblib.dump(log_reg,filename)


# In[ ]:





# In[ ]:


testing =  pd.


# ### BUSINESS INSIGHTS AND RECOMMENDATIONS

# - We have been able to build a predictable model
# - (a That ConnectTel  can deploy to identify customers who are at the risk of Churning.
# - (b The ConnectTel can use to find the key causes that drive the Churn
# - Factors that drive the Churn- Tenure /MonthlyCharges because they are correlated,InternetService_Fiber optic
# - Female customers should be the target customers for any kind of marketing campaign as they are the ones who make use of the services more.
# - There should be incentives to promote the use of TechSupport to encourage users 

# In[ ]:




