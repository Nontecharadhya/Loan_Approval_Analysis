#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max.rows',None)
pd.set_option('display.max.columns',None)


# In[3]:


os.getcwd()


# In[4]:


os.chdir('C:\\Users\\Abhi\\documents\\readings')


# In[5]:


df =pd.read_csv('loan_approval_dataset.csv')


# In[6]:


df.shape


# In[7]:


df.isnull().sum()


# In[8]:


df.duplicated(keep=False).sum()


# In[9]:


df.dtypes


# In[10]:


df.describe()


# In[11]:


df[' residential_assets_value']=df[' residential_assets_value'].abs()
(df[' residential_assets_value']<0).sum()


# In[12]:


#To trim extra spaces.
df.columns = df.columns.str.strip()


# In[13]:


def cibil_rate(value):
    if 300<= value <=549:
        return 'poor'
    elif 550<= value <=699:
        return 'good'
    elif 700<= value <=850:
        return 'very good'
    elif 851<= value <=864:
        return 'excellent'
    else :
        'unknown'
# Apply the function to create a new column
df['cibil_range']=df['cibil_score'].apply(cibil_rate)
df.shape


# In[14]:


def income_value(value):
    if 3000000<= value <=5490000:
        return 'low'
    elif 5500000<= value <=6990000:
        return 'moderate'
    elif 7000000<= value <=8400000:
        return 'high'
    elif 8500000<= value <=9000000:
        return 'very high'
    else :
        'unknown'
# Apply the function to create a new column
df['income_range']=df['income_annum'].apply(income_value)
df.shape


# In[15]:


def dependents(value):
    if 0<= value <=1:
        return 'low'
    elif 2<= value <=3:
        return 'moderate'
    elif 4<= value <=5:
        return 'high'
    else :
         return 'unknown'
# Apply the function to create a new column
df['dependency']=df['no_of_dependents'].apply(dependents)
df.shape


# # EDA

# In[16]:


income =df['income_range'].value_counts().reset_index()
income.rename(columns={'income_range':'income_levels','count':'total'},inplace=True)
income


# In[17]:


plt.figure(figsize=(8,6))
plt.title('Distribution of InocmeRange')
plt.pie(income.total,labels=income.income_levels,autopct='%.1f%%')
plt.axis('equal')
plt.show()


# In[18]:


dependent =df['dependency'].value_counts().reset_index()
dependent.rename(columns={'count':'total'},inplace=True)
dependent


# In[19]:


employment=df['self_employed'].value_counts().reset_index()
employment.rename(columns={'count':'no_of_applicant'},inplace=True)
employment


# In[20]:


df['loan_status'].value_counts()


# In[21]:


loan=df['loan_status'].value_counts().reset_index()
loan.rename(columns={'count':'Total'},inplace=True)
loan


# In[22]:


# plot a pie chart
plt.figure(figsize=(8,6))
plt.title('Overall rate of Loan Status')
plt.pie(loan.Total,labels=loan.loan_status,autopct='%.1f%%')
plt.axis('equal')
plt.show()


# In[23]:


edu=df.groupby(['education','loan_status']).size().reset_index()
edu.rename(columns={' education':'Education',0:'Total'},inplace=True)
edu


# In[24]:


pivot_table = edu.pivot_table(index='education', columns='loan_status',values='Total')
row_total = pivot_table.sum(axis=1)
percentage = pivot_table.div(row_total, axis=0) * 100
percentage


# In[25]:


# create a heat map 
plt.title('Loan Approval by Education')
sns.heatmap(percentage,annot=True,fmt='.1f',cmap='coolwarm',linewidth=0.5)
plt.xlabel('Loan Status')
plt.ylabel('Education')
plt.show()


# In[26]:


employment_type =df.groupby(['loan_status','self_employed']).size().reset_index()
employment_type.rename(columns={'loan_status':'LoanStatus','self_employed':'SelfEmployed',0:'Total'},inplace=True)
employment_type


# In[27]:


# create a pivot table
pivot_table =employment_type.pivot(index='LoanStatus',columns='SelfEmployed',values='Total')
row_total =pivot_table.sum(axis=1)
percentage =pivot_table.div(row_total,axis=0)*100
percentage


# In[28]:


# create a heat map 
plt.title('Loan Approval by SelfEmployed')
sns.heatmap(percentage,annot=True,fmt='.1f',cmap='coolwarm',linewidth=0.5)
plt.xlabel('Loan Status')
plt.ylabel('Self Employed')
plt.show()


# # Correlation

# In[29]:


correlation_matrix =df.corr(numeric_only =True)
correlation_matrix

# create a heatmap
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',fmt='.2f')
plt.show()


# In[30]:


assets =['residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value','income_annum']
income ='income_annum'

correlation =df[assets + [income]].corr()
correlation.head()


# # Asset value impact/influence on loan approval/elegiblity
# ## If there is a relation asset value and getting a loan approval

# In[31]:


from scipy import stats
from scipy.stats import f_oneway 
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency 


# In[32]:


# splitting data into two group based on loan status
approved =df[df['loan_status']=='Accepted']
rejected =df[df['loan_status']=='Rejected']

# performance a t-test
columns = ['residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value', 'income_annum']
df[columns].head()


# In[33]:


print(df['loan_status'].unique())


# In[34]:


print(df['loan_status'].isnull().sum())


# In[35]:


df['loan_status'].value_counts()


# In[36]:


# splitting data into two group based on loan status
approved =df[df['loan_status']==' Approved']
rejected =df[df['loan_status']==' Rejected']

# performance a t-test
columns = ['residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value', 'income_annum']

print(approved.shape,rejected.shape)


# In[37]:


for column in columns:
    t_stat,p_value =ttest_ind(approved[column],rejected[column],equal_var=False)
    print(f'T-Test for{column}:')
    print(f'T-Statistic: {t_stat}')
    print(f' P-value:{p_value}')

if p_value <0.05:
    print(f'The differnce in {column} between and rejected loans statistically significant')
else :
    print(f'The differnce in {column} between and rejected loans not statistically significant')


# # Using ANOVA to determine a correlation between income and asset value

# ## Null Hypothesis: There is no relation (significant difference) between assets value and income.
# ## Alternative Hypothesis :There exist a relationship (significant difference) between asset value and income.

# In[38]:


from scipy.stats import f_oneway

# Extracting values in columns
residential_assets = df['residential_assets_value']
commercial_assets = df['commercial_assets_value']
luxury_assets = df['luxury_assets_value']
bank_assets = df['bank_asset_value']
income = df['income_annum']

# Perform ANOVA
f_statistic, p_value = f_oneway(residential_assets, commercial_assets, luxury_assets, bank_assets, income)

print('The F-statistic:', f_statistic)
print('The p-value is:', p_value)

# Interpretation of the results
if p_value > 0.05:
    print('There is a significant influence of asset values on income per annum.')
else:
    print('There is no significant influence of asset values on income per annum.')


# # Correlation between income and Loan amount

# In[39]:


income =df['income_annum']
loan =df['loan_amount']

#perform an independent t-test
t_stat,p_value =ttest_ind(income,loan,equal_var=True)

print('The T-Statistic:',t_stat)
print('The p value is:',p_value)
if p_value <0.5:
    print('There is a statistical significant difference in loan amount and income per annum')
else:
    print('There is no statistical significant difference in loan amount and income per annum')


# # Chi-Square Test: Determine correlation between loan status and number of dependents

# In[40]:


# Creating a contingency table
contingency_table = pd.crosstab(df['loan_status'], df['dependency'])

# Perform chi-square test
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

print('Chi-Square test statistic:', chi2)
print('The P-value is:', p_value)
print('The degrees of freedom (dof) is:', dof)

# Set the significance level
alpha = 0.05

if p_value < alpha:
    print('There is a statistically significant relationship between loan status and the number of dependents.')
else:
    print('There is no statistically significant relationship between loan status and the number of dependents.')


# # Logistic Regression

# In[43]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[44]:


# Example: Using LabelEncoder for binary categorical columns
le = LabelEncoder()
df['education'] = le.fit_transform(df['education'])
df['self_employed'] = le.fit_transform(df['self_employed'])

# If there are other categorical variables with more than two categories, use pd.get_dummies()
df = pd.get_dummies(df, columns=['cibil_range', 'income_range', 'dependency'], drop_first=True)


# In[45]:


scaler = StandardScaler()
numerical_features = ['income_annum', 'loan_amount', 'loan_term', 'cibil_score',
                      'residential_assets_value', 'commercial_assets_value',
                      'luxury_assets_value', 'bank_asset_value']

df[numerical_features] = scaler.fit_transform(df[numerical_features])


# In[46]:


X = df.drop(columns=['loan_id', 'loan_status'])
y = df['loan_status']


# In[47]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[48]:


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# In[49]:


y_pred = model.predict(X_test)


# In[ ]:


# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred)
print('Classification Report:\n', class_report)


# In[50]:


# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy Score:', accuracy)


# ## The logistic regression model achieved an accuracy of 0.97 on the test set, indicating high performance.
# 

# In[51]:


# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred)
print('Classification Report:\n', class_report)

# Precision, Recall, and F1-Score
precision = class_report.split()[-4]
recall = class_report.split()[-3]
f1_score = class_report.split()[-2]

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1_score}')


# Interpretation: The model is very precise, particularly in predicting loan approvals, meaning it has a low false positive rate.

# In[ ]:




