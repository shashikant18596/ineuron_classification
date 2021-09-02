#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)


# In[3]:


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'


# In[4]:


columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','salary']


# In[5]:


df = pd.read_csv(url,names = columns)
df.head(2)


# In[6]:


df.shape


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


df.isnull().sum()


# In[10]:


df.workclass.unique()


# In[11]:


df.workclass.value_counts()


# In[12]:


df =df.replace(' Never-worked',' Without-pay')
df['workclass'].value_counts()


# In[13]:


df.replace(' ?',np.nan,inplace= True)
df['workclass'].fillna('0',inplace=True)


# In[14]:


sns.countplot(x = df['workclass'])
plt.xticks(rotation = 45)
plt.show()


# In[15]:


df['salary'].unique()


# In[16]:


salary = {' <=50K': 0 , ' >50K':'1'}
df = df.replace(salary)
df.head(2)


# In[17]:


df['salary'].value_counts()


# In[18]:


sns.countplot(x=df['salary'])
plt.xticks(rotation = 0)


# In[19]:


df['education'].value_counts()


# In[20]:


sns.catplot(x='education',y=pd.to_numeric(df['salary']),data=df,height=10,palette='muted',kind='bar')
plt.xticks(rotation=45)


# In[21]:


df['marital-status'].value_counts()


# In[22]:


df['marital-status'].replace(' Married-AF-spouse', ' Married-civ-spouse',inplace=True)


# In[23]:


sns.catplot(x='marital-status',y=pd.to_numeric(df['salary']),data=df,palette='muted',kind='bar',height=8)
plt.xticks(rotation=45)


# In[24]:


df['occupation'].fillna('0',inplace=True)
df['occupation'].value_counts()


# In[25]:


df['occupation'].replace(' Armed-Forces','0',inplace=True)
df['occupation'].value_counts()


# In[26]:


sns.catplot(x='occupation',y=pd.to_numeric(df['salary']),data=df,palette='muted',kind='bar',height=8)
plt.xticks(rotation=45)


# In[28]:


sns.pairplot(df,hue='salary',height=3)
plt.plot()


# In[34]:


corr = df.corr()
sns.heatmap(corr,annot = True,cmap='YlGnBu')


# In[35]:


df.drop('fnlwgt',axis=1,inplace=True)


# In[36]:


df.head(n=2)


# In[37]:


X = df.drop('salary',axis=1)
y = pd.to_numeric(df['salary'])


# In[38]:


X_d = pd.get_dummies(X)
X_d.head(2)


# In[39]:


from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
x_train,x_test,y_train,y_test = train_test_split(X_d,y,test_size=0.3,random_state=101)


# In[41]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[42]:


classifier = [DecisionTreeClassifier(random_state=42),RandomForestClassifier(random_state=42)]


# In[43]:


dt_grid_param = { "min_samples_split" : range(10,500,20),
                "max_depth": range(1,20,2)

}


# In[46]:


rf_grid_param = {"max_features": [1,3,10],
                "min_samples_split":[2,3,10],
                "min_samples_leaf":[1,3,10],
                "bootstrap":[False],
                "n_estimators":[100,300],
                "criterion":["gini"]}


# In[47]:


classifier_param = [dt_grid_param,rf_grid_param]


# In[48]:


cv_result = []
best_estimators = []
for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)
    clf.fit(x_train,y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    print(cv_result[i])


# In[50]:


cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier", "RandomForestClassifier"]})


# In[55]:


cv_results


# In[56]:


g = sns.barplot(y="Cross Validation Means", x="ML Models", data = cv_results)
g.set_xlabel("Mean Accuracy")
g.set_title("Cross Validation Scores")


# In[ ]:




