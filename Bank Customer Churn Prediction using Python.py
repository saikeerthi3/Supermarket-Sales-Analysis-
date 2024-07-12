#!/usr/bin/env python
# coding: utf-8

# Importing libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report


# Exploring Dataset

# In[2]:


df=pd.read_csv("C:/Users/saike/Downloads/Churn_Modelling.csv")
df.head(5)


# In[3]:


df.describe()


# In[4]:


print(df.isnull().sum())


# In[5]:


df.duplicated().sum()


# we do not have any missing values

# In[6]:


df.dtypes


# In[7]:


unwanted_features = ['RowNumber', 'CustomerId', 'Surname']
df = df.drop(columns=unwanted_features)


# In[8]:


df.info()


# In[9]:


numerical_columns = df.select_dtypes(include=['int', 'float']).columns.tolist()
categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

print("Numerical columns:", numerical_columns)
print("Categorical columns:", categorical_columns)


# In[10]:


unique_values = df['Geography'].unique()

print("Unique values in the column:", unique_values)


# In[11]:


unique_values = df['Gender'].value_counts()
print("Unique values in the column:")
print(unique_values)


# In[12]:


df['Gender'] = df['Gender'].map({'Male':1,'Female':0})
df['Gender'].head(5)


# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(data=df, x='Age', bins=10, color='skyblue', edgecolor='black', kde=False)
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[14]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Defining age groups
age_groups = pd.cut(
    df['Age'],
    bins=[18, 25, 40, 60, 94],  
    labels=['Young Adult', 'Adult', 'Middle-aged', 'Senior Citizen'],  
    include_lowest=True  
)


df['age_group'] = age_groups


group_counts = df['age_group'].value_counts().sort_index()  


plt.figure(figsize=(8, 6))
sns.barplot(x=group_counts.index, y=group_counts.values, palette='muted')  
plt.xlabel('Age Group')
plt.ylabel('Count of People')
plt.title('Count of People in Each Age Group')
plt.show()


# In[15]:


age_group_counts = df.groupby(['age_group', 'Exited']).size().reset_index(name='Count')


sns.lineplot(x='age_group', y='Count', hue='Exited', data=age_group_counts, marker='o')


plt.xlabel('Age Group')
plt.ylabel('Count')
plt.title('Exited Customers by Age Group (Line Graph)')

plt.show()


# In[16]:


product_counts = df.groupby(['NumOfProducts', 'Exited']).size().reset_index(name='Count')


sns.lineplot(x='NumOfProducts', y='Count', hue='Exited', data=product_counts, marker='o')

plt.xlabel('Number of Products')
plt.ylabel('Count')
plt.title('Exited Customers by Number of Products (Line Graph)')
plt.show()


# In[17]:


tenure_exited_counts = df.groupby(['Tenure', 'Exited']).size().reset_index(name='Count')

sns.lineplot(data=tenure_exited_counts, x='Tenure', y='Count', hue='Exited', marker='o')


plt.xlabel('Tenure')
plt.ylabel('Count')
plt.title('Exited Customers by Tenure (Line Graph)')


plt.show()


# In[18]:


sns.histplot(data=df, x='CreditScore', bins=10, color='skyblue', edgecolor='black', kde=False)
plt.title('Histogram of Credit - Score')
plt.xlabel('Credit Score')
plt.ylabel('Frequency')
plt.show()


# In[19]:


ax = sns.countplot(x='IsActiveMember', hue='Exited', data=df)

for p in ax.patches:
    ax.annotate(
        f'{int(p.get_height())}',  
        (p.get_x() + p.get_width() / 2, p.get_height()),  
        ha='center', 
        va='center', 
        fontsize=10, 
        color='black',  
        xytext=(0, 10),  
        textcoords='offset points'  
    )

ax.set_xlabel('Active Member')
ax.set_ylabel('Count')
ax.set_title('Exited Customers by Activity')
plt.show()


# In[20]:


plt.figure(figsize=(8, 6))
sns.countplot(x='Exited', data=df)
plt.title('Distribution of Exited')
plt.xlabel('Exited')
plt.ylabel('Count')
plt.show()


# In[21]:


plt.figure(figsize=(8, 6))
sns.countplot(x='Geography', hue='Exited', data=df)
plt.title('Count of Exited Customers by Geography')
plt.xlabel('Geography')
plt.ylabel('Count')
plt.legend(title='Exited', labels=['No', 'Yes'])
plt.show()


# In[22]:


sns.countplot(x='Gender', hue='Exited', data=df)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Exited Customers by Gender')
plt.show()


# In[23]:


plt.figure(figsize=(8, 6))
sns.countplot(x='HasCrCard', hue='Exited', data=df)
plt.title('Count of Exited Customers')
plt.xlabel('Have a credit card')
plt.ylabel('Count')
plt.legend(title='Exited', labels=['No', 'Yes'])
plt.show()


# In[24]:


plt.figure(figsize=(8, 6))
plt.boxplot(df['EstimatedSalary'], vert=True, widths=0.7, patch_artist=True)
plt.title('Boxplot of Estimated Salary')
plt.xlabel('Estimated Salary')
plt.show()


# In[25]:


plt.figure(figsize=(8, 6))
plt.boxplot(df['Balance'], vert=True, widths=0.7, patch_artist=True)
plt.title('Boxplot of Balance')
plt.xlabel('Balance')
plt.show()


# In[26]:


def outlier_Identification():
    Q1 = df['Balance'].quantile(0.25)
    Q3 = df['Balance'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df['Balance'] < lower_bound) | (df['Balance'] > upper_bound)]['Balance']
    return outliers
outliers = outlier_Identification()
print(outliers)


# As we see that there is no outlier from the boxplot and also using the IOR method, we can confirm that Balance has no Outliers.

# In[27]:


categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
encoded_data = pd.get_dummies(df, columns=categorical_columns, dtype=int)
encoded_data = encoded_data.drop(columns = ['Age'])
encoded_data.head(5)


# In[28]:


correlation_matrix = encoded_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[29]:


y=encoded_data['Exited']
encoded_data = encoded_data.drop(columns = ['Exited'])


# In[30]:


encoded_data.head(5)


# In[31]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X = encoded_data


X_scaled = scaler.fit_transform(X)


# In[32]:


print(y.value_counts())


# In[33]:


from imblearn.over_sampling import SMOTE
x_resample, y_resample = SMOTE().fit_resample(X_scaled, y.values.ravel())
print(x_resample.shape)
print(y_resample.shape)


# In[34]:


get_ipython().system('pip uninstall -y imbalanced-learn')
get_ipython().system('pip install imbalanced-learn')


# In[35]:


unique, counts = np.unique(y_resample, return_counts=True)
class_counts = dict(zip(unique, counts))

print(class_counts)


# # TRAIN TEST SPLIT

# In[36]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x_resample,y_resample,test_size = 0.2, random_state = 50,stratify = y_resample)


# In[37]:


print(len(X_train), len(y_train))


# # RANDOM FOREST CLASSIFIER

# In[38]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


# In[39]:


param_grid = {
    'n_estimators': [100, 300],  
    'max_depth': [None, 20],     
    'min_samples_split': [2, 10],  
    'min_samples_leaf': [1, 4],    
    'max_features': ['sqrt', None]  
}


rf = RandomForestClassifier(random_state=50)


grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)


grid_search.fit(X_train, y_train)


best_rf = grid_search.best_estimator_

cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='accuracy')


print(f'CV Accuracy Scores: {cv_scores}')
print(f'CV Accuracy Mean: {np.mean(cv_scores):.4f}')


y_pred = best_rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy of the model: {accuracy:.4f}')


print(f'Best hyperparameters: {grid_search.best_params_}')


# In[40]:


class_report1 = classification_report(y_test, y_pred)
print('\nClassification Report:')
print(class_report1)


# In[41]:


conf_matrix1 = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix1, annot=True, fmt="d", cmap='magma')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# In[42]:


importances = best_rf.feature_importances_
feature_names = encoded_data.columns if isinstance(encoded_data, pd.DataFrame) else [f'Feature {i}' for i in range(X_train.shape[1])]
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
n = len(feature_importances['Feature'])
colors = plt.cm.viridis(np.linspace(0, 1, n))

plt.figure(figsize=(10, 6))
plt.bar(feature_importances['Feature'], feature_importances['Importance'], color=colors)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.xticks(rotation=45, ha='right')
plt.show()


# # XGBOOST CLASSIFIER

# In[43]:


get_ipython().system(' pip install xgboost')
from xgboost import XGBClassifier


# In[44]:


from sklearn.model_selection import GridSearchCV


param_grid = {
    'n_estimators': [200,300],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3,4, 5],
    'colsample_bytree': [0.5, 0.7],
    'gamma': [0, 0.05,0.10],
    'subsample': [0.9, 1.0]
}


grid_search = GridSearchCV(
    estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=50),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)


grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_


y_pred2 = best_model.predict(X_test)
train_accuracy = accuracy_score(y_train, best_model.predict(X_train))
test_accuracy = accuracy_score(y_test, y_pred2)


print(f'Train Accuracy: {train_accuracy:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
print('Best parameters found: ', grid_search.best_params_)


# In[45]:


class_report2 = classification_report(y_test, y_pred2)
print('\nClassification Report:')
print(class_report2)


# In[46]:


conf_matrix2 = confusion_matrix(y_test, y_pred2)
sns.heatmap(conf_matrix2, annot=True, fmt="d", cmap='magma')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# In[47]:


importances = best_model.feature_importances_
feature_names = encoded_data.columns if isinstance(encoded_data, pd.DataFrame) else [f'Feature {i}' for i in range(X_train.shape[1])]
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
n = len(feature_importances['Feature'])
colors = plt.cm.viridis(np.linspace(0, 1, n))

plt.figure(figsize=(10, 6))
plt.bar(feature_importances['Feature'], feature_importances['Importance'], color=colors)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.xticks(rotation=45, ha='right')
plt.show()


# # Neural Network Model

# In[48]:


get_ipython().system('pip install tensorflow')


# In[49]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


model = Sequential([
    Dense(64, activation='relu', input_shape=(15,)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=38, batch_size=32, validation_data=(X_train, y_train), verbose=1)


test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')


# In[50]:


y_pred_prob = model.predict(X_test)
y_pred3 = (y_pred_prob > 0.5).astype("int32") 
print(classification_report(y_test, y_pred3))


# In[51]:


conf_matrix3 = confusion_matrix(y_test, y_pred3)
sns.heatmap(conf_matrix3, annot=True, fmt="d", cmap='magma')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# In[52]:


from sklearn.metrics import roc_curve, roc_auc_score


fpr = []
tpr = []
auc_val = []

y_test_prob1 = best_rf.predict_proba(X_test)[:, 1]
fpr_1, tpr_1, thresholds = roc_curve(y_test, y_test_prob1)
auc_score1 = roc_auc_score(y_test, y_test_prob1)
fpr.append(fpr_1)
tpr.append(tpr_1)
auc_val.append(auc_score1)


y_test_prob2 = best_model.predict_proba(X_test)[:, 1]
fpr_2, tpr_2, thresholds = roc_curve(y_test, y_test_prob2)
auc_score2 = roc_auc_score(y_test, y_test_prob2)
fpr.append(fpr_2)
tpr.append(tpr_2)
auc_val.append(auc_score2)

y_test_prob3 = model.predict(X_test)
fpr_3, tpr_3, thresholds = roc_curve(y_test, y_test_prob3)
auc_score3 = roc_auc_score(y_test, y_test_prob3)
fpr.append(fpr_3)
tpr.append(tpr_3)
auc_val.append(auc_score3)

plt.figure(figsize=(10, 8))
colors = ['green', 'red', 'blue']  # Added 'blue' as an additional color
labels = ['Random Forest', 'XGBoost Classifier', 'Neural Networks']
for i in range(len(fpr)):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=f'{labels[i]} (AUC = {auc_val[i]:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves for various models')
plt.legend(loc='lower right')
plt.show()


# In[53]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import pandas as pd


accuracy_rf = accuracy_score(y_test, best_rf.predict(X_test))
accuracy_xgb = accuracy_score(y_test, best_model.predict(X_test))
precision_rf = precision_score(y_test, best_rf.predict(X_test))
precision_xgb = precision_score(y_test, best_model.predict(X_test))
recall_rf = recall_score(y_test, best_rf.predict(X_test))
recall_xgb = recall_score(y_test, best_model.predict(X_test))
auc_score_rf = roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1])
auc_score_xgb = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])


y_pred_prob = model.predict(X_test)
y_pred_nn = (y_pred_prob > 0.5).astype(int)  
accuracy_nn = accuracy_score(y_test, y_pred_nn)
precision_nn = precision_score(y_test, y_pred_nn)
recall_nn = recall_score(y_test, y_pred_nn)
auc_score_nn = roc_auc_score(y_test, y_pred_prob)


evaluation_metrics = {
    'Model': ['Random Forest Classifier', 'XGBoost Classifier', 'Neural Network Model'],
    'Accuracy': [accuracy_rf, accuracy_xgb, accuracy_nn],
    'Precision': [precision_rf, precision_xgb, precision_nn],
    'Recall': [recall_rf, recall_xgb, recall_nn],
    'AUC Score': [auc_score_rf, auc_score_xgb, auc_score_nn]
}


evaluation_metrics_df = pd.DataFrame(evaluation_metrics)

print(evaluation_metrics_df)


# In[ ]:




