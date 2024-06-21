# titanic-classification-
@codealpha_tasks


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv(titanic.csv)

from pandas import DataFrame
import DataFrame as df

# Display basic information and statistics
print(df.info())
print(df.describe())

# Visualize missing data
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.show()

# Visualize survival rate by different features
sns.countplot(x='Survived', data=df)
plt.show()

sns.countplot(x='Survived', hue='Pclass', data=df)
plt.show()

sns.countplot(x='Survived', hue='Sex', data=df)
plt.show()

sns.countplot(x='Survived', hue='Embarked', data=df)
plt.show()
# Fill missing values for 'Age' with the median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing 'Embarked' with the most frequent value
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' due to too many missing values
df.drop('Cabin', axis=1, inplace=True)

# Drop 'Ticket' as it doesn't seem to provide useful information
df.drop('Ticket', axis=1, inplace=True)

# Encode categorical features
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])

# Create 'FamilySize' feature
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Drop irrelevant features
df.drop(['Name', 'PassengerId'], axis=1, inplace=True)



X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1,random_state=0)

y_test

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

 model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
 y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

