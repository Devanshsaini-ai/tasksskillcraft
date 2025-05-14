

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


titanic = pd.read_csv('train.csv') 


titanic.drop(columns='Cabin', inplace=True)


titanic['Age'].fillna(titanic['Age'].median(), inplace=True)


titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace=True)


print("Missing values:\n", titanic.isnull().sum())


titanic['Pclass'] = titanic['Pclass'].astype(str)



sns.set(style="whitegrid")


sns.countplot(data=titanic, x='Survived')
plt.title('Survival Count (0 = Died, 1 = Survived)')
plt.show()


sns.countplot(data=titanic, x='Survived', hue='Sex')
plt.title('Survival by Sex')
plt.show()


sns.countplot(data=titanic, x='Survived', hue='Pclass')
plt.title('Survival by Passenger Class')
plt.show()


sns.histplot(data=titanic, x='Age', bins=30, kde=True)
plt.title('Age Distribution')
plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(titanic.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


sns.boxplot(data=titanic, x='Pclass', y='Age')
plt.title('Age Distribution by Class')
plt.show()



print("\nSummary Statistics:\n")
print(titanic.describe(include='all'))
