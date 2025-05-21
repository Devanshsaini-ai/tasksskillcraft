from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# Step 1: Fetch the dataset
bank_marketing = fetch_ucirepo(id=222)

# Step 2: Extract features and target
X = bank_marketing.data.features
y = bank_marketing.data.targets

# Step 3: Handle missing values (optional: drop or fill)
X = X.fillna('unknown')

# Step 4: Encode categorical variables using one-hot encoding
X_encoded = pd.get_dummies(X)

# Encode target variable if it's in string format ('yes', 'no')
y_encoded = y.replace({'yes': 1, 'no': 0})

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Step 6: Build and train the Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = clf.predict(X_test)

# Step 8: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))