from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np
import joblib

# Load data
df = pd.read_csv("forestfires.csv")
df['fire'] = df['area'].apply(lambda x: 1 if x > 0 else 0)

# One-hot encode month and day
df = pd.get_dummies(df, columns=['month', 'day'], drop_first=True)

# Drop 'area'
df = df.drop(['area'], axis=1)

# Split into features and target
X = df.drop('fire', axis=1)
y = df['fire']

# Feature selection
selector = SelectKBest(score_func=f_classif, k=4)
X_new = selector.fit_transform(X, y)

# Save selected feature names
selected_columns = X.columns[selector.get_support()]
print("Selected features:", selected_columns)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "knn_model.pkl")
joblib.dump(scaler, "knn_scaler.pkl")
joblib.dump(selected_columns.tolist(), "selected_features.pkl")
