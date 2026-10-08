import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load dataset
df = pd.read_csv("data/stock_market_survey_updated.csv")

# 2. Select input features and target output
features = [
    'Gender', 
    'Age', 
    'Occupation', 
    'Invested_in_Stock_Market', 
    'Investment_Interest', 
    'Uses_Stock_Market_Apps_Websites', 
    'Visualization_Importance'
]
target = 'Dashboard_Useful'

# Drop missing values
data = df[features + [target]].dropna().copy()

# 3. Explicitly convert all categorical text columns to numeric codes
encoders = {}
categorical_cols = [
    'Gender', 
    'Occupation', 
    'Invested_in_Stock_Market', 
    'Investment_Interest',  
    'Uses_Stock_Market_Apps_Websites', 
    'Visualization_Importance'
]

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    encoders[col] = le

target_le = LabelEncoder()
data[target] = target_le.fit_transform(data[target].astype(str))

X = data[features]
y = data[target]

# 4. Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Random Forest Model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 6. Evaluate Model
y_pred = clf.predict(X_test)
print("=" * 45)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("=" * 45)
print(classification_report(y_test, y_pred, target_names=target_le.classes_))

# 7. Save model and encoders to 'models/' folder
joblib.dump(clf, "models/dashboard_predictor.pkl")
joblib.dump(encoders, "models/encoders.pkl")
joblib.dump(target_le, "models/target_encoder.pkl")
print("\nSuccess: Trained model and encoders saved to 'models/' folder!")
