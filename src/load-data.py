import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB as NaiveBayes
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from ndcg import ndcg



age_data_path = './data/csv/age_gender_bkts.csv'
age_gender_df = pd.read_csv(age_data_path)

users_data_path = './data/csv/train_users_2.csv'
users_df = pd.read_csv(users_data_path)

test_users_path = './data/csv/test_users.csv'
test_users_df = pd.read_csv(test_users_path)

print(age_gender_df.head())
print(users_df.head())
print(test_users_df.head())

def process_data(df: pd.DataFrame):
    X = df.drop(columns=["id", "country_destination"], axis=1)
    
    # Handling date_account_created
    pdt = pd.to_datetime(X["date_account_created"])
    X["year_created"] = pdt.dt.year
    X["month_created"] = pdt.dt.month
    X["day_created"] = pdt.dt.day
    X.drop(columns=["date_account_created"], axis=1, inplace=True)

    # Dropping date_first_booking
    X.drop(columns=["date_first_booking"], axis=1, inplace=True)

    # Handling missing values in age
    median_age = X["age"].median()
    X["age"].fillna(median_age, inplace=True)
    X["age"] = X["age"].clip(lower=0, upper=120).astype('int8')  # Ensure age is within a reasonable range

    # Encoding categorical variables
    categorical_columns = ["language", "gender", "signup_method", "affiliate_channel", 
                           "affiliate_provider", "first_affiliate_tracked", "signup_app", 
                           "first_device_type", "first_browser"]
    for col in categorical_columns:
        X[col] = X[col].astype('category').cat.codes
    
    Y = df["country_destination"].astype('category').cat.codes
    return X, Y

# Do a basic prediction
print("Processing data...")
x, y = process_data(users_df)

# Print the columns and their datatypes
print("Columns and their datatypes:")
print(x.dtypes.to_string())

# Print the distribution of classes in y
distribution = y.value_counts()
print("Distribution of classes in y:")
print(distribution)

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# print("Smoting...") 
# smote = SMOTE()
# X_train, y_train = smote.fit_resample(X_train, y_train)

# print("Scaling...")
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)

# model = RandomForestClassifier(n_jobs=-1)
# model = LogisticRegression(n_jobs=-1)
# model = NaiveBayes()
# model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=100000)
model = XGBClassifier(tree_method="hist", early_stopping_rounds=5, n_estimators=500, n_jobs=-1)

print("Model created, fitting...")

# model.fit(X_train, y_train)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)


print("Model fitted, predicting...")
predictions = model.predict(X_test)
print("Predictions made, calculating metrics...")
report = classification_report(y_test, predictions)
print(report)
cm = confusion_matrix(y_test, predictions)
print(cm)

print("Predicting probs")
prob_preds = model.predict_proba(X_test)

val = ndcg(prob_preds, y_test)
print("SCORE: ", val)

