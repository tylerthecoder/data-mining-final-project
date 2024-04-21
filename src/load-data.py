import pandas as pd
import numpy as np
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
from session_summerizer import get_summerized_session_data

print("Loading user data...")
users_data_path = './data/csv/train_users_2.csv'
users_df = pd.read_csv(users_data_path)

# Summerize the session data
print("Loading session data...")
sum_session = get_summerized_session_data()

print(users_df.head())
print(sum_session.head())

def process_data(df: pd.DataFrame):
    # Combine the session data
    X = df.merge(sum_session, left_on="id", right_on="user_id", how="left")

    X = X.drop(columns=["id", "country_destination", "user_id"], axis=1)
    
    # Handling date_account_created
    pdt = pd.to_datetime(X["date_account_created"])
    X["year_created"] = pdt.dt.year
    X["month_created"] = pdt.dt.month
    X["day_created"] = pdt.dt.day
    X.drop(columns=["date_account_created"], axis=1, inplace=True)

    X["secs_elapsed"] = X["secs_elapsed"].fillna(-1)
    # X = X.drop(columns=["secs_elapsed"], axis=1)

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
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

print("Training data shape: ", X_train.shape)
print(X_train.head())

print("Testing data shape: ", X_test.shape)
print(X_test.head())


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
cm_df = pd.DataFrame(cm, model.classes_, model.classes_)
print(cm_df)

print("Predicting probs")
# (num_samples, num_classes)
prob_preds = model.predict_proba(X_test)

# Get the second most likely class
second_most_likely_indices = [np.argsort(preds)[-2] for preds in prob_preds]
print("Second guesses")
cm2 = confusion_matrix(y_test, second_most_likely_indices)
cm_df2 = pd.DataFrame(cm2, model.classes_, model.classes_)
print(cm_df2)

third = [np.argsort(preds)[-3] for preds in prob_preds]
print("Third guesses")
cm3 = confusion_matrix(y_test, third)
cm_df3 = pd.DataFrame(cm3, model.classes_, model.classes_)
print(cm_df3)

val = ndcg(prob_preds, y_test)
print("SCORE: ", val)

