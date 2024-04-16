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



age_data_path = './data/csv/age_gender_bkts.csv'
age_gender_df = pd.read_csv(age_data_path)

users_data_path = './data/csv/train_users_2.csv'
users_df = pd.read_csv(users_data_path)

test_users_path = './data/csv/test_users.csv'
test_users_df = pd.read_csv(test_users_path)

print(age_gender_df.head())
print(users_df.head())
print(test_users_df.head())

def proccess_data(df: pd.DataFrame):
    X = df.drop(columns=["id", "country_destination"], axis=1)
    
    # split date_account_created (YYYY-MM-DD) into 3 different columns year, month, day
    pdt = pd.to_datetime(X["date_account_created"])
    X["year_created"] = pdt.dt.year
    X["month_created"] = pdt.dt.month
    X["day_created"] = pdt.dt.day
    X.drop(columns=["date_account_created"], axis=1, inplace=True)

    # pdf = pd.to_datetime(X["date_first_booking"])
    # X["year_first_booking"] = pdf.dt.year
    # X["month_first_booking"] = pdf.dt.month
    # X["day_first_booking"] = pdf.dt.day

    # This isn't in the test set, so we drop it. 
    X.drop(columns=["date_first_booking"], axis=1, inplace=True)

    X["age"].fillna(0, inplace=True)
    X["age"] = X["age"].astype('int8')

    X["language"] = X["language"].astype('category').cat.codes
    X["gender"] = X["gender"].astype('category').cat.codes
    X["signup_method"] = X["signup_method"].astype('category').cat.codes
    X["affiliate_channel"] = X["affiliate_channel"].astype('category').cat.codes
    X["affiliate_provider"] = X["affiliate_provider"].astype('category').cat.codes
    X["first_affiliate_tracked"] = X["first_affiliate_tracked"].astype('category').cat.codes
    X["signup_app"] = X["signup_app"].astype('category').cat.codes
    X["first_device_type"] = X["first_device_type"].astype('category').cat.codes
    X["first_browser"] = X["first_browser"].astype('category').cat.codes

    Y = df["country_destination"].astype('category').cat.codes
    return X, Y


# Do a basic prediction
print("Proccessing data...")
x, y = proccess_data(users_df)

# Print the columns and their datatypes
print("Columns and their datatypes:")
print(x.dtypes.to_string())

# Print the distribution of classes in y
distribution = y.value_counts()
print("Distribution of classes in y:")
print(distribution)

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("Smotting") 
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

# print("Scaling...")
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)

# model = RandomForestClassifier(n_jobs=-1)
# model = LogisticRegression(n_jobs=-1)
# model = NaiveBayes()
# model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=100000)
model = XGBClassifier(tree_method="hist", early_stopping_rounds=2, n_estimators=500, n_jobs=-1)

print("Model created, fitting...")

# model.fit(train_x, train_y)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

print("Model fitted, predicting...")
predictions = model.predict(X_test)
print("Predictions made, calculating metrics...")
report = classification_report(y_test, predictions)
print(report)
cm = confusion_matrix(y_test, predictions)
print(cm)


