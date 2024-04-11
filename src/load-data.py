import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

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
split_x = int(0.8*len(x))
split_y = int(0.8*len(y))
train_x = x[:split_x]
train_y = y[:split_y]
test_x = x[split_x:]
test_y = y[split_y:]

model = RandomForestClassifier(n_jobs=-1)
print("Model created, fitting...")
model.fit(train_x, train_y)
print("Model fitted, predicting...")
predictions = model.predict(test_x)
print("Predictions made, calculating metrics...")
report = classification_report(test_y, predictions)
print(report)
cm = confusion_matrix(test_y, predictions)
print(cm)


