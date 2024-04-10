import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

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
    Y = df["country_destination"]
    X = pd.get_dummies(X, drop_first=True)
    return X, Y

# Do a basic prediction
x, y = proccess_data(users_df)

# Split the data into training and testing
train_x = x[:int(0.8*len(x))]
train_y = y[:int(0.8*len(y))]
test_x = x[int(0.8*len(x)):]
test_y = y[int(0.8*len(y)):]

model = DecisionTreeClassifier()
print("Model created, fitting...")
model.fit(train_x, train_y)
print("Model fitted, predicting...")
predictions = model.predict(test_x)
print("Predictions made, calculating metrics...")
report = classification_report(test_y, predictions)
print(report)





