import pandas as pd
import numpy as np
from session_summerizer import get_summerized_session_data

print("Loading user data...")
users_data_path = './data/csv/train_users_2.csv'
users_df = pd.read_csv(users_data_path)

# Summerize the session data
print("Loading session data...")
sum_session = get_summerized_session_data()

print(users_df.head())
print(sum_session.head())

def set_age_group(x):
    if x < 40:
        return 'Young'
    elif x >=40 and x < 60:
        return 'Middle'
    elif x >= 60 and x <= 125:
        return 'Old'
    else:
        return 'Unknown'

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

    # Dropping date_first_booking
    X.drop(columns=["date_first_booking"], axis=1, inplace=True)

    X['age'] = X['age'].apply(lambda x: np.nan if x > 120 else x)
    X['age_bucket'] = X['age'].apply(set_age_group).astype('category').cat.codes

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


