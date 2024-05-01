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
        return 'Unknown Age Group'

def classify_device(x):
    if x.find('Desktop') != -1:
        return 'Desktop'
    elif x.find('Tablet') != -1 or x.find('iPad') != -1:
        return 'Tablet'
    elif x.find('Phone') != -1:
        return 'Phone'
    else:
        return 'Unknown Device'

def classify_gender(x):
    if x == "MALE":
        return "man"
    elif x == "FEMALE":
        return "girl"
    else:
        return "Unknown Gender"

def languages(df):
    df['language'] = df['language'].apply(lambda x: 'foreign' if x != 'en' else x)
    return df

def affiliate_tracked(df):
    df['first_affiliate_tracked'] = df['first_affiliate_tracked'].fillna('Unknown_affiliate_tracked')
    df['first_affiliate_tracked'] = df['first_affiliate_tracked'].apply(lambda x: 'Other' if x != 'Unknown' and x != 'untracked' else x)
    return df

def affiliate_channel(df):
    df['affiliate_channel'] = df['affiliate_channel'].apply(lambda x: 'other' if x  not in ['direct_ac', 'content_ac'] else x)
    return df

def affiliate_provider(df):
    df['affiliate_provider'] = df['affiliate_provider'].apply(lambda x: 'rest' if x not in ['direct_ap', 'google_ap', 'other_ap'] else x)
    return df

def browsers(df):
    df['first_browser'] = df['first_browser'].apply(lambda x: "Mobile_Safari" if x == "Mobile Safari" else x)
    major_browsers = ['Chrome', 'Safari', 'Firefox', 'IE', 'Mobile_Safari']
    df['first_browser'] = df['first_browser'].apply(lambda x: 'Other Browser' if x not in major_browsers else x)
    return df

def process_data(df: pd.DataFrame):
    # Combine the session data
    X = df.merge(sum_session, left_on="id", right_on="user_id", how="left")

    # If country_destination is present, drop it
    if "country_destination" in X.columns:
        X = X.drop(columns=["country_destination"], axis=1)

    X = X.drop(columns=["id", "user_id", "date_first_booking"], axis=1)

    # Handling date_account_created
    pdt = pd.to_datetime(X["date_account_created"])
    X["year_created"] = pdt.dt.year
    X["month_created"] = pdt.dt.month
    X["day_created"] = pdt.dt.day
    X.drop(columns=["date_account_created"], axis=1, inplace=True)

    X['age'] = X['age'].apply(lambda x: np.nan if x > 120 else x)
    X['age_bucket'] = X['age'].apply(set_age_group).astype('category').cat.codes

    X['first_device_type'] = X['first_device_type'].apply(classify_device)

    X["gender"] = X["gender"].apply(classify_gender)

    X['is_3'] = X['signup_flow'].apply(lambda x: 1 if x==3 else 0)


    X = affiliate_tracked(X)
    X = affiliate_provider(X)
    X = affiliate_channel(X)
    X = languages(X)
    X = browsers(X)

    # Encoding categorical variables
    # categorical_columns = ["language", "gender", "signup_method", "affiliate_channel", 
    #                        "affiliate_provider", "first_affiliate_tracked", "signup_app", 
    #                        "first_device_type", "first_browser"]
    # for col in categorical_columns:
    #     X[col] = X[col].astype('category').cat.codes

    X = pd.get_dummies(X, prefix="is")

    return X
    


# Do a basic prediction
print("Processing data...")

y = users_df["country_destination"].astype('category').cat.codes
x = process_data(users_df)

# Print the columns and their datatypes
print("Columns and their datatypes:")
print(x.dtypes.to_string())

# Print the distribution of classes in y
distribution = y.value_counts()
print("Distribution of classes in y:")
print(distribution)


