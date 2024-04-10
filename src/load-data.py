import pandas as pd
import numpy as np

age_data_path = './data/csv/age_gender_bkts.csv'
age_gender_df = pd.read_csv(age_data_path)

users_data_path = './data/csv/train_users_2.csv'
users_df = pd.read_csv(users_data_path)

print(age_gender_df.head())
print(users_df.head())
