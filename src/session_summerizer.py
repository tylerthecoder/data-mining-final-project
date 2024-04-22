import pandas as pd
import numpy as np



def calculate_entropy(column):
    # Counts occurrence of each value
    counts = column.value_counts()
    # Calculates probabilities
    probabilities = counts / counts.sum()
    # Calculates entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def get_summerized_session_data(use_cache=True):
    cache_path = './data/csv/session_summerized.csv'

    if use_cache:
        try:
            # Return cached data if available
            return pd.read_csv(cache_path)
        except FileNotFoundError:
            pass

    print("Summerized session data not found. Creating new one")
    
    sessions_data_path = './data/csv/sessions.csv'

    print("Reading session file")
    sessions_df = pd.read_csv(sessions_data_path)
    sessions_df = sessions_df.fillna(value=pd.NA)

    row_count_df = sessions_df.groupby('user_id').size().reset_index(name='row_count')


    # Count rows with an action present per user
    action_present_df = sessions_df[sessions_df['action'].notna()].groupby('user_id').size().reset_index(name='actions_present_count')


    aggregated_df = sessions_df.groupby('user_id')['secs_elapsed'].agg(
        sum_secs='sum',           # Sum of seconds
        mean_secs='mean',         # Mean of seconds
        min_secs='min',           # Minimum seconds
        max_secs='max',           # Maximum seconds
        median_secs='median',     # Median of seconds
        sd_secs='std',            # Standard deviation of seconds
        skewness_secs='skew',     # Skewness of seconds
        # Todo add kurtosis
    ).reset_index()


    # Count unique actions, action_types, action_details, and devices per user
    unique_counts_df = sessions_df.groupby('user_id').agg({
        'action': 'nunique',
        'action_type': 'nunique',
        'action_detail': 'nunique',
        'device_type': 'nunique'
    }).reset_index().rename(columns={
        'action': 'unique_actions',
        'action_type': 'unique_action_types',
        'action_detail': 'unique_action_details',
        'device_type': 'unique_device_types'
    })




    # Merge all the dataframes
    aggregated_df = pd.merge(aggregated_df, row_count_df, on='user_id', how='left')
    aggregated_df = pd.merge(aggregated_df, action_present_df, on='user_id', how='left')
    aggregated_df = pd.merge(aggregated_df, unique_counts_df, on='user_id', how='left')



    print(aggregated_df.shape)
    aggregated_df.head()


    # Save the data to a file
    print("Saving to file")
    aggregated_df.to_csv(cache_path, index=False)

if __name__ == '__main__':
    get_summerized_session_data(use_cache=False)


