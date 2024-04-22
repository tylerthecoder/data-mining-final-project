import pandas as pd


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


    aggregated_df = pd.merge(aggregated_df, row_count_df, on='user_id')


    print(aggregated_df.shape)
    aggregated_df.head()


    # Save the data to a file
    print("Saving to file")
    aggregated_df.to_csv(cache_path, index=False)

if __name__ == '__main__':
    get_summerized_session_data(use_cache=False)


