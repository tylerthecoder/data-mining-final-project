import pandas as pd


def get_summerized_session_data():
    # Return cached data if available
    cache_path = './data/csv/session_summerized.csv'
    try:
        return pd.read_csv(cache_path)
    except FileNotFoundError:
        pass

    print("Summerized session data not found. Creating new one")
    
    sessions_data_path = './data/csv/sessions.csv'

    print("Reading session file")
    sessions_df = pd.read_csv(sessions_data_path)

    # agregate the number of seconds in each session for each user
    # fields is `secs_elapsed`
    print("Aggregating session data")
    summerized_df = sessions_df.groupby('user_id').agg({'secs_elapsed': 'sum'}).reset_index()

    # Save the data to a file
    print("Saving to file")
    summerized_df.to_csv(cache_path, index=False)





