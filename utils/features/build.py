import pandas as pd
from haversine import haversine

def build_hour_group(df:pd.DataFrame):
    """Builds hour column

    Args:
        df (pd.DataFrame): A pandas data frame with column promised_time.

    Returns:
        df (pd.DataFrame): A pandas data frame with the new columns.
    """
    # hour of the day
    df['hour'] = df['promised_time'].apply(lambda x: x.hour)
    # Create groups to represent different moments of the day
    df['hour_group'] = 'afternoon'
    df.loc[df['hour'].between(0, 6, inclusive=True), 'hour_group'] = 'dawn'
    df.loc[df['hour'].between(7, 12, inclusive=True), 'hour_group'] = 'morning'
    df.loc[df['hour'].between(18, 23, inclusive=True), 'hour_group'] = 'night'
    return df

def build_distance(df:pd.DataFrame):
    """Builds distance_km column

    Args:
        df (pd.DataFrame): A pandas data frame with lat_os, lng_os, lat_strb, lng_strb columns

    Returns:
        df (pd.DataFrame): A pandas data frame with the new column.
    """
    df['distance_km'] = df.apply(lambda x: haversine((x['lat_os'], x['lng_os']), 
                                                    (x['lat_strb'], x['lng_strb'])), axis=1)
    return df