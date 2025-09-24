# Aggregate time series data
def aggregate_time_series(df, freq_list=["D", "M"]):
    """
    Returns aggregated time series DataFrames for the specified frequencies.
    
    Parameters:
    - df: cleaned DataFrame with DateTime index
    - freq_list: list of resample frequencies. "D"=daily, "M"=monthly, etc.
    
    Returns:
    - dict of aggregated DataFrames, keys are frequencies
    """
    agg_dict = {}
    for freq in freq_list:
        df_agg = df.resample(freq).sum()
        agg_dict[freq] = df_agg
        print(f"{freq} aggregation done: {df_agg.shape[0]} rows, {df_agg.shape[1]} columns")
    return agg_dict
