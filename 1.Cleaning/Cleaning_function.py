# Function to clean the data
def clean_data(df, zero_thresh=1.0, near_const_thresh=0.99, convert_to_kwh=True):

    # Drop duplicates
    before = df.shape[0]
    df = df.drop_duplicates()
    print(f"Dropped {before - df.shape[0]} duplicate rows")
    
    # Drop all-zero columns
    zero_cols = df.columns[(df == 0).all()]
    df = df.drop(columns=zero_cols)
    print(f"Dropped {len(zero_cols)} all-zero columns")
    
    # Drop constant columns
    const_cols = df.columns[df.nunique() <= 1]
    df = df.drop(columns=const_cols)
    print(f"Dropped {len(const_cols)} constant columns")
    
    # Drop near-constant columns (e.g. >99% same value)
    near_const_cols = []
    for col in df.columns:
        top_freq = df[col].value_counts(normalize=True, dropna=False).values[0]
        if top_freq >= near_const_thresh:
            near_const_cols.append(col)
    df = df.drop(columns=near_const_cols)
    print(f"Dropped {len(near_const_cols)} near-constant columns")
    
    # Handle missing values
    missing_before = df.isna().sum().sum()
    df = df.fillna(method="ffill").fillna(method="bfill")
    missing_after = df.isna().sum().sum()
    print(f"Filled {missing_before - missing_after} missing values (ffill+bfill)")
    
    # Convert kW → kWh if requested
    if convert_to_kwh:
        # Skip first column if it’s DateTime or non-numeric
        numeric_cols = df.select_dtypes(include='number').columns
        df[numeric_cols] = df[numeric_cols] / 4
        print(f"Converted {len(numeric_cols)} columns from kW to kWh")
    
    print(f"Final dataset shape: {df.shape}")
    
    return df
