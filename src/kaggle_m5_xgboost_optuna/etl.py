# Standardized ETL 
import polars as pl
def load_data(path: str = '../data/raw/' ) -> pl.DataFrame: 

    """ Load all CSV files into Polars DataFrames and return as a dictionary.   
    Args:
        path (str): The directory path where the CSV files are located.
    Returns:
        dict: A dictionary containing Polars DataFrames for each CSV file.
    """
    result_dict = {}
    calendar_pl = pl.read_csv(f'{path}calendar.csv')
    sales_train_evaluation_pl = pl.read_csv(f'{path}sales_train_evaluation.csv')
    sales_train_validation_pl = pl.read_csv(f'{path}sales_train_validation.csv')
    sell_prices_pl = pl.read_csv(f'{path}sell_prices.csv')
    sample_submission_pl = pl.read_csv(f'{path}sample_submission.csv')
    result_dict['calendar'] = calendar_pl
    result_dict['sales_train_evaluation'] = sales_train_evaluation_pl
    result_dict['sales_train_validation'] = sales_train_validation_pl
    result_dict['sell_prices'] = sell_prices_pl
    result_dict['sample_submission'] = sample_submission_pl

    return result_dict

def transform_data(sales_data : pl.DataFrame, calendar_pl: pl.DataFrame, sell_prices_pl: pl.DataFrame) -> pl.DataFrame:
    """ Transform the loaded data into a long format DataFrame suitable for modeling.
    Args:
        data_dict (dict): A dictionary containing Polars DataFrames for each CSV file.
    Returns:
        pl.DataFrame: A transformed Polars DataFrame in long format.
    """ 
    
    # Melt wide to long in Polars
    df_long = sales_data.melt(
    id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
    value_vars=[col for col in sales_data.columns if col.startswith('d_')],
    variable_name='day',
    value_name='sales'
    )
    # Join with calendar data
    df_long = df_long.join(calendar_pl, left_on='day', right_on='d', how='left')

    # Join with sell prices data
    df_long = df_long.join(sell_prices_pl, left_on=['store_id', 'item_id', 'wm_yr_wk'], right_on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

    # encoding the id, store_id, item_id as categorical variables to save memory
    df_long = df_long.with_columns([
    pl.col('store_id').cast(pl.Categorical),
    pl.col('item_id').cast(pl.Categorical),
    pl.col('wm_yr_wk').cast(pl.Categorical)
    ])

    # replace the missing values in the event names and event types by 'no_event'
    df_long = df_long.with_columns([
        pl.col('event_name_1').fill_null('no_event'),
        pl.col('event_name_2').fill_null('no_event'),
        pl.col('event_type_1').fill_null('no_event'),
        pl.col('event_type_2').fill_null('no_event')
    ])
    # encoding the different calendar data categorical features
    categorical_cols = ['event_name_1', 'event_name_2', 'event_type_1', 'event_type_2', 'wm_yr_wk', 'weekday', 'month', 'year']
    for col in categorical_cols:
        df_long = df_long.with_columns([
            pl.col(col).cast(pl.Categorical)
        ])  
    

    return df_long


def add_features(df_long: pl.DataFrame) -> pl.DataFrame:
    """ Add lag features, rolling means, and other relevant features to the DataFrame.
    Args:
        df_long (pl.DataFrame): The long format Polars DataFrame.
    Returns:
        pl.DataFrame: The DataFrame with added features.    
    """
    # Melt wide to long in Polars
    df_long = df_long.melt(
    id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
    value_vars=[col for col in df_long.columns if col.startswith('d_')],
    variable_name='day',
    value_name='sales'
    )

    # Create lag and rolling mean features in Polars
    df_long = df_long.with_columns([
    pl.col('sales').shift(1).over('id').alias('lag_1'),
    pl.col('sales').shift(2).over('id').alias('lag_2'),
    pl.col('sales').shift(3).over('id').alias('lag_3'),
    pl.col('sales').shift(4).over('id').alias('lag_4'),
    pl.col('sales').shift(5).over('id').alias('lag_5'),
    pl.col('sales').shift(6).over('id').alias('lag_6'),
    pl.col('sales').shift(7).over('id').alias('lag_7'),
    pl.col('sales').shift(14).over('id').alias('lag_14'),
    pl.col('sales').shift(21).over('id').alias('lag_21'),
    pl.col('sales').shift(28).over('id').alias('lag_28'),

    pl.col('sales').rolling_mean(7).over('id').alias('rolling_mean_7'),
    pl.col('sales').rolling_mean(14).over('id').alias('rolling_mean_14'),
    pl.col('sales').rolling_mean(28).over('id').alias('rolling_mean_28'),
    pl.col('sales').rolling_mean(56).over('id').alias('rolling_mean_56')
    ])
    # Adding the target variable : 
    # add the target variable 
    df_long = df_long.with_columns([
        pl.col('sales').shift(-28).over('id').alias('sales_plus_28')
    ])
    # add the target variable to the long clean 
    df_long_clean = df_long_clean.with_columns([
        pl.col('sales').shift(-28).over('id').alias('sales_plus_28')
    ])
    # dropping the rows with nulls in the target
    target_col = 'sales_plus_28'
    df_long_clean = df_long_clean.drop_nulls(subset=[target_col])
    return df_long_clean



       


def load_train_validation_data(path='../data/raw/'):
    """
    Load and process the entire sales, calendar, and price data for M5 forecasting.
    Args:
        path (str): The directory path where the CSV files are located.
    Returns:
        dict: Polars DataFrames for the training and validation sets.
    """
    import polars as pl


    sales_train_evaluation_pl = pl.read_csv(f'{path}/sales_train_evaluation.csv')
    sales_train_validation_pl = pl.read_csv(f'{path}/sales_train_validation.csv')
    calendar_pl = pl.read_csv(f'{path}/calendar.csv')
    sell_prices_pl = pl.read_csv(f'{path}/sell_prices.csv')

    def process_sales_df(sales_df):
        df_long = sales_df.melt(
            id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
            value_vars=[col for col in sales_df.columns if col.startswith('d_')],
            variable_name='day',
            value_name='sales'
        )
        df_long = df_long.with_columns([
            pl.col('sales').shift(1).over('id').alias('lag_1'),
            pl.col('sales').shift(7).over('id').alias('lag_7'),
            pl.col('sales').rolling_mean(7).over('id').alias('rolling_mean_7')
        ])
        # Join calendar
        cal = calendar_pl.with_columns([pl.col('d').alias('day')])
        df_long = df_long.join(cal, on='day', how='left')
        # Join prices
        sp = sell_prices_pl.with_columns([
            pl.col('store_id').cast(pl.Categorical),
            pl.col('item_id').cast(pl.Categorical),
            pl.col('wm_yr_wk').cast(pl.Categorical)
        ])
        df_long = df_long.with_columns([
            pl.col('store_id').cast(pl.Categorical),
            pl.col('item_id').cast(pl.Categorical),
            pl.col('wm_yr_wk').cast(pl.Categorical)
        ])
        df_long = df_long.join(sp, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
        # Add target
        df_long = df_long.with_columns([
            pl.col('sales').shift(-28).over('id').alias('sales_plus_28')
        ])
        # Clean
        feature_cols = ['lag_1', 'lag_7', 'rolling_mean_7']
        df_long_clean = df_long.drop_nulls(subset=feature_cols + ['sales_plus_28'])
        df_long_clean = df_long_clean.with_columns([
            pl.col('day').str.replace('d_', '').cast(pl.Int32).alias('day_num')
        ])
        return df_long_clean

    train_set = process_sales_df(sales_train_evaluation_pl).filter(pl.col('day_num') < 1914)
    validation_set = process_sales_df(sales_train_validation_pl).filter(pl.col('day_num') >= 1914)

    return {'train': train_set, 'validation': validation_set}


