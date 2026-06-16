import pandas as pd
import numpy as np

# CMAPSS has no header row, so we define column names manually
# First 5 columns: engine ID, time cycle, and 3 operational settings
# # Next 21 columns: sensor measurements (temprature, pressure, speed, etc.)

COLUMNS = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + \
          [f'sensor{i}' for i in range(1, 22)]  # generates ['sensor1', 'sensor2', ..., 'sensor21']

def load_data(subset='FD001', data_dir='data/raw/CMaps'):
    # Read train files - reach row is one engine at one time cycle
    train = pd.read_csv(f'{data_dir}/train_{subset}.txt',
                        sep=r'\s+', #columns are white space seperated
                        header=None, # no header row in the file
                        names=COLUMNS) #apply manual column names
    
    # read test files - same format but runs are cutoff before failure
    test = pd.read_csv(f'{data_dir}/test_{subset}.txt',
                       sep=r'\s+', header = None, names=COLUMNS)
    
    # Read ground truth RUL for test engines(one value per engine)
    rul = pd.read_csv(f'{data_dir}/RUL_{subset}.txt',
                      header=None, names=['RUL'])

    return train, test, rul

def add_rul(df):
    # for training data, (last cycle = failure)
    # so RUL at any point = max_cycle - current_cycle

    # Step 1: find the last cycle for each engine(i.e when the engine failed)
    max_cycle = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycle.columns = ['engine_id', 'max_cycle']

    # Step 2: merge that failure cycle back into the main dataframe
    df = df.merge(max_cycle, on='engine_id')

    # Step 3: Calculate RUL = how many cycles left before failure
    # e.g. if engine failed at cycle 200 and we're at cycle 150, RUL = 50
    df['RUL'] = df['max_cycle'] - df['cycle']

    # Step 4: drop max_cycle column, we dont need it anymore
    df.drop(columns=['max_cycle'], inplace=True)

    return df

if __name__ == '__main__':
    # load all 3 files for FD001 subset
    train, test, rul = load_data()

    # Add RUL column to training data
    train = add_rul(train)

    #Print first 5 rows to sanity check
    print(train.head())

    #Print shape to confirm data loaded correctly
    print(f"Train shape: {train.shape}")