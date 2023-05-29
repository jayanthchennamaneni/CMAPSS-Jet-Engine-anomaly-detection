import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data():
    column_names = ['unit_number', 'time', 'operational_setting_1', 'operational_setting_2', 'operational_setting_3'] + ['sensor_measurement_{}'.format(i) for i in range(1, 22)]

    df = pd.read_csv(r'./data/train_FD004.txt', delimiter=' ', header=None)

    df.drop(df.columns[[26, 27]], axis=1, inplace=True)  # drop the last two columns, which are NaNs
    df.columns = column_names

    cols_to_consider = [col for col in df.columns if col.startswith('sensor')]

    for col in cols_to_consider:
        df[col] = df[col].rolling(window=10, min_periods=1).mean()

    df['RUL'] = df.groupby(['unit_number'])['time'].transform(max) - df['time']

    features = df.drop(['RUL'], axis=1)
    target = df['RUL']

    X_train, x_val, Y_train, y_val = train_test_split(features, target, test_size=0.25, random_state=17)

    return X_train, x_val, Y_train, y_val


if __name__ == '__main__':

    X_train, x_val, Y_train, y_val = preprocess_data()
    print(X_train.shape, x_val.shape, Y_train.shape, y_val.shape)
