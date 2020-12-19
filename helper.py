import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def load_data():
    headers = pd.read_csv('data/field_names.txt', header=None)
    data = pd.read_csv('data/breast-cancer.csv', names=headers.values[:,0])
    return data

def bootstrap_df(df, n_samples):
    return df.sample(n_samples, replace=True, random_state=2)

def select_features(df, y, num_features=3):
    selector = SelectKBest(chi2, k=num_features)
    features = selector.fit_transform(df, y)
    column_mask = selector.get_support()
    selected_features = df.columns[column_mask]
    return selected_features