import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def get_data(file_path):
    df = pd.read_csv(file_path)
    return df

def get_features(df):
    df_new = df[['MachineID','ModelID','YearMade','saledate','SalePrice']].copy()
    mode = df_new.YearMade[df_new.YearMade > 1900].mode()
    df_new.loc[df_new.YearMade <= 1900, 'YearMade'] = mode
    df_new['saledate_converted'] = pd.to_datetime(df_new.saledate)
    df_new['equipment_age'] = df_new.saledate_converted.dt.year - df_new.YearMade
    df_new = df_new[['MachineID', 'ModelID', 'equipment_age', 'SalePrice']]
    df_new.dropna(inplace=True)
    return df_new

def split_data(df_new):
    feature_cols = ['MachineID', 'ModelID', 'equipment_age']
    X = df_new[feature_cols]
    y = df_new.SalePrice
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

def run_model(model, X_train, y_train):
    r2_score_train = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1).mean()
    return r2_score_train

def test_model(model, X_test, y_test):
    r2_score_test = model.score(X_test, y_test)
    return r2_score_test

if __name__ == '__main__':
    df = get_data('../data/train.csv')
    df_new = get_features(df)
    X_train, X_test, y_train, y_test = split_data(df_new)
    model = RandomForestRegressor(n_estimators=50)
    r2_score_train = run_model(model, X_train, y_train)
    print 'training_r2_score =', r2_score_train
    model.fit(X_train, y_train)
    r2_score_test = test_model(model, X_test, y_test)
    print 'test_r2_score =', r2_score_test
