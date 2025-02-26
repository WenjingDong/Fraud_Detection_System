import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import time
import joblib
import pickle


# Group feature transformation
def transform_features(df, onehot_features, label_features, transformations, train=False):
    # One-Hot encoding for gender and category
    for column in onehot_features:
        if train:
            one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoded_columns = [column  + "_" + x for x in df[column].unique()]
            df[encoded_columns] = one_hot_encoder.fit_transform(df[column].values.reshape(-1, 1))
            transformations[column] = one_hot_encoder
        else:
            encoded_columns = [column + "_" + x for x in transformations[column].categories_[0]]
            df[encoded_columns] =  transformations[column].transform(df[column].values.reshape(-1, 1))
    
    # Label Encoding 
    for column in label_features:
        if train:
            label_encoder = LabelEncoder()
            # Add unknown to deal with unseen labels
            categories = np.concatenate((df[column].values, np.array(['Unknown']))) 
            label_encoder.fit(categories)
            df[column + "_encoded"] = label_encoder.transform(df[column].values)
            transformations[column] = label_encoder
        else:
            categories =  set(transformations[column].classes_.tolist())
            df[column + "_encoded"] = [x if x  in categories else 'Unknown' for x in df[column].values]
            df[column + "_encoded"] = transformations[column].transform(df[column + "_encoded"].values)
#             features.append(columns + "_encoded")
            
    df['amt'] = np.log1p(df['amt'])
    
    # Extract user age
    df['dob'] = pd.to_datetime(df['dob'])
    df['year_of_birth'] = df['dob'].dt.year
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['year_of_trans'] = df['trans_date_trans_time'].dt.year
    df['age'] = df['year_of_trans'] - df['year_of_birth']
    
    # Transaction time encoding
    df["trans_year"] = df["trans_date_trans_time"].dt.year - 2019
    df["trans_month"] = df["trans_date_trans_time"].dt.month
    df["trans_day"] = df["trans_date_trans_time"].dt.day
    df["trans_hour"] = df["trans_date_trans_time"].dt.hour
    df["trans_minute"] = df["trans_date_trans_time"].dt.minute
    df["trans_second"] = df["trans_date_trans_time"].dt.second
    
    return df
    
def scale_features(df, scaled_features, scalers, train=False):
    # Scale features 
    for column in scaled_features:
        if train:
            scaler = StandardScaler()
            df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))
            scalers[column] = scaler
        else:
            df[column] = scalers[column].transform(df[column].values.reshape(-1, 1))
    return df

from sklearn.base import BaseEstimator, TransformerMixin
class FeatureTransformer(BaseEstimator, TransformerMixin):
    # Feature Engineering
    def __init__(self, transform_function, onehot_features, label_features, train_transformers):
        self.transform_function = transform_function
        self.onehot_features = onehot_features
        self.label_features = label_features
        self.train_transformers = train_transformers
        
    def fit(self, X):
        return self
    
    def transform(self, X):
        X = self.transform_function(X, self.onehot_features, self.label_features, self.train_transformers)
        return X

class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self,  scale_function, feature_scaler, scaled_features):
        self.scale_function = scale_function
        self.feature_scaler = feature_scaler
        self.scaled_features = scaled_features
    
    def fit(self, X):
        return self
    
    def transform(self, X):
        X = self.scale_function(X, self.scaled_features, self.feature_scaler)
        return X
    
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_use):
        self.features_to_use = features_to_use

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.features_to_use]