#best
import logging
import pandas as pd
import numpy as np
import json
import zipfile
import xgboost as xgb # Import XGBoost
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import numpy as np

def add_engineered_features(df):
    # Handle zero division
    df['guests_per_room'] = df['guests'] / df['rooms'].replace(0, np.nan)

    # Create a binary feature for "apartment" in name
    if "name" in df.columns:
        df["name"] = df["name"].astype(str).fillna("")
        df["is_studio"] = df["name"].str.lower().str.contains("studio|wifi|residence|cosy|courtyard|train").astype(int)
    
    # Generate interaction term between lat and lon
    if "lat" in df.columns and "lon" in df.columns:
        df["lat_lon_interaction"] = df["lat"] * df["lon"]
        city_center_lat, city_center_lon = df["lat"].mean(), df["lon"].mean()

    return df

def baseline():
    logging.info("Reading train and test files")
    train = pd.read_json("train.json", orient='records')
    test = pd.read_json("test.json", orient='records')
    train, valid = train_test_split(train, test_size=1/3, random_state=123)

    # Apply feature engineering
    train = add_engineered_features(train)
    valid = add_engineered_features(valid)
    test = add_engineered_features(test)

    preprocess = ColumnTransformer(
        transformers=[
        ("lat", Pipeline(steps=[('imputer', SimpleImputer(strategy='mean'))]), ["lat"]),
        ("lon", Pipeline(steps=[('imputer', SimpleImputer(strategy='mean'))]), ["lon"]),
        ("lat_lon_interaction", Pipeline(steps=[('imputer', SimpleImputer(strategy='mean'))]), ["lat_lon_interaction"]),
        ("min_nights", Pipeline(steps=[('imputer', SimpleImputer(strategy='mean'))]), ["min_nights"]),
        ("num_reviews", Pipeline(steps=[('imputer', SimpleImputer(strategy='mean'))]), ["num_reviews"]),
        ("guests", Pipeline(steps=[('imputer', SimpleImputer(strategy='mean'))]), ["guests"]),
        ("cancellation", Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ]), ["cancellation"]),
        ("guests_per_room", Pipeline(steps=[('imputer', SimpleImputer(strategy='mean'))]), ["guests_per_room"]),
        ("rating", Pipeline(steps=[('imputer', SimpleImputer(strategy='mean'))]), ["rating"]),
        ("bathrooms", Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent'))]), ["bathrooms"]),
        ("listing_type", Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), ["listing_type"]),
        ("room_type", Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), ["room_type"]),
        ("is_studio", Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent'))]), ["is_studio"]),
        
    ],
    remainder='drop'
    )

    dummy = make_pipeline(preprocess, DummyRegressor())
    xgb_model = make_pipeline(preprocess, xgb.XGBRegressor(
        n_estimators=285, learning_rate=0.1, max_depth=5, subsample=0.6, colsample_bytree=0.7,
        objective='reg:squarederror', random_state=123))

    label = 'revenue'
    for model_name, model in [("mean", dummy), ("xgboost", xgb_model)]:
        logging.info(f"Fitting model {model_name}")
        model.fit(train.drop([label], axis=1), np.log1p(train[label].values))

        for split_name, split in [("train", train), ("valid", valid)]:
            pred = np.expm1(model.predict(split.drop([label], axis=1)))
            mae = mean_absolute_error(split[label], pred)
            logging.info(f"{model_name} {split_name} {mae:.3f}")

    pred_test = np.expm1(xgb_model.predict(test))
    test[label] = pred_test
    predicted = test[['revenue']].to_dict(orient='records')

    with zipfile.ZipFile("baseline.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr("predicted.json", json.dumps(predicted, indent=2))

    r2 = r2_score(valid["revenue"], pred)
    rmse = mean_squared_error(valid["revenue"], pred)

    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    baseline()