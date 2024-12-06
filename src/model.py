import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime

# Define features globally to ensure consistency
FEATURES = [
    'DEP_DELAY', 'DISTANCE', 'CANCELLED', 'DIVERTED',
    'DEP_HOUR', 'DAY_OF_WEEK', 'IS_WEEKEND', 'IS_MORNING', 
    'IS_EVENING', 'IS_SUMMER', 'IS_WINTER', 'MONTH', 'DAY'
]

def process_batch(batch):
    numeric_columns = ['DEP_DELAY', 'ARR_DELAY', 'CANCELLED', 'DIVERTED', 'DISTANCE', 'CRS_DEP_TIME']
    batch[numeric_columns] = batch[numeric_columns].fillna(0)
    
    batch['MONTH'] = pd.to_datetime(batch['FL_DATE']).dt.month
    batch['DAY'] = pd.to_datetime(batch['FL_DATE']).dt.day
    batch['DAY_OF_WEEK'] = pd.to_datetime(batch['FL_DATE']).dt.dayofweek
    
    batch['DEP_HOUR'] = batch['CRS_DEP_TIME'] // 100
    batch['IS_WEEKEND'] = (batch['DAY_OF_WEEK'] >= 5).astype(int)
    batch['IS_MORNING'] = ((batch['DEP_HOUR'] >= 6) & (batch['DEP_HOUR'] <= 12)).astype(int)
    batch['IS_EVENING'] = (batch['DEP_HOUR'] >= 18).astype(int)
    batch['IS_SUMMER'] = batch['MONTH'].isin([6, 7, 8]).astype(int)
    batch['IS_WINTER'] = batch['MONTH'].isin([12, 1, 2]).astype(int)
    
    batch['DELAYED'] = (batch['ARR_DELAY'] > 0).astype(int)
    
    return batch[FEATURES + ['DELAYED']]

def preprocess_data(df):
    df = pd.DataFrame(df)
    
    BATCH_SIZE = 100000
    if len(df) > BATCH_SIZE:
        dfs = []
        for start in range(0, len(df), BATCH_SIZE):
            end = start + BATCH_SIZE
            batch = df[start:end].copy()
            processed_batch = process_batch(batch)
            dfs.append(processed_batch)
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = process_batch(df)
    
    return df

def train_model(X_train, y_train):
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    rf_model.fit(X_train, y_train)
    return rf_model

def create_flight_delay_model(df):
    try:
        processed_df = preprocess_data(df)
        
        X = processed_df[FEATURES]
        y = processed_df['DELAYED']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = train_model(X_train_scaled, y_train)
        
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        train_accuracy = (train_pred == y_train).mean()
        test_accuracy = (test_pred == y_test).mean()
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Testing Accuracy: {test_accuracy:.4f}")
        
        model_data = {
            'model': model,
            'scaler': scaler,
            'features': FEATURES
        }
        
        with open('random_forest_delay_model.pkl', 'wb') as file:
            pickle.dump(model_data, file)
        
        return model
        
    except Exception as e:
        print(f"Error in model creation: {str(e)}")
        return None

def predict_delay(flight_data):
    try:
        for feature in FEATURES:
            if feature not in flight_data:
                raise ValueError(f"Missing feature: {feature}")
        
        with open('random_forest_delay_model.pkl', 'rb') as file:
            model_data = pickle.load(file)
        
        model = model_data['model']
        scaler = model_data['scaler']
        
        input_df = pd.DataFrame([flight_data])[FEATURES]
        input_scaled = scaler.transform(input_df)
        
        delay_probability = model.predict_proba(input_scaled)
        is_delayed = model.predict(input_scaled)
        
        return bool(is_delayed[0]), float(delay_probability[0][1])
    
    except Exception as e:
        print(f"Error in prediction setup: {str(e)}")
        return None, None

if __name__ == "__main__":
    try:
        print("Loading data...")
        df = pd.read_csv('flights.csv')
        
        print("\nTraining model...")
        model = create_flight_delay_model(df)
        
        if model is not None:
            print("\nTesting prediction...")
            sample_flight = {
                'DEP_DELAY': 10,
                'DISTANCE': 1000,
                'CANCELLED': 0,
                'DIVERTED': 0,
                'DEP_HOUR': 14,
                'DAY_OF_WEEK': 5,
                'IS_WEEKEND': 1,
                'IS_MORNING': 0,
                'IS_EVENING': 0,
                'IS_SUMMER': 1,
                'IS_WINTER': 0,
                'MONTH': 7,
                'DAY': 15
            }
            
            is_delayed, probability = predict_delay(sample_flight)
            
            if is_delayed is not None:
                print(f"\nPrediction Results:")
                print(f"Delay Predicted: {'Yes' if is_delayed else 'No'}")
                print(f"Delay Probability: {probability:.2%}")
                
                feature_importance = pd.DataFrame({
                    'feature': FEATURES,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print("\nFeature Importance:")
                print(feature_importance)
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")