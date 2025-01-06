# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import mlflow
import mlflow.sklearn
import warnings

warnings.filterwarnings('ignore')

# --- Load Data ---
print("Loading datasets...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Drop unnecessary columns
drop_columns = ['Id', 'LowQualFinSF', 'MiscVal', 'BsmtHalfBath', 'BsmtFinSF2']
train.drop(columns=drop_columns, inplace=True)
test.drop(columns=drop_columns, inplace=True)

# --- Handle Missing Data ---
print("Handling missing data...")
Data = pd.concat((train, test)).reset_index(drop=True)
Target = train['SalePrice']
Data.drop(['SalePrice'], axis=1, inplace=True)

# Fill missing values
Data['LotFrontage'].fillna(Data['LotFrontage'].median(), inplace=True)
garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
Data[garage_cols] = Data[garage_cols].fillna('None')

garage_num_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars']
Data[garage_num_cols] = Data[garage_num_cols].fillna(0)

bsmt_cols = ['BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtQual', 
             'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
Data[bsmt_cols[:4]] = Data[bsmt_cols[:4]].fillna(0)
Data[bsmt_cols[4:]] = Data[bsmt_cols[4:]].fillna('None')

Data['MasVnrType'].fillna('None', inplace=True)
Data['MasVnrArea'].fillna(0, inplace=True)
Data['Functional'].fillna('Typ', inplace=True)
Data['Electrical'].fillna(Data['Electrical'].mode()[0], inplace=True)
Data.drop('Utilities', axis=1, inplace=True)

# Feature Engineering
Data['TotalSF'] = Data['TotalBsmtSF'] + Data['1stFlrSF'] + Data['2ndFlrSF']

# --- Encoding ---
print("Encoding categorical features...")
Data = pd.get_dummies(Data, drop_first=True)

# --- Feature Selection ---
print("Splitting dataset...")
ntrain = train.shape[0]
train = Data[:ntrain]
test = Data[ntrain:]
X_train, X_val, y_train, y_val = train_test_split(train, Target.values, test_size=0.25, random_state=42)

# --- Define Models ---
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, verbosity=0),
    "LightGBM": LGBMRegressor(random_state=42),
    "CatBoost": CatBoostRegressor(random_state=42, verbose=0)
}

# --- Train and Evaluate Models with MLflow ---
print("Training and evaluating models...")
mlflow.set_experiment("House Price Prediction - Model Comparison")

results = []

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        
        # Log results in MLflow
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("RMSE", rmse)
        mlflow.sklearn.log_model(model, artifact_path="model")
        
        # Append results
        results.append((model_name, rmse))
        print(f"{model_name} - RMSE: {rmse:.4f}")


results_df = pd.DataFrame(results, columns=["Model", "RMSE"]).sort_values(by="RMSE")
print("\nModel Performance Comparison:")
print(results_df)

best_model_name = results_df.iloc[0]["Model"]
print(f"\nBest Model: {best_model_name}")
