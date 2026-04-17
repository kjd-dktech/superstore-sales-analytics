#========================================
# Utility functions
# Kodjo Jean DEGBEVI
#========================================

#========================================
#==== US State Abbreviations Utility ====
#----------------------------------------

def get_us_state_abbrev():
    return {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
        'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
        'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
        'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
        'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
        'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
        'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
        'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
        'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
        'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
        'District of Columbia': 'DC'
    }

#========================================
#===== Data Preprocessing Utilities =====
#----------------------------------------

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate(model_name, y_true_log, y_pred_log):
    y_true_dollar = np.sign(y_true_log) * np.expm1(np.abs(y_true_log))
    y_pred_dollar = np.sign(y_pred_log) * np.expm1(np.abs(y_pred_log))
    
    rmse = np.sqrt(mean_squared_error(y_true_dollar, y_pred_dollar))
    mae = mean_absolute_error(y_true_dollar, y_pred_dollar)
    r2 = r2_score(y_true_log, y_pred_log)
    
    print(f"[{model_name}] RMSE: {rmse:.2f} $ | MAE: {mae:.2f} $ | R²: {r2:.4f}")

def custom_dollar_rmse_func(y_true_log, y_pred_log):
    y_true_dollar = np.sign(y_true_log) * np.expm1(np.abs(y_true_log))
    y_pred_dollar = np.sign(y_pred_log) * np.expm1(np.abs(y_pred_log))
    return np.sqrt(mean_squared_error(y_true_dollar, y_pred_dollar))

#=================================================
#===== Hyperparameter Optimization Utilities =====
#-------------------------------------------------

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
import optuna
seed=42

def objective_rf(trial, X, y, scorer, cv=5):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
    }

    rf = RandomForestRegressor(**params, random_state=seed, n_jobs=-1)
    
    score = cross_val_score(rf, X, y, cv=cv, scoring=scorer).mean()
    return score

def objective_xgb(trial, X, y, scorer, cv=5):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1400),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.08, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'gamma': trial.suggest_float('gamma', 1e-8, 5.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'eval_metric': 'rmse'
    }
    
    xgb = XGBRegressor(**params, random_state=seed, n_jobs=-1, objective='reg:squarederror')
    score = cross_val_score(xgb, X, y, cv=cv, scoring=scorer).mean()
    
    return score