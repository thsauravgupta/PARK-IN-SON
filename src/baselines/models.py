from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

def get_baseline_models():
    """Returns a dictionary of un-fitted baseline models."""
    return {
        'rf': RandomForestRegressor(n_estimators=200, random_state=42),
        'xgboost': XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42),
        'svm': SVR(C=1.0, kernel='rbf'),
        'ridge': Ridge(alpha=1.0, random_state=42),
        'mlp': MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    }
