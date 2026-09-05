import logging

from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

logger = logging.getLogger(__name__)


def get_baseline_models():
    """
    Returns the 12-model baseline suite as {name: un-fitted estimator}.

    Families covered: linear (linear/ridge/lasso/elastic_net), kernel (svm),
    instance-based (knn), bagging ensembles (random_forest/extra_trees),
    boosting ensembles (gradient_boosting/xgboost/lightgbm), and a neural
    network (mlp). Defaults are regularized for clinical-cohort sizes:
    depth limits, subsampling, L1/L2 penalties, and internal early stopping
    guard against overfitting; nothing here is aggressive enough to
    underfit a linear-to-moderately-nonlinear signal.

    xgboost / lightgbm are optional dependencies: if an import fails the
    model is skipped with a warning (AdaBoost fills in so the suite stays
    at 12), rather than crashing the whole pipeline.
    """
    models = {
        # ── Linear family ────────────────────────────────────────────
        'linear': LinearRegression(),
        'ridge': Ridge(alpha=1.0, random_state=42),
        'lasso': Lasso(alpha=0.1, max_iter=10000, random_state=42),
        'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000,
                                  random_state=42),
        # ── Kernel / instance-based ──────────────────────────────────
        'svm': SVR(C=1.0, kernel='rbf', gamma='scale'),
        'knn': KNeighborsRegressor(n_neighbors=10, weights='distance'),
        # ── Bagging ensembles ────────────────────────────────────────
        'random_forest': RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=3,
            max_features='sqrt', random_state=42, n_jobs=-1,
        ),
        'extra_trees': ExtraTreesRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=3,
            max_features='sqrt', random_state=42, n_jobs=-1,
        ),
        # ── Boosting ensembles ───────────────────────────────────────
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.8, random_state=42,
        ),
        # ── Neural network ───────────────────────────────────────────
        'mlp': MLPRegressor(
            hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=1000,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=15, random_state=42,
        ),
    }

    try:
        from xgboost import XGBRegressor
        models['xgboost'] = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=42, n_jobs=-1,
        )
    except ImportError:
        logger.warning("xgboost not installed — substituting AdaBoost to keep "
                       "the suite at 12 models.")
        models['adaboost'] = AdaBoostRegressor(
            n_estimators=200, learning_rate=0.05, random_state=42,
        )

    try:
        from lightgbm import LGBMRegressor
        models['lightgbm'] = LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbose=-1,
        )
    except ImportError:
        logger.warning("lightgbm not installed — substituting HistGradientBoosting "
                       "to keep the suite at 12 models.")
        from sklearn.ensemble import HistGradientBoostingRegressor
        models['hist_gradient_boosting'] = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=4,
            l2_regularization=1.0, random_state=42,
        )

    return models
