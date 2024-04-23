import xgboost as xgb
from sklearn.model_selection import train_test_split

def train_xgb(X_train, y_train):
    X_train_part, X_val_part, y_train_part, y_val_part = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train_part, label=y_train_part)
    dval = xgb.DMatrix(X_val_part, label=y_val_part)

    params = {
        'max_depth': 6,
        'eta': 0.01,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': 100
    }

    model = xgb.train(params, dtrain, num_boost_round=2000, evals=[(dval, 'eval')], early_stopping_rounds=50, verbose_eval=100)
    model.save_model('xgb_model.model')
