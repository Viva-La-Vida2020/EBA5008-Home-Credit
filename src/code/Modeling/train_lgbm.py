import lightgbm as lgb
from sklearn.model_selection import train_test_split

def train_lgbm(X_train, y_train):
    X_train_part, X_val_part, y_train_part, y_val_part = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    train_data = lgb.Dataset(X_train_part, label=y_train_part)
    val_data = lgb.Dataset(X_val_part, label=y_val_part, reference=train_data)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'n_estimators': 100,
        'max_bin': 255,
        'verbose': -1
    }

    model = lgb.train(params, train_data, valid_sets=[train_data, val_data], num_boost_round=2000, early_stopping_rounds=50, verbose_eval=100)
    model.save_model('lgbm_model.txt')

# Example usage:
# train_lgbm(X_train, y_train)
