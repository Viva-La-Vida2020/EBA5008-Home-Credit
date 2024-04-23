import numpy as np
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from sklearn.metrics import roc_auc_score
from train_lgbm import train_lgbm
from train_xgb import train_xgb
from train_mlp import train_mlp
from train_rnn import train_rnn

# X_train, X_test, y_train, y_test

# Define search space for each model
space_lgbm = [
    Integer(10, 100, name='n_estimators'),
    Integer(3, 30, name='max_depth'),
    Real(0.01, 0.2, name='learning_rate')
]

space_xgb = [
    Integer(10, 100, name='n_estimators'),
    Integer(3, 30, name='max_depth'),
    Real(0.01, 0.2, name='eta')
]

space_mlp = [
    Integer(32, 128, name='layer1_neurons'),
    Integer(16, 64, name='layer2_neurons')
]

space_rnn = [
    Integer(10, 100, name='units'),
    Integer(5, 20, name='n_layers')
]

# For LGBM
@use_named_args(space_lgbm)
def objective_lgbm(**params):
    model = train_lgbm(X_train, y_train, params)
    preds = model.predict(X_test)
    score = roc_auc_score(y_test, preds)
    return -score

gp_result = gp_minimize(objective_lgbm, space_lgbm, n_calls=10, random_state=42)

# Print the best parameters and save the model with those parameters
print("Best parameters:", gp_result.x)
model = train_lgbm(X_train, y_train, gp_result.x)
model.save_model('optimized_lgbm_model.txt')

# Evaluate on X_test, y_test and save predictions
preds_test = model.predict(X_test)
np.save('lgbm_test_predictions.npy', preds_test)
