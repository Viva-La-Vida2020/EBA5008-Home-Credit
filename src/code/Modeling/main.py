import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import load_model

# Load pre-trained models
lgbm_model = xgb.Booster()
lgbm_model.load_model('optimized_lgbm_model.txt')

xgb_model = xgb.Booster()
xgb_model.load_model('optimized_xgb_model.model')

mlp_model = load_model('mlp_model.h5')

rnn_model = load_model('rnn_model.h5')

# Function to generate predictions for stacking
def generate_predictions(X_test):
    dtest = xgb.DMatrix(X_test)
    
    preds_lgbm = lgbm_model.predict(dtest)
    preds_xgb = xgb_model.predict(dtest)
    preds_mlp = mlp_model.predict(X_test).flatten()  # MLP output needs flattening
    preds_rnn = rnn_model.predict(X_test).flatten()  # RNN output needs flattening
    
    # Stack predictions as new features
    stacked_features = np.column_stack((preds_lgbm, preds_xgb, preds_mlp, preds_rnn))
    return stacked_features

# X_test, y_test
stacked_features = generate_predictions(X_test)

# Train Meta Model (XGB)
meta_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3)
meta_model.fit(stacked_features, y_test)

# Optionally, evaluate the Meta Model
meta_predictions = meta_model.predict_proba(stacked_features)[:, 1]
auc_score = roc_auc_score(y_test, meta_predictions)
print(f"The AUC score of the Meta Model is: {auc_score}")

# Save the Meta Model
meta_model.save_model('meta_xgb_model.model')