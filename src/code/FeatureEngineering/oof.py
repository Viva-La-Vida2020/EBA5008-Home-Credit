import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Load data here
# Example:
# y_train = train['TARGET'].ravel()
# X_train = train.drop(['TARGET'], axis=1).values

n_folds = 5  # Number of folds for cross-validation
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Prepare to collect the OOF predictions
oof_preds_rf = np.zeros(X_train.shape[0])
oof_preds_lr = np.zeros(X_train.shape[0])
oof_preds_ridge = np.zeros(X_train.shape[0])
oof_preds_svm = np.zeros(X_train.shape[0])
oof_preds_ffnn = np.zeros(X_train.shape[0])

def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_dim=X_train.shape[1]),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Wrap Keras model so it can be used by scikit-learn
ffnn_model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32, verbose=0)

# Iterate through each fold
for train_index, test_index in skf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[test_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_fold, y_train_fold)
    oof_preds_rf[test_index] = rf.predict_proba(X_val_fold)[:, 1]

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_fold, y_train_fold)
    oof_preds_lr[test_index] = lr.predict_proba(X_val_fold)[:, 1]
    
    # Ridge Regression
    ridge = RidgeClassifier(random_state=42)
    ridge.fit(X_train_fold, y_train_fold)
    oof_preds_ridge[test_index] = ridge.decision_function(X_val_fold)
    
    # SVM
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train_fold, y_train_fold)
    oof_preds_svm[test_index] = svm.predict_proba(X_val_fold)[:, 1]
    
    # FFNN
    ffnn_model.fit(X_train_fold, y_train_fold)
    oof_preds_ffnn[test_index] = ffnn_model.predict_proba(X_val_fold)[:, 0]

# Now you can use oof_preds_* as new features your next layer model
