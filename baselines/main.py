from adaboost import train as adaboost_train
from bagging import train as bagging_train
from decision_tree import train as dt_train
from extra_trees import train as et_train
from gradient_boost import train as gb_train
from linear_descrim import train as ld_train
from logistic_regression import train as lr_train
from naive_bayes import train as nb_train
from qd import train as qd_train
from random_forest import train as rf_train
from xgb import train as xg_train
from svm import train as svm_train
from light_gbm import train as lgbm_train
from simple_mlp import train as mlp_train
from sgd import train as sgd_train

from utils import load_data

X_train, X_test, y_train, y_test = load_data()

# print("-"*50, flush=True)
# print("Decision Tree", flush=True)
# dt_train(X_train, X_test, y_train, y_test)
# print("-"*50, flush=True)

# print("-"*50, flush=True)
# print("Naive Bayes", flush=True)
# nb_train(X_train, X_test, y_train, y_test)
# print("-"*50, flush=True)

# print("-"*50, flush=True)
# print("Quadratic Descriminant", flush=True)
# qd_train(X_train, X_test, y_train, y_test)
# print("-"*50, flush=True)

# print("-"*50, flush=True)
# print("Extra Trees Classifier", flush=True)
# et_train(X_train, X_test, y_train, y_test)
# print("-"*50, flush=True)

# print("-"*50, flush=True)
# print("Bagging Classifier", flush=True)
# bagging_train(X_train, X_test, y_train, y_test)
# print("-"*50, flush=True)

# print("-"*50, flush=True)
# print("Logistic Regression", flush=True)
# lr_train(X_train, X_test, y_train, y_test)
# print("-"*50, flush=True)

# print("-"*50, flush=True)
# print("Linear Descriminant", flush=True)
# ld_train(X_train, X_test, y_train, y_test)
# print("-"*50, flush=True)

# print("-"*50, flush=True)
# print("Random Forest", flush=True)
# rf_train(X_train, X_test, y_train, y_test)
# print("-"*50, flush=True)

# print("-"*50, flush=True)
# print("Adaboost", flush=True)
# adaboost_train(X_train, X_test, y_train, y_test)
# print("-"*50, flush=True)

# print("-"*50, flush=True)
# print("Gradient Boost", flush=True)
# gb_train(X_train, X_test, y_train, y_test)
# print("-"*50, flush=True)

# print("-"*50, flush=True)
# print("XGBoost", flush=True)
# xg_train(X_train, X_test, y_train, y_test)
# print("-"*50, flush=True)

# print("-"*50, flush=True)
# print("SGD", flush=True)
# sgd_train(X_train, X_test, y_train, y_test)
# print("-"*50, flush=True)

# print("-"*50, flush=True)
# print("LightGBM", flush=True)
# lgbm_train(X_train, X_test, y_train, y_test)
# print("-"*50, flush=True)

print("-"*50, flush=True)
print("Simple MLP", flush=True)
mlp_train(X_train, X_test, y_train, y_test)
print("-"*50, flush=True)

