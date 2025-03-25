from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier  # Base estimator

from utils import evaluate, save_shap


def train(X_train, X_test, y_train, y_test):
    # Create and train the AdaBoost Classifier model
    base_estimator = DecisionTreeClassifier(max_depth=2, random_state=42, class_weight='balanced') # A shallow decision tree is commonly used.
    model = AdaBoostClassifier(estimator=base_estimator, random_state=42)
    model.fit(X_train, y_train)
    print("model trained.")

    print("evaluating test")
    evaluate(X_test, y_test, model)
    print("evaluating train")
    evaluate(X_train, y_train, model)
    print("evaluated")
    
    save_shap(X_train, X_test, model, f"plots_shap/{__name__}.png")