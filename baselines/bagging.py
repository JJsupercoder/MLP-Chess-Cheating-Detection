from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier  # Base estimator

from utils import evaluate, save_shap


def train(X_train, X_test, y_train, y_test):
    # Create and train the Bagging Classifier model
    base_estimator = DecisionTreeClassifier(random_state=42, class_weight='balanced')  # Choose a base estimator
    model = BaggingClassifier(estimator=base_estimator, random_state=42)
    model.fit(X_train, y_train)
    print("model trained.")

    print("evaluating test")
    evaluate(X_test, y_test, model)
    print("evaluating train")
    evaluate(X_train, y_train, model)
    print("evaluated")
    
    save_shap(X_train, X_test, model, f"plots_shap/{__name__}.png")