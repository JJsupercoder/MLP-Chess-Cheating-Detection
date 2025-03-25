from sklearn.neural_network import MLPClassifier

from utils import evaluate, save_shap


def train(X_train, X_test, y_train, y_test):
    # Create and train the MLP Classifier model
    model = MLPClassifier(random_state=42, max_iter=1000, early_stopping=True)  # Increase max_iter if needed
    model.fit(X_train, y_train)
    print("model trained.")

    print("evaluating test")
    evaluate(X_test, y_test, model)
    print("evaluating train")
    evaluate(X_train, y_train, model)
    print("evaluated")
    
    # save_shap(X_train, X_test, model, f"plots_shap/{__name__}.png")