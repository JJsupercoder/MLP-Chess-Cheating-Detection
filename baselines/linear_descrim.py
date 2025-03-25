from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from utils import evaluate, save_shap


def train(X_train, X_test, y_train, y_test):
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)
    print("model trained.")

    print("evaluating test")
    evaluate(X_test, y_test, model) # Use scaled data
    print("evaluating train")
    evaluate(X_train, y_train, model) # Use scaled data
    print("evaluated")
    
    save_shap(X_train, X_test, model, f"plots_shap/{__name__}.png")
