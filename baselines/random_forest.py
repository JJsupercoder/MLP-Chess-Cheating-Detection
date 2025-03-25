from sklearn.ensemble import RandomForestClassifier

from utils import evaluate, save_shap


def train(X_train, X_test, y_train, y_test):
    # Create and train the Random Forest Classifier model
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print("model trained.")

    print("evaluating test")
    evaluate(X_test, y_test, model)
    print("evaluating train")
    evaluate(X_train, y_train, model)
    print("evaluated")
    
    save_shap(X_train, X_test, model, f"plots_shap/{__name__}.png")
