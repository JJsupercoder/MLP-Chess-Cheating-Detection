import xgboost as xgb  # Import XGBoost

from utils import evaluate, save_shap


def train(X_train, X_test, y_train, y_test):
    model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    print("model trained.")

    print("evaluating test")
    evaluate(X_test, y_test, model)
    print("evaluating train")
    evaluate(X_train, y_train, model)
    print("evaluated")
    
    save_shap(X_train, X_test, model, f"plots_shap/{__name__}.png")


if __name__ == '__main__':
    train()