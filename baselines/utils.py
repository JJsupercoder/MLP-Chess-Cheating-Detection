import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
import shap
import matplotlib.pyplot as plt

feature_cols = ['depth_1', 'depth_2', 'depth_3', 'depth_4', 'depth_5',
    'depth_6', 'depth_7', 'depth_8', 'depth_9', 'depth_10', 'depth_11',
    'depth_12', 'depth_13', 'depth_14', 'depth_15', 'depth_16', 'depth_17',
    'depth_18', 'depth_19', 'depth_20', 'best_move_eval', 'player_rating',
    'sigmoid_eval_ratio']
feature_cols = ['depth_5',
    'depth_10', 'depth_15', 'depth_20', 'best_move_eval', 'player_rating',
    'sigmoid_eval_ratio']

feature_cols = ['depth_5',
    'depth_10', 'depth_15', 'depth_20', 'best_move_eval', 'best_minus_depth', 'player_rating']

def balance_data(df, target_column='label', samples_per_class=100000):
    """Balances a DataFrame by randomly undersampling the majority class."""

    df_majority = df[df[target_column] == 0.0]
    df_minority = df[df[target_column] == 1.0]

    df_majority_downsampled = resample(df_majority,
                                       replace=False,  # sample without replacement
                                       n_samples=samples_per_class,
                                       random_state=42)

    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    return df_balanced


def load_data():
    df = pd.read_csv("dataset/train_norm.txt", delimiter=",", low_memory=True)
    df = df.drop_duplicates()
    df["best_minus_depth"] = df["best_move_eval"] - df[feature_cols[:4]].mean(axis=1)
    # df = balance_data(df)
    X_train = df[feature_cols]
    y_train = df['label']

    df = pd.read_csv("dataset/test_norm.txt", delimiter=",", low_memory=True)
    df = df.drop_duplicates()
    df["best_minus_depth"] = df["best_move_eval"] - df[feature_cols[:4]].mean(axis=1)
    X_test = df[feature_cols]
    y_test = df['label']
    
    print("data loaded.")
    # return X_train, X_test, y_train, y_test
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # return X_train_scaled, X_test_scaled, y_train, y_test
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    return X_train_resampled, X_test_scaled, y_train_resampled, y_test


def evaluate(X_test, y_test, model):
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")
    print(f"Confusion Matrix:\n{confusion}")


def save_shap(X_train, X_test, model, save_path):
    background_sample_train = shap.sample(X_train, 100)
    background_sample_test = shap.sample(X_test, 100)
    try:predict_func = model.predict_proba
    except: predict_func = model.predict_log_proba
    explainer = shap.KernelExplainer(predict_func, background_sample_train)
    
    # Convert NumPy arrays to DataFrames with feature names
    shap_values = explainer.shap_values(background_sample_test, silent=True)
    shap.summary_plot(shap_values[:, :, 0], background_sample_test, show=False, feature_names=feature_cols)
    plt.savefig(save_path.replace(".png", "_0.png"))
    plt.close()
    shap.summary_plot(shap_values[:, :, 1], background_sample_test, show=False, feature_names=feature_cols)
    plt.savefig(save_path.replace(".png", "_1.png"))
    plt.close()