import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import joblib  # For saving scalers


def statistical_analysis(data, feature_name, data_type):
    # Statistical Tests
    shapiro_test = stats.shapiro(data)
    skew_test = stats.skew(data)
    kurtosis_test = stats.kurtosis(data)
    print(f"-------------{feature_name} {data_type}------------")
    print(f"Shapiro-Wilk Test: {shapiro_test}")
    print(f"Skewness: {skew_test}")
    print(f"Kurtosis: {kurtosis_test}")

    all_stats = {
        "feature_name": feature_name,
        "normalization_stage": data_type,
        "shapiro_statistic": shapiro_test.statistic,
        "shapiro_pvalue": shapiro_test.pvalue,
        "skewness": skew_test,
        "kurtosis": kurtosis_test,
        "skew_degree": "low" if -0.5 <= skew_test <= 0.5 else "medium" if -1 <= skew_test <= -0.5 or 0.5 <= skew_test <= 1 else "high",
        "kurtosis_distribution": "normal" if 2 <= kurtosis_test <= 4 else "platykurtic" if skew_test < 2 else "leptokurtic"
    }
    return all_stats



def normalize_skewed_data(data, feature_name, capping=True, is_train=True, config_dict=None):
    """Normalizes highly skewed data with outliers."""
    all_stats = []
    all_stats.append(statistical_analysis(data, feature_name, "Raw Data"))
    
    if is_train:
        cap_config = {} #dictionary to store capping config.
        # 1. Address extreme values (capping)
        if capping:
            percentile_99 = np.percentile(data, 99)
            percentile_1 = np.percentile(data, 1)
            capped_data = np.clip(data, percentile_1, percentile_99)
            all_stats.append(statistical_analysis(capped_data, feature_name, "Capped Data"))
            cap_config['percentile_99'] = percentile_99
            cap_config['percentile_1'] = percentile_1
        else:
            capped_data = data

        pos = capped_data.min() >= 0
        # 2. Arcsinh transformation, for skewness reduction
        if not pos:
            transformed_data = np.arcsinh(capped_data)
        else:
            transformed_data = np.log1p(capped_data)
        all_stats.append(statistical_analysis(transformed_data, feature_name, "Normalized Data"))

        # 3. Scale using RobustScaler or MinMaxScaler
        if not pos:
            scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            scaler = RobustScaler()
        normalized_data = scaler.fit_transform(transformed_data.values.reshape(-1, 1))
        normalized_data = normalized_data.reshape(-1)
        all_stats.append(statistical_analysis(normalized_data, feature_name, "Scaled Data"))
        if config_dict is not None:
            config_dict[feature_name] = {'scaler': scaler, 'capping': cap_config, 'transform': np.arcsinh if not pos else np.log1p}  # Save scaler and capping config
    else:
        # Use saved scaler and capping config for validation/test data
        cap_config = config_dict[feature_name]['capping']
        scaler = config_dict[feature_name]['scaler']
        transform = config_dict[feature_name]['transform']
        
        if capping:
            capped_data = np.clip(data, cap_config['percentile_1'], cap_config['percentile_99'])
            all_stats.append(statistical_analysis(capped_data, feature_name, "Capped Data"))
        else:
            capped_data = data
            
        transformed_data = transform(capped_data)
        all_stats.append(statistical_analysis(transformed_data, feature_name, "Normalized Data"))
            
        normalized_data = scaler.transform(transformed_data.values.reshape(-1, 1))
        normalized_data = normalized_data.reshape(-1)
        all_stats.append(statistical_analysis(normalized_data, feature_name, "Scaled Data"))

    return normalized_data, all_stats


def analyze_feature(data, feature_name, norm=False):
    """Analyzes a single feature for distribution and outliers."""
    print(f"Analysis for feature: {feature_name}")

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    sns.histplot(data, kde=True)
    plt.title(f"Histogram of {feature_name}")

    plt.subplot(2, 2, 2)
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot of {feature_name}")

    plt.subplot(2, 2, 3)
    sns.boxplot(x=data)
    plt.title(f"Box Plot of {feature_name}")

    plt.savefig(f"plots/{feature_name}_{'norm' if norm else 'raw'}.png")
    plt.close()

def prep_data(col_names, features_to_norm, is_train, loaded_configs, data_path, stats_path, norm_path):
    records = []
    df = pd.read_csv(data_path, names=col_names, delimiter=",", low_memory=True)
    for feature in features_to_norm:
        data = df[feature]
        # analyze_feature(data, feature)
        normalized_data, all_stats = normalize_skewed_data(data, feature, capping=feature != "player_rating", is_train=is_train, config_dict=loaded_configs)
        # analyze_feature(normalized_data, feature, norm=True)
        records += all_stats
        df[feature] = normalized_data
    stats_df = pd.DataFrame(records)
    stats_df.to_csv(stats_path, index=False)
    df.to_csv(norm_path, index=False)

if __name__ == '__main__':
    col_names = [
        "depth_1", "depth_2", "depth_3", "depth_4", "depth_5",
        "depth_6", "depth_7", "depth_8", "depth_9", "depth_10",
        "depth_11", "depth_12", "depth_13", "depth_14", "depth_15",
        "depth_16", "depth_17", "depth_18", "depth_19", "depth_20",
        "best_move_eval", "player_rating", "label", "sigmoid_eval_ratio",
    ]
    features_to_norm = col_names.copy()
    features_to_norm.remove("label")
    features_to_norm.remove("player_rating")
    print(f"Normalizing following features: {features_to_norm}")
    print(f"All columns: {col_names}")

    config_dict = {}
    prep_data(col_names, features_to_norm, True, config_dict,
              "dataset/train.txt", "dataset/stats_train.csv", 
              "dataset/train_norm.txt")
    # # Save configs
    # joblib.dump(config_dict, 'dataset/preprocessing_config.joblib')

    # # loading and using the configs on val/test data
    # loaded_configs = joblib.load('dataset/preprocessing_config.joblib')

    # prep_data(col_names, features_to_norm, False, loaded_configs, "dataset/val.txt", "dataset/stats_val.csv", "dataset/val_norm.txt")
    # prep_data(col_names, features_to_norm, False, loaded_configs, "dataset/test.txt", "dataset/stats_test.csv", "dataset/test_norm.txt")
