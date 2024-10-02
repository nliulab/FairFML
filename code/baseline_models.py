import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, roc_curve
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, demographic_parity_ratio, equalized_odds_ratio

col_names = ['education_11th', 'education_12th', 'education_1st-4th',
             'education_5th-6th', 'education_7th-8th', 'education_9th',
             'education_Assoc-acdm', 'education_Assoc-voc', 'education_Bachelors',
             'education_Doctorate', 'education_HS-grad', 'education_Masters',
             'education_Preschool', 'education_Prof-school',
             'education_Some-college', 'marital.status_Married-AF-spouse',
             'marital.status_Married-civ-spouse',
             'marital.status_Married-spouse-absent', 'marital.status_Never-married',
             'marital.status_Separated', 'marital.status_Widowed',
             'race_Asian-Pac-Islander', 'race_Black', 'race_Other', 'race_White',
             'workclass_Local-gov', 'workclass_Private', 'workclass_Self-emp-inc',
             'workclass_Self-emp-not-inc', 'workclass_State-gov',
             'workclass_Without-pay']

def prepare_data(train_data, test_data):
    train_label, train_sensitive_feature = train_data['income.new'], train_data['gender'].map({'Male': 1, 'Female': 0})
    train_data = train_data[['workclass', 'education', 'marital.status', 'race']]
    train_data = pd.get_dummies(train_data, dtype='int', drop_first=True)
    feature_diff = set(col_names) - set(train_data.columns)
    if feature_diff:
        dummy_df = pd.DataFrame(data=np.zeros((train_data.shape[0], len(feature_diff))), columns=list(feature_diff),
                                index=train_data.index)
        train_data = pd.concat([train_data, dummy_df], axis=1)
    train_data = train_data.sort_index(axis=1)

    test_label, test_sensitive_feature = test_data['income.new'], test_data['gender'].map({'Male': 1, 'Female': 0})
    test_data = test_data[['workclass', 'education', 'marital.status', 'race']]
    test_data = pd.get_dummies(test_data, dtype='int', drop_first=True)
    feature_diff = set(col_names) - set(test_data.columns)
    if feature_diff:
        dummy_df = pd.DataFrame(data=np.zeros((test_data.shape[0], len(feature_diff))), columns=list(feature_diff),
                                index=test_data.index)
        test_data = pd.concat([test_data, dummy_df], axis=1)
    test_data = test_data.sort_index(axis=1)
    return train_data, test_data, train_label, test_label, train_sensitive_feature, test_sensitive_feature

def evaluate_model(model, test_data, test_label, test_sensitive_feature, model_type, site):
    pred_prob = model.predict_proba(test_data)[:, 1]
    fpr, tpr, thresholds = roc_curve(test_label, pred_prob)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    pred = (pred_prob >= optimal_threshold).astype('int')

    acc = accuracy_score(test_label, pred)
    auc = roc_auc_score(test_label, pred_prob)
    mse = mean_squared_error(test_label, pred_prob)
    dpd = demographic_parity_difference(y_true=test_label, y_pred=pred, sensitive_features=test_sensitive_feature)
    dpr = demographic_parity_ratio(y_true=test_label, y_pred=pred, sensitive_features=test_sensitive_feature)
    eod = equalized_odds_difference(y_true=test_label, y_pred=pred, sensitive_features=test_sensitive_feature)
    eor = equalized_odds_ratio(y_true=test_label, y_pred=pred, sensitive_features=test_sensitive_feature)
    if model_type == 'central' and site == 'all':
        print(f'Central model on all data: Accuracy {acc:.4f} AUC {auc:.4f} MSE {mse:.4f} DPD {dpd:.4f} DPR {dpr:.4f} EOD {eod:.4f} EOR {eor:.4f}')
    elif model_type == 'central':
        print(f'Central model on site {site}: Accuracy {acc:.4f} AUC {auc:.4f} MSE {mse:.4f} DPD {dpd:.4f} DPR {dpr:.4f} EOD {eod:.4f} EOR {eor:.4f}')
    elif model_type == 'local':
        print(f'Local model on site {site}: Accuracy {acc:.4f} AUC {auc:.4f} MSE {mse:.4f} DPD {dpd:.4f} DPR {dpr:.4f} EOD {eod:.4f} EOR {eor:.4f}')

def fit_local_model(site):
    train_data = pd.read_csv(f'../data/adult/C5/train_S{site}.csv', index_col=0)
    test_data = pd.read_csv(f'../data/adult/C5/tests_S{site}.csv', index_col=0)
    train_data, test_data, train_label, test_label, train_sensitive_feature, test_sensitive_feature = (
        prepare_data(train_data, test_data))

    model = LogisticRegression()
    model.fit(X=train_data, y=train_label)
    evaluate_model(model, test_data, test_label, test_sensitive_feature, model_type='local', site=site)

def fit_central_model():
    directory = "../data/adult/C5"
    train_data, test_data = pd.DataFrame(), pd.DataFrame()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                data = pd.read_csv(os.path.join(root, file), index_col=0).reset_index(drop=True)
            if "train" in file:
                train_data = pd.concat([train_data, data], axis=0)
            elif "test" in file:
                test_data = pd.concat([test_data, data], axis=0)

    train_data, test_data, train_label, test_label, train_sensitive_feature, test_sensitive_feature = (
        prepare_data(train_data, test_data))

    print("Central model on all available test data:")
    model = LogisticRegression(penalty=None, max_iter=1000)
    model.fit(X=train_data, y=train_label)
    evaluate_model(model, test_data, test_label, test_sensitive_feature, model_type='central', site='all')

    print("Central model performance on each client:")
    for site in range(1, 6):
        train_data = pd.DataFrame()
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".csv"):
                    data = pd.read_csv(os.path.join(root, file), index_col=0).reset_index(drop=True)
                if "train" in file:
                    train_data = pd.concat([train_data, data], axis=0)
        test_data = pd.read_csv(f'../data/adult/C5/tests_S{site}.csv', index_col=0)
        train_data, test_data, train_label, test_label, train_sensitive_feature, test_sensitive_feature = prepare_data(train_data, test_data)

        model = LogisticRegression(penalty=None, max_iter=1000)
        model.fit(X=train_data, y=train_label)
        evaluate_model(model, test_data, test_label, test_sensitive_feature, model_type='central', site=site)

print("Local model on each client:")
for i in range(1, 6):
    fit_local_model(i)
fit_central_model()

