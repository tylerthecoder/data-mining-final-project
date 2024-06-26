from load_data import process_data, set_age_group, x, y
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB as NaiveBayes
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import pandas as pd
from ndcg import ndcg, ndcg_scorer
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)

import sys
cross_val = len(sys.argv) > 1 and sys.argv[1] == "cross"

if not cross_val:
    model = XGBClassifier(
        tree_method="hist",
        early_stopping_rounds=5,
        n_estimators=200,
        # n_jobs=-1,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
else:
    model = XGBClassifier(
        tree_method="hist",
        colsample_bytree=0.8,
        objective= 'multi:softprob',
        eval_metric= 'ndcg',
        random_state=42,
    )

    search_space = {
        'n_estimators': [200],
        'max_depth': [3],
        "gamma" : [0.1],
        "learning_rate" : [1]
    }

    print("Grid searching...", search_space)
    model = GridSearchCV(
        model,
        search_space,
        verbose=100,
        scoring=ndcg_scorer,
    )

    model.fit(x, y)

    print(model.best_estimator_) # to get the complete details of the best model
    print(model.best_params_) # to get only the best hyperparameter values that we searched for

    # Assuming you have already performed the grid search and obtained the results
    df = pd.DataFrame(model.cv_results_)

    # Sort the DataFrame by the mean test score for the default scoring metric, typically 'mean_test_score'
    df = df.sort_values('rank_test_score')  # Sorting by rank which is directly related to the default score

    # Save the sorted DataFrame to a CSV file
    df.to_csv("cv_results.csv", index=False)

    print("Results sorted by default scoring (accuracy) and saved to 'cv_results.csv'.")


print("Model fitted, predicting...")
predictions = model.predict(X_test)
print("Predictions made, calculating metrics...")
report = classification_report(y_test, predictions)
print(report)
cm = confusion_matrix(y_test, predictions)
cm_df = pd.DataFrame(cm, model.classes_, model.classes_)
print(cm_df)

print("Predicting probs")
# (num_samples, num_classes)
prob_preds = model.predict_proba(X_test)

# Get the second most likely class
second_most_likely_indices = [np.argsort(preds)[-2] for preds in prob_preds]
print("Second guesses")
cm2 = confusion_matrix(y_test, second_most_likely_indices)
cm_df2 = pd.DataFrame(cm2, model.classes_, model.classes_)
print(cm_df2)

third = [np.argsort(preds)[-3] for preds in prob_preds]
print("Third guesses")
cm3 = confusion_matrix(y_test, third)
cm_df3 = pd.DataFrame(cm3, model.classes_, model.classes_)
print(cm_df3)

val = ndcg(prob_preds, y_test)
print("SCORE: ", val)





print("Predict on test data")
df_test = pd.read_csv('data/csv/test_users.csv')

ids = df_test['id']

df_test = process_data(df_test)
df_test = df_test.drop('is_weibo', axis=1)
test_preds = model.predict_proba(df_test)
test_preds = pd.DataFrame(test_preds)

# Join the predictions with the ids
test_preds['id'] = ids
test_preds.set_index('id', inplace=True)


print("Test preds:")
print(test_preds.head())

class_dict = {
    'NDF': 7,
    'US': 10,
    'other': 11,
    'FR': 4,
    'CA': 1,
    'GB': 5,
    'ES': 3,
    'IT': 6,
    'PT': 9,
    'NL': 8,
    'DE': 2,
    'AU': 0
}
inv_classes = {v: k for k, v in class_dict.items()}

def get_top(s):
    indexes = [i for i in range(0,12)]
    lst = list(zip(indexes, s))
    top_five = sorted(lst, key=lambda x: x[1])[::-1][:5]
    top_five = [inv_classes[i[0]] for i in top_five]
    return str(top_five)


test_preds['get_top'] = test_preds.apply(get_top, axis=1)


print("Test preds:")
print(test_preds.head())

import ast
test_preds['get_top'] = test_preds['get_top'].apply(lambda x: ast.literal_eval(x))

print("Test preds:")
print(test_preds.head())

s = test_preds.apply(lambda x: pd.Series(x['get_top']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'country'

print("S")
print(s.head())

submission = test_preds.drop([i for i in range(0,12)] + ['get_top'], axis=1).join(s)

print("Submission")
print(submission.head())

submission.to_csv('submission.csv')

