from load_data import process_data, set_age_group, x, y
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB as NaiveBayes
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import pandas as pd
from ndcg import ndcg
import numpy as np



# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.75, stratify=y)


# X_train = x
# X_test = x
# y_train = y
# y_test = y

# print("Smoting...") 
# smote = SMOTE()
# X_train, y_train = smote.fit_resample(X_train, y_train)

# print("Scaling...")
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)

# model = RandomForestClassifier(n_jobs=-1)
# model = LogisticRegression(n_jobs=-1)
# model = NaiveBayes()
# model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=100000)


model = XGBClassifier(
    tree_method="hist",
    early_stopping_rounds=5,
    n_estimators=200,
    n_jobs=-1,
)

print(X_train)
print("Model created, fitting...")
# model.fit(X_train, y_train)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

print("Model fitted, predicting...")
predictions = model.predict(X_test)
print("Predictions made, calculating metrics...")
report = classification_report(y_test, predictions)
print(report)
# cm = confusion_matrix(y_test, predictions)
# cm_df = pd.DataFrame(cm, model.classes_, model.classes_)
# print(cm_df)

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

# df_test = pd.read_csv('data/test_users.csv')
# df_test = process_data(df_test)
#
# pred_prob = pd.DataFrame(pred_prob, index=df_test.index)






