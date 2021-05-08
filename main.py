import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')

# ... fill or drop missing values ...
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop('Cabin', inplace=True, axis=1)
df.drop('Name', inplace=True, axis=1)

test_df['Age'].fillna(test_df['Age'].mean(), inplace=True)
test_df['Fare'].fillna(test_df['Fare'].mean(), inplace=True)
test_df.drop('Cabin', inplace=True, axis=1)
test_df.drop('Name', inplace=True, axis=1)

# ... transform string values ...
trans_df = pd.get_dummies(df, columns=['Sex', 'Ticket', 'Embarked'])
trans_test_df = pd.get_dummies(test_df, columns=['Sex', 'Ticket', 'Embarked'])

# ... to check model choice ...
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# x = trans_df.drop('Survived', axis=1)
# y = trans_df.Survived
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# rfc = RandomForestClassifier(max_leaf_nodes=1000)
# rfc.fit(x_train, y_train)
# prediction = rfc.predict(x_test)
# accuracy_score(y_test, prediction)  # 0.8324022346368715

# ... concatenate test and train data ...
train_len = trans_df.shape[0]
final_df = pd.concat([trans_df, trans_test_df])

df_train = final_df.iloc[:train_len]
df_train = df_train.fillna(0)

df_test = final_df.iloc[train_len:]
df_test.drop('Survived', axis=1, inplace=True)
df_test = df_test.fillna(0)

x_train = df_train.drop('Survived', axis=1)
y_train = df_train.Survived

x_test = df_test

# ... train model ...
# RANDOM FOREST CLASSIFIER

rfc = RandomForestClassifier(max_leaf_nodes=1000)
rfc.fit(x_train, y_train)
prediction = rfc.predict(x_test)

# ... write results in file ...

pred_df = pd.DataFrame(prediction, dtype='integer')
sample_sub = pd.read_csv('dataset/sample_submission.csv')
dataset = pd.concat([sample_sub['PassengerId'], pred_df], axis=1)
dataset.columns = ['PassengerId', 'Survived']
dataset.to_csv('submission.csv', index=False)

# --------------------------------------
# GRADIENT BOOSTING CLASSIFIER
# from sklearn.ensemble import GradientBoostingClassifier
# gbc = GradientBoostingClassifier(max_depth=10, n_estimators=200)
# gbc.fit(x_train, y_train)
# prediction = gbc.predict(x_test)


