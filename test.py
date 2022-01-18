import pandas
import numpy as np

titanic = pandas.read_csv('data_all_920.csv')
print(titanic.head(10))

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

# predictors = ['sex', 'age', #'year',
#               '2jzr', '2jzl', '2fcr', '2fcl', '2jnr', '2jnl', '2sgr',
#               # '1jzr', '1jzl', '1fcr', '1fcl', '1jnr', '1jnl', '1sgr',
#               # 'ajzr', 'ajzl', 'afcr', 'afcl', 'ajnr', 'ajnl', 'asgr',
#               'rjzr', 'rjzl', 'rfcr', 'rfcl', 'rjnr', 'rjnl', 'rsgr']

#
# predictors = ['sex', 'age', #'year',
#               # '2jzr', '2jzl', '2fcr', '2fcl', '2sgr',
#               # 'ajzr', 'ajzl', 'afcr', 'afcl',  'asgr',
#               'rjzr', 'rjzl', 'rfcr', 'rfcl', 'rsgr'
#               ]



# # ---------- Linear regression --------------------
# alg = LinearRegression()
#
# kf = KFold(n_splits=3,shuffle=True, random_state=1)
#
#
# predictions = []
# for train, test in  kf.split(titanic):
#
#     train_predictors = titanic[predictors].iloc[train, :]
#
#     train_target = titanic['result'].iloc[train]
#
#     alg.fit(train_predictors, train_target)
#
#     test_predictions = alg.predict(titanic[predictors].iloc[test, :])
#     predictions.append(test_predictions)
#
# predictions = np.concatenate(predictions,axis=0)
#
# predictions[predictions < 1.5 ] = 1
# predictions[predictions > 4.5 ] = 5
# for i in range(0,920):
#     if predictions[i]>1.5 and predictions[i]<2.5:
#         predictions[i] = 2
#     if predictions[i]>2.5 and predictions[i]<3.5:
#         predictions[i] = 3
#     if predictions[i]>3.5 and predictions[i]<4.5:
#         predictions[i] = 4
#
#
# print(predictions)
#
# accuracy = sum(predictions==titanic['result'])/len(predictions)
#
# print(accuracy)


# ----------------logistic regression-----------------------
#
# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LogisticRegression
#
# alg = LogisticRegression(random_state=1)
# # Compute the accuracy score for all the cross validation folds,
# # (much simper than what we did before!)
# scores = cross_val_score(alg, titanic[predictors], titanic["result"], cv=3)
# # Take the mean of the scores (because we have one for each fold)
# print(scores.mean())
#
#
#
# ----------------------- Random forest ------------------------

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier


predictors = ['sex', 'age', #'year',
              # '2jzr', '2jzl', '2fcr', '2fcl',  '2sgr',
              # 'ajzr', 'ajzl', 'afcr', 'afcl',  'asgr',
              'rjzr', 'rjzl', 'rfcr', 'rfcl',  'rsgr'
              ]


# # Initialize our algorithm with the default paramters
# # n_estimators is the number of trees we want to make
# # min_samples_split is the minimum number of rows we need to make a split
# # min_samples_leaf is the minimum number of samples we can have at the place where a
# # tree branch(分支) ends (the bottom points of the tree)
alg = RandomForestClassifier(random_state=1,
                             n_estimators=100,
                             min_samples_split=4,
                             min_samples_leaf=2)
# # Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
kf = KFold(n_splits=3, shuffle=True, random_state=1)
scores = cross_val_score(alg, titanic[predictors], titanic["result"], cv=kf)

# # Take the mean of the scores (because we have one for each fold)
print(scores.mean())
