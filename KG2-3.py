
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

#240columns for 8 connections
embedding_df = pd.read_csv('./imp2/embedding_df_imp2.csv')

df = pd.read_csv('./Data/bank-additional/bank-additional-full.csv', sep=';')


df.replace('unknown',np.nan,inplace = True)
df.replace('nonexistent',np.nan,inplace = True)

y = df['y']
X = df.drop('y',axis = 1)

X = pd.get_dummies(X, prefix_sep = '_', drop_first = True)

X_em = pd.concat([X,embedding_df], axis = 1)

kf = KFold(n_splits = 10, random_state=200, shuffle = True)
#kf = KFold(n_splits = 10)

#Get the 100 pairs measurements for accuracy
acclist = list()
acc_emlist = list()

for i in list(range(10)):
    for fold_, (train_index, test_index) in enumerate(kf.split(X)):
#    print('===============================================================================')
        print("fold n°{}".format(fold_ + 1))
        X_train, X_test = X.iloc[list(train_index)], X.iloc[list(test_index)]
        y_train, y_test = y.iloc[list(train_index)], y.iloc[list(test_index)]
        X_em_train, X_em_test = X_em.iloc[list(train_index)], X_em.iloc[list(test_index)]
        y_em_train, y_em_test = y.iloc[list(train_index)], y.iloc[list(test_index)]
        clf = LogisticRegression(penalty='l2', random_state = 0)
        clf.fit(X_train, y_train)
        prediction_tp = clf.predict_proba(X_test)
        prediction_t = clf.predict(X_test)
        acc = accuracy_score(y_test,prediction_t)
        acclist.append(acc)
    #    print(confusion_matrix(y_test,prediction_t))
    #    print(classification_report(y_test,prediction_t))
        print(acc)
        print('---')
        clf = LogisticRegression(penalty='l2', random_state = 42)
        clf.fit(X_em_train, y_em_train)
        prediction_tp = clf.predict_proba(X_em_test)
        prediction_t = clf.predict(X_em_test)
    #    print(confusion_matrix(y_em_test,prediction_t))
    #    print(classification_report(y_em_test,prediction_t))
        acc_em = accuracy_score(y_test,prediction_t)
        acc_emlist.append(acc_em)
        print(acc_em)
    #    acc_diff = acc_em - acc
    #    acclist.append(acc_diff)


#Hyphothesis Test
from scipy import stats

#Use this if you self generate the result
#ttest, pval = stats.ttest_rel(acc_emlist,acclist)
#
#data1 = pd.Series(acclist)
#data2 = pd.Series(acc_emlist)
#
#data1.to_csv(r'./imp2/result.csv', index = False)
#data2.to_csv(r'./imp2/result_em.csv', index = False)

#Use this if you use the saved data
measurements = pd.read_csv('./imp2/result.csv', header = None)
measurements_em = pd.read_csv('./imp2/result_em.csv', header = None)

#measurements.describe()
#measurements_em.describe()

ttest, pval = stats.ttest_rel(measurements_em,measurements)



