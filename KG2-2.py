import numpy as np
import pandas as pd

#Here the input is a npy file containing index - embedding and a tsv file containing index - feature category name
#Relation is ignored because we only have one relation - causes

entity = np.load('/Users/yuchen/my_task/TransE_l2_mywork_23/mywork_TransE_l2_entity.npy')
indexid = pd.read_csv("/Users/yuchen/my_task/TransE_l2_mywork_23/entities.tsv",sep='\t', header = None)
relation = np.load('/Users/yuchen/my_task/TransE_l2_mywork_23/mywork_TransE_l2_relation.npy')
relationid = pd.read_csv("/Users/yuchen/my_task/TransE_l2_mywork_23/relations.tsv",sep='\t', header = None)

entity_embedding = pd.DataFrame(entity)

entity_embedding['name'] = indexid[1]

relation_embedding = pd.DataFrame(relation)
relation_embedding['name'] = relationid[1]

df = pd.read_csv('./Data/bank-additional/bank-additional-full.csv', sep=';')

result_df_sl = pd.read_csv('./imp2/result_df_sl_01.csv')

slrelation = ['education causes job', 'job causes marital', 'job causes poutcome','job causes housing',
              'job causes loan', 'job causes contact', 'education causes marital', 'education causes housing']

'''load the embedding to result_df_sl'''
#enlarge the result_df_sl, let it contain the emebeddings for each pair
def result_embedding (row, entity_embedding, relation_embedding):
    father_embedding = entity_embedding[entity_embedding['name']==row['father']].iloc[0,0:10]
    child_embedding = entity_embedding[entity_embedding['name']==row['child']].iloc[0,0:10]
    if row['weight']>0:
        relation_emb = relation_embedding.iloc[0,0:10]
    else:
        relation_emb = relation_embedding.iloc[1,0:10]
    embedding = pd.concat([father_embedding, relation_emb, child_embedding])
    embedding.reset_index(drop = True, inplace = True)
    return embedding

result_df_sl = pd.concat([result_df_sl, result_df_sl.apply(lambda row: result_embedding(row, entity_embedding, relation_embedding), axis =1)] , axis = 1)

'''Apply the embedding to df'''
def get_father_child_feature_name (tempdf):
    string = tempdf['relation'].unique()[0]
    father = string.split()[0]
    child = string.split()[2]
    return father, child


#write a function input is tempdf and row, output is its corresponind weights
def create_embeddings (tempdf, row):
    father, child = get_father_child_feature_name(tempdf)
    father_category = row[father]
    child_category = row[child]
    tempdf_father = father+'_'+father_category
    tempdf_child = child+'_'+child_category
#    if any((tempdf['father']==tempdf_father)&(tempdf['child']==tempdf_child)) is True:
#        weights = tempdf[(tempdf['father']==tempdf_father)&(tempdf['child']==tempdf_child)]['weight'].unique()[0]
#    else:
#        weights = 0 
#    
    try:
#        weights = tempdf[(tempdf['father']==tempdf_father)&(tempdf['child']==tempdf_child)]['weight'].unique()[0]
        embeddings = tempdf[(tempdf['father']==tempdf_father)&(tempdf['child']==tempdf_child)].iloc[0,4:]
    except IndexError:
#        weights = 0
        embeddings = pd.Series(np.zeros(30))
    return embeddings

#Write a function apply this to orginal dataset
def apply_embedding_to_df (embedding_df, df, tempdf):
    father, child = get_father_child_feature_name(tempdf)
    feature_name = father + ' causes ' + child
    name_list = [feature_name + ' %s' %s for s in range(30)]
    embedding_df[name_list] = df.apply(lambda row: create_embeddings(tempdf, row), axis=1)
    
# apply to all the relations
#input is relation list 
def add_embeddings_for_all (df, slrelation, result_df_sl):
    result_group = result_df_sl.groupby('relation')
    embedding_df = pd.DataFrame()
    for i in range(len(slrelation)):
        tempdf = result_group.get_group(slrelation[i])
        apply_embedding_to_df (embedding_df, df, tempdf)
    return embedding_df

embedding_df = add_embeddings_for_all (df, slrelation, result_df_sl)

embedding_df.to_csv(r'./imp2/embedding_df_imp2.csv', index = False)

#embedding_df = pd.read_csv('./embedding_df_imp1.csv')




#Test 

#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix, classification_report
#from sklearn.model_selection import KFold
#from sklearn.metrics import accuracy_score


#df.replace('unknown',np.nan,inplace = True)
#df.replace('nonexistent',np.nan,inplace = True)
#
#y = df['y']
#X = df.drop('y',axis = 1)
#
#X = pd.get_dummies(X, prefix_sep = '_', drop_first = True)
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0 ,test_size=0.3 )
#
#X_em = pd.concat([X,embedding_df], axis = 1)
#
#X_train_em, X_test_em, y_train_em, y_test_em = train_test_split(X_em, y, random_state=0 ,test_size=0.3 )
#
#clf = LogisticRegression(penalty='l2')
#clf.fit(X_train, y_train)
#
#prediction_tp = clf.predict_proba(X_test)
#prediction_t = clf.predict(X_test)
#
#print(confusion_matrix(y_test,prediction_t))
#print(classification_report(y_test,prediction_t))
#
#
#clf = LogisticRegression(penalty='l2')
#clf.fit(X_train_em, y_train_em)
#
#prediction_tp = clf.predict_proba(X_test_em)
#prediction_t = clf.predict(X_test_em)
#
#print(confusion_matrix(y_test_em,prediction_t))
#print(classification_report(y_test_em,prediction_t))



#df.replace('unknown',np.nan,inplace = True)
#df.replace('nonexistent',np.nan,inplace = True)
#
#y = df['y']
#X = df.drop('y',axis = 1)
#
#X = pd.get_dummies(X, prefix_sep = '_', drop_first = True)
#
#X_em = pd.concat([X,embedding_df], axis = 1)
#
#kf = KFold(n_splits = 10, random_state=200, shuffle = True)
##kf = KFold(n_splits = 10)
#
#acclist = list()
#acc_emlist = list()
#
#for i in list(range(10)):
#    for fold_, (train_index, test_index) in enumerate(kf.split(X)):
##    print('===============================================================================')
#        print("fold n°{}".format(fold_ + 1))
#        X_train, X_test = X.iloc[list(train_index)], X.iloc[list(test_index)]
#        y_train, y_test = y.iloc[list(train_index)], y.iloc[list(test_index)]
#        X_em_train, X_em_test = X_em.iloc[list(train_index)], X_em.iloc[list(test_index)]
#        y_em_train, y_em_test = y.iloc[list(train_index)], y.iloc[list(test_index)]
#        clf = LogisticRegression(penalty='l2', random_state = 0)
#        clf.fit(X_train, y_train)
#        prediction_tp = clf.predict_proba(X_test)
#        prediction_t = clf.predict(X_test)
#        acc = accuracy_score(y_test,prediction_t)
#        acclist.append(acc)
#    #    print(confusion_matrix(y_test,prediction_t))
#    #    print(classification_report(y_test,prediction_t))
#        print(acc)
#        print('---')
#        clf = LogisticRegression(penalty='l2', random_state = 42)
#        clf.fit(X_em_train, y_em_train)
#        prediction_tp = clf.predict_proba(X_em_test)
#        prediction_t = clf.predict(X_em_test)
#    #    print(confusion_matrix(y_em_test,prediction_t))
#    #    print(classification_report(y_em_test,prediction_t))
#        acc_em = accuracy_score(y_test,prediction_t)
#        acc_emlist.append(acc_em)
#        print(acc_em)
#    #    acc_diff = acc_em - acc
#    #    acclist.append(acc_diff)
#
#
#from scipy import stats
#ttest, pval = stats.ttest_rel(acc_emlist,acclist)
#
#data1 = pd.Series(acclist)
#data2 = pd.Series(acc_emlist)
#
#data1.to_csv(r'./imp2/result.csv', index = False)
#data2.to_csv(r'./imp2/result_em.csv', index = False)
#
#
#data21 = pd.read_csv('./imp2/result.csv')
#data22 = pd.read_csv('./imp2/result_em.csv')
#
#data21.describe()
#
#
#ttest, pval = stats.ttest_rel(data21,data22)



