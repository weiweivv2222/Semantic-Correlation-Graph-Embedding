
import pandas as pd
import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


'''load the data'''
df = pd.read_csv('./Data/bank-additional/bank-additional-full.csv', sep=';')

'''Functions to run logistic regression and get the triplet table'''
#write a function, input is two feature names, output is two dataframe 
def preprocessing (df, father, child):
    ndf = df.loc[:,[father,child]]
    ndf.replace('unknown',np.nan,inplace = True)
#    ndf.replace('nonexistent',np.nan,inplace = True)
    ndf.dropna(inplace = True)
    father_df = pd.get_dummies(ndf[father],prefix=father)
    child_df = pd.get_dummies(ndf[child],prefix=child)
    return father_df, child_df

#write a function, input is a feature category name(one of the father, such as edu_4y), and a feature name (such as job)
#out put is a list(number of child) of list (father child weight)
def child_lr(father_df, child_df, father_category):
#    import sklearn.linear_model as lm
    log_model = lm.LogisticRegression(random_state = 0)
    log_model.fit(child_df, father_df[father_category])
    child_list = []
    for i in range(len(child_df.columns)):
        child_list.append([father_category, child_df.columns[i],log_model.coef_[0][i]])
#        if abs(log_model.coef_[0][i])>0.2:
#            child_list.append([father_category, child_df.columns[i],log_model.coef_[0][i]])
    return child_list
    
def father_lr(df, father, child):
    father_df, child_df = preprocessing(df, father, child)
    father_list = []
    for father_category in father_df.columns:
        child_list = child_lr(father_df, child_df, father_category)
#        father_list.append(child_list)
        father_list = father_list+child_list
    return father_list

'''Apply functions and get the triplet table'''
result_df = pd.DataFrame()
for father in df.select_dtypes(include = ['object','category']).drop(['y','default','month','day_of_week'],axis = 1).columns:
    for child in df.select_dtypes(include = ['object','category']).drop(['y','default','month','day_of_week'],axis = 1).columns:
        if not child == father:
            result_list = father_lr(df, father, child)
            resultdf = pd.DataFrame(result_list,columns = ['father','child','weight'])
#            resultdf['relation'] ='\'' + father + '\''+ ',\'causes\',' + '\''+child+'\''
            resultdf['relation'] = father + ' causes '+child
            result_df = pd.concat([result_df,resultdf])
result_df.reset_index(drop = True, inplace = True)

#result_df.describe()

#result_df['relation'].value_counts()
#208

#slrelation = ['education causes job','job causes marital','job causes poutcome',
#              'education causes poutcome','job causes contact','job causes loan','education causes marital'
#             'job causes housing','education causes housing','education causes contact']
'''Filter the triplet table, remove sementically unrelated relations'''
#select those have semetic meanings
slrelation = ['education causes job', 'job causes marital', 'job causes poutcome','job causes housing',
              'job causes loan', 'job causes contact', 'education causes marital', 'education causes housing']

#slrelation = ['\'education\',\'causes\',\'job\'', '\'marital\',\'causes\',\'job\'', '\'job\',\'causes\',\'poutcome\'', 
#              '\'housing\',\'causes\',\'poutcome\'', '\'contact\',\'causes\',\'poutcome\'']

#Our example has 28 nodes 58 edges(triplets) and 1 relations
result_df_sl = result_df[result_df['relation'].isin(slrelation)]

result_df_sl.to_csv(r'./imp2/result_df_sl_01.csv', index = False)
#
#result_group = result_df_sl.groupby('relation')
#
#tempdf = result_group.get_group(slrelation[0])
#
#tempdf.head()

#write a function, input this df and create a mapping

#first write a function return father child name


def afunction(row):
    fathername = row['father']
    childname = row['child']
    if row['weight'] > 0:
        triplet = fathername + '\t'+ 'causes'+ '\t' + childname + '\t' + str(row['weight'])
    else:
        triplet = fathername + '\t'+ 'prevents'+ '\t' + childname + '\t' + str(abs(row['weight']))
    return triplet

result_df_sl['triplet'] = result_df_sl.apply(lambda row: afunction(row), axis = 1)

'''Get the input for embedding'''
def create_embedding_input(slrelation,df):
    embeddinginput = pd.DataFrame()
    for item in slrelation:
        embedding = df[(df[item+' weight']>0)|(df[item+' weight']<0)][item]
        embeddinginput = pd.concat([embeddinginput,embedding])
    return embeddinginput

embeddinginput = create_embedding_input(slrelation,df)

result_df_sl['triplet'].to_csv(r'./imp2/dgl_triplets_02.txt', header=None, index=None, sep=' ', mode='a')

result_df_sl.drop(columns = ['triplet'], inplace = True)







'''
It takes time, several minutes
These function create two feature for each relation over each instance, the triplet and the triplet weights, 
by apply filtered triplet table on the fulldata.

'''

def get_father_child_feature_name (tempdf):
    string = tempdf['relation'].unique()[0]
    father = string.split()[0]
    child = string.split()[2]
    return father, child

#tempdf = result_df_sl.groupby('relation').get_group('job causes contact')
#row = df.loc[1100,:]
#aaa = create_triplet_and_weights (tempdf, row)
#np.array([triplet,weights])
def create_triplet_and_weights (tempdf, row):
    father, child = get_father_child_feature_name(tempdf)
    father_category = row[father]
    child_category = row[child]
    tempdf_father = father+'_'+father_category
    tempdf_child = child+'_'+child_category    
    try:
        weights = tempdf[(tempdf['father']==tempdf_father)&(tempdf['child']==tempdf_child)]['weight'].unique()[0]
        if weights > 0:
            triplet =father+'_'+str(row[father]) +'\t'+ 'causes'+ '\t'+child+'_'+str(row[child])
        else:
            triplet =father+'_'+str(row[father]) +'\t'+ 'prevents'+ '\t'+child+'_'+str(row[child])
    except IndexError:
        weights = 0
        triplet = father+'_'+str(row[father]) +'\t'+ 'causes'+ '\t'+child+'_'+str(row[child])
#    array = np.array([triplet,weights])
    array = pd.Series([triplet,weights])
    return array
#    return triplet

def apply_triplet_weights_df (df,tempdf):
    father, child = get_father_child_feature_name(tempdf)
    feature_name = father + ' causes ' + child
    weight_name = father + ' causes ' + child + ' weight'
    df[[feature_name,weight_name]] = df.apply(lambda row: create_triplet_and_weights(tempdf, row), axis=1)
#    print(df.apply(lambda row: create_triplet_and_weights(tempdf, row), axis=1))

def add_two_feature_for_relation (df, slrelation, result_df_sl):
    result_group = result_df_sl.groupby('relation')
    for i in range(len(slrelation)):
        tempdf = result_group.get_group(slrelation[i])
        apply_triplet_weights_df(df,tempdf)
#        apply_weight_to_df (df, tempdf)

add_two_feature_for_relation (df, slrelation, result_df_sl)

## write a function input is father child name and the row, output is a new feature "father category causes child category"
#def create_triplet (tempdf, row):
#    father, child = get_father_child_feature_name(tempdf)
#    triplet =father+'_'+str(row[father]) +'\t'+ 'causes'+ '\t'+child+'_'+str(row[child])
#    return triplet
#
##write a function apply this two orginal dataset
#def apply_to_df (df,tempdf):
#    father, child = get_father_child_feature_name(tempdf)
#    feature_name = father + ' causes ' + child
#    df[feature_name] = df.apply(lambda row: create_triplet(tempdf, row), axis=1)
#
##write a function input is tempdf and row, output is its corresponind weights
#def create_triplet_weights (tempdf, row):
#    father, child = get_father_child_feature_name(tempdf)
#    father_category = row[father]
#    child_category = row[child]
#    tempdf_father = father+'_'+father_category
#    tempdf_child = child+'_'+child_category
##    if any((tempdf['father']==tempdf_father)&(tempdf['child']==tempdf_child)) is True:
##        weights = tempdf[(tempdf['father']==tempdf_father)&(tempdf['child']==tempdf_child)]['weight'].unique()[0]
##    else:
##        weights = 0 
##    
#    try:
#        weights = tempdf[(tempdf['father']==tempdf_father)&(tempdf['child']==tempdf_child)]['weight'].unique()[0]
#    except IndexError:
#        weights = 0
#    return weights
#
##Write a function apply this to orginal dataset
#def apply_weight_to_df (df, tempdf):
#    father, child = get_father_child_feature_name(tempdf)
#    feature_name = father + ' causes ' + child + ' weight'
#    df[feature_name] = df.apply(lambda row: create_triplet_weights(tempdf, row), axis=1)
#    
## apply to all the relations
##input is relation list 
#def add_two_feature_for_relation (df, slrelation, result_df_sl):
#    result_group = result_df_sl.groupby('relation')
#    for i in range(len(slrelation)):
#        tempdf = result_group.get_group(slrelation[i])
#        apply_to_df(df,tempdf)
#        apply_weight_to_df (df, tempdf)
#
#add_two_feature_for_relation (df, slrelation, result_df_sl)


#select those are 
#Do two embedding
'''Get the input for embedding'''
def create_embedding_input(slrelation,df):
    embeddinginput = pd.DataFrame()
    for item in slrelation:
        embedding = df[(df[item+' weight']>0)|(df[item+' weight']<0)][item]
        embeddinginput = pd.concat([embeddinginput,embedding])
    return embeddinginput

embeddinginput = create_embedding_input(slrelation,df)

embeddinginput.to_csv(r'./dgl_triplets_01.txt', header=None, index=None, sep=' ', mode='a')

#def create_embedding_input_neg(slrelation,df):
#    embeddinginput = pd.DataFrame()
#    for item in slrelation:
#        embedding = df[(df[item+' weight']<0)][item]
#        embeddinginput = pd.concat([embeddinginput,embedding])
#    return embeddinginput
#
#embeddinginput = create_embedding_input_neg(slrelation,df)
#
#embeddinginput.to_csv(r'./dgl_triplets_1_2.txt', header=None, index=None, sep=' ', mode='a')


