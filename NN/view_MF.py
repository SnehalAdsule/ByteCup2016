import graphlab
#import graphlab.recommender.ranking_factorization_recommender
from graphlab.toolkits.recommender import ranking_factorization_recommender
import numpy as np
import pandas as pd
import ndcg_bytecup
import json
from sklearn.utils import shuffle

def user_process(users):
    tag_col=[]
    word_col=[]
    char_col=[]
    for i in range(users.shape[0]):
        tags_list = users.iloc[i, 1].split('/')
        words_list = users.iloc[i, 2].split('/')
        character_list = users.iloc[i, 3].split('/')
        tag_col.append(tags_list)
        word_col.append(words_list)
        char_col.append(character_list)

    users.tags=tag_col
    users.word=word_col
    users.char=char_col
    #print users

def questions_process(questions):
    tag_col=[]
    word_col=[]
    char_col=[]
    for i in range(questions.shape[0]):
        tags_list = questions.iloc[i, 1]
        words_list = questions.iloc[i, 2].split('/')
        character_list = questions.iloc[i, 3].split('/')
        tag_col.append(tags_list)
        word_col.append(words_list)
        char_col.append(character_list)

    questions.tags=tag_col
    questions.word=word_col
    questions.char=char_col
    #print questions

names=['qid','tags','word','char','feat1','feat2','feat3']
user_info=pd.read_csv("question_info.txt",delimiter="\t",header=None,names=names)
names=[ 'uid','tags','word','char']
item_info=pd.read_csv("user_info.txt",delimiter="\t",header=None,names=names)
names = ['qid', 'uid', 'rating']
#invited=pd.read_csv("invited_info_train.txt",delimiter="\t",header=None, names=names)
invited=pd.read_csv("invited_info_clean.csv",delimiter=",", header=0, names=names)
invited_train=pd.read_csv("invited_train_80.csv",delimiter=",", header=0, names=names)
invited_train_val=pd.read_csv("invited_train_20.csv",delimiter=",", header=None, names=names)
validate=pd.read_csv("validate_nolabel.txt",delimiter=",",header=0, names=names)
test=pd.read_csv("test_nolabel.txt",delimiter=",",header=0, names=names)
invited_train=shuffle(invited_train)
invited=shuffle(invited)
invited_train_val=shuffle(invited_train_val)
'''
user_process(item_info)
questions_process(user_info)
#print invited.shape,invited.columns, user_info.shape,user_info.columns, item_info.shape,item_info.columns


print item_info.columns
print user_info.columns

join_uid = pd.merge(invited_train, item_info, how='left', on=['uid'])
invited_train = pd.merge(join_uid, user_info, how='left', on=['qid'])

join_uid = pd.merge(invited_train_val, item_info, how='left', on=['uid'])
invited_train_val = pd.merge(join_uid, user_info, how='left', on=['qid'])

join_uid = pd.merge(validate, item_info, how='left', on=['uid'])
validate = pd.merge(join_uid, user_info, how='left', on=['qid'])

join_uid = pd.merge(test, item_info, how='left', on=['uid'])
test = pd.merge(join_uid, user_info, how='left', on=['qid'])
'''

print 'invited',invited.shape,invited.columns

invited=graphlab.SFrame(invited)
validate=graphlab.SFrame(validate)
test=graphlab.SFrame(test)
user_info=graphlab.SFrame(user_info)
item_info=graphlab.SFrame(item_info)
train=graphlab.SFrame(invited_train)
train_val=graphlab.SFrame(invited_train_val)

sf = graphlab.SFrame(invited)
#train, train_val = graphlab.recommender.util.random_split_by_user(sf, user_id='qid', item_id='uid')

ranking_regularization=0.1
m3 = ranking_factorization_recommender.create(invited, target='rating', user_id="qid",item_id="uid",
                                               #user_data=user_info,
                                               #item_data=item_info,
                                               ranking_regularization=ranking_regularization,
                                               unobserved_rating_value=1,
                                               max_iterations=100
                                               )

print '============== M3 Evalauating HOLD out train'
m3.evaluate(train_val)
m3_pred_val=m3.predict(train_val)
train_val_pred=train_val.copy()
train_val_pred['rating']=m3_pred_val
#print train_val_pred
train_val.save("train_val_TRUTH"+str(ranking_regularization)+".csv", format='csv')
train_val_pred.save("train_val_m3_factor"+str(ranking_regularization)+".csv", format='csv')

print '============== NDCD Analysis '
ndcg_scores=ndcg_bytecup.check("train_val_TRUTH"+str(ranking_regularization)+".csv","train_val_m3_factor"+str(ranking_regularization)+".csv")

#m3_pred_val

print '===============Predict val'
m3_pred=m3.predict(validate)
m3_pred.save("val_m3_factor"+str(ranking_regularization)+".csv", format='csv')


print '===============Predict test'
m3_pred_test=m3.predict(test)
m3_pred_test.save("test_m3_factor"+str(ranking_regularization)+".csv", format='csv')

'''
view = m3.views.evaluate(train_val)
view.show()

# Explore predictions
view = m3.views.explore(item_data=item_info,
              item_name_column='uid')

# Explore evals
view = m3.views.overview(
     validation_set=train_val,
     user_data=user_info,
     user_name_column='qid',
     item_data=item_info,
     item_name_column='uid')
view.show()
'''