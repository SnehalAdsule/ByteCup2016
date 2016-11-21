import graphlab
import graphlab.toolkits.recommender as recommender
from graphlab.toolkits.recommender import ranking_factorization_recommender
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse

names=['qid', 'uid']
user_info=pd.read_csv("question_info.txt",delimiter=",",header=None,names=names)
item_info=pd.read_csv("user_info.txt",delimiter=",",header=None,names=names)

names = ['qid', 'uid', 'rating']
invited=pd.read_csv("invited_info_train.txt",delimiter="\t",header=None, names=names)
validate=pd.read_csv("validate_nolabel.txt",delimiter=",",header=0, names=names)
test=pd.read_csv("test_nolabel.txt",delimiter=",",header=0, names=names)

invited=graphlab.SFrame(invited)
validate=graphlab.SFrame(validate)
test=graphlab.SFrame(test)

sf = graphlab.SFrame(invited)
train, train_val = graphlab.recommender.util.random_split_by_user(sf, user_id='qid', item_id='uid')

#m1 = ranking_factorization_recommender.create(sf)

m1 = ranking_factorization_recommender.create(sf, user_id="qid",item_id="uid",target='rating')

user_info = graphlab.SFrame(user_info)
item_info = graphlab.SFrame(item_info)
'''
m2 = ranking_factorization_recommender.create(sf, target='rating', user_id="qid",item_id="uid",
                                               user_data=user_info,
                                               item_data=item_info)
'''
m3 = ranking_factorization_recommender.create(sf, target='rating',
                                               ranking_regularization = 0.1, user_id="qid",item_id="uid",
                                               unobserved_rating_value = 1)
m4 = ranking_factorization_recommender.create(sf, target='rating', user_id="qid",item_id="uid",solver = 'ials')
#graphlab.recommender.util.compare_models(test, [m1,m3,m4])
print '==============M1 Evalauating HOLD out train '
m1.evaluate(train_val)
#m2.evaluate(train_val)
print '============== M3 Evalauating HOLD out train'
m3.evaluate(train_val)
print '==============M4 Evalauating HOLD out train'
m4.evaluate(train_val)
print '===============Predict val'
m1_pred=m1.predict(validate)
m1_pred.save("val_m1_factor.csv", format='csv')
#m2.predict(validate)
m3_pred=m3.predict(validate)
m3_pred.save("val_m3_factor.csv", format='csv')
m4_pred=m4.predict(validate)
m4_pred.save("val_m4_factor.csv", format='csv')

print '===============Predict test'
m1_pred_test=m1.predict(test)
m1_pred_test.save("test_m1_factor.csv", format='csv')
#m2.predict(test)
m2_pred_test=m3.predict(test)
m2_pred_test.save("test_m3_factor.csv", format='csv')
m3_pred_test=m4.predict(test)
m3_pred_test.save("test_m4_factor.csv", format='csv')

