import pandas as pd
import numpy as np
import timeit
from sklearn.metrics import mean_squared_error

def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    n_train = int(ratings.shape[0]*0.8)
    print n_train
    train=ratings[0:n_train,:]
    test=ratings[n_train:,:]
    print train.shape, test.shape

    '''
    for user in xrange(ratings.shape[0]):
        print ratings[user, :].nonzero()[0]
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0])
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]

    # Test and training are truly disjoint
    assert (np.all((train * test) == 0))
    '''
    print 'train test split'
    return train, test

def slow_similarity(ratings, kind='user'):
    if kind == 'user':
        axmax = 0
        axmin = 1
    elif kind == 'item':
        axmax = 1
        axmin = 0
    sim = np.zeros((ratings.shape[axmax], ratings.shape[axmax]))
    for u in xrange(ratings.shape[axmax]):
        for uprime in xrange(ratings.shape[axmax]):
            rui_sqrd = 0.
            ruprimei_sqrd = 0.
            for i in xrange(ratings.shape[axmin]):
                sim[u, uprime] = ratings[u, i] * ratings[uprime, i]
                rui_sqrd += ratings[u, i] ** 2
                ruprimei_sqrd += ratings[uprime, i] ** 2
            sim[u, uprime] /= rui_sqrd * ruprimei_sqrd
    return sim

def fast_similarity(ratings, kind='user', epsilon=1e-9):
    # epsilon -> small number for handling dived-by-zero errors
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    elif kind == 'item':
        sim = ratings.T.dot(ratings) + epsilon

    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

def predict_slow_simple(ratings, similarity, kind='user'):
    pred = np.zeros(ratings.shape)
    if kind == 'user':
        for i in xrange(ratings.shape[0]):
            for j in xrange(ratings.shape[1]):
                pred[i, j] = similarity[i, :].dot(ratings[:, j])\
                             /np.sum(np.abs(similarity[i, :]))
        return pred
    elif kind == 'item':
        for i in xrange(ratings.shape[0]):
            for j in xrange(ratings.shape[1]):
                pred[i, j] = similarity[j, :].dot(ratings[i, :].T)\
                             /np.sum(np.abs(similarity[j, :]))

        return pred

def predict_fast_simple(ratings, similarity, kind='user'):
    if kind == 'user':
        return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif kind == 'item':
        return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])

def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

names = ['qid', 'uid', 'answered']
invited=pd.read_csv("invited_info_train.txt",delimiter="\t",header=None, names=names)
validate=pd.read_csv("validate_nolabel.txt",delimiter=",",header=0, names=names)

list_qid=invited.qid.unique()
list_uid=invited.uid.unique()

map_uid={}
map_qid={}
id=0
for uid in list_uid:
    map_uid[uid]=id
    id+=1
id=0
for qid in list_qid:
    map_qid[qid]=id
    id+=1

n_users = invited.qid.unique().shape[0]
n_items = invited.uid.unique().shape[0]
print str(n_users) + ' users'
print str(n_items) + ' items'

ratings = np.zeros((n_users, n_items))
for row in invited.itertuples():
    ratings[map_qid[row[1]], map_uid[row[2]]] = row[3]
print 'ratings',ratings.shape

sparsity = float(len(ratings.nonzero()[0]))
sparsity /= (ratings.shape[0] * ratings.shape[1])
sparsity *= 100
print 'Sparsity: {:4.2f}%'.format(sparsity)

train, test = train_test_split(ratings)

print 'Caluclating Similarity'
user_similarity = fast_similarity(train, kind='user')
item_similarity = fast_similarity(train, kind='item')
print 'item_similarity', item_similarity.shape
print 'user_similarity', user_similarity.shape


print 'Caluclating Prediction'
item_prediction = predict_fast_simple(train, item_similarity, kind='item')
user_prediction = predict_fast_simple(train, user_similarity, kind='user')

print 'User-based CF MSE: ' + str(get_mse(user_prediction, test))
print 'Item-based CF MSE: ' + str(get_mse(item_prediction, test))
