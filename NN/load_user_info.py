import numpy as np
import pandas as pd
import operator
import hw_utils
from keras.utils.np_utils import to_categorical

questions=pd.read_csv("question_info.txt",delimiter="\t",header=None)
users=pd.read_csv("user_info.txt", delimiter="\t",header=None)
invited=pd.read_csv("invited_info_train.txt",delimiter="\t",header=None)
validate=pd.read_csv("validate_nolabel.txt",delimiter=",",header=0)

all_users_tag={} #[tag]=all_freq
all_users_word={} #[tag]=all_freq
all_users_character={} #[tag]=all_freq
all_questions_tag={}
all_questions_word={}
all_questions_character={}

def user_process():
    for i in range(users.shape[0]):
        tags_list = users.iloc[i, 1].split('/')
        words_list = users.iloc[i, 2].split('/')
        character_list = users.iloc[i, 3].split('/')

        for tag in tags_list:
            if tag in all_users_tag:
                all_users_tag[tag] += 1
            else:
                all_users_tag[tag] = 1

        for word in words_list:
            if word in all_users_word:
                all_users_word[word] += 1
            else:
                all_users_word[word] = 1

        for chr in character_list:
            if chr in all_users_character:
                all_users_character[chr] += 1
            else:
                all_users_character[chr] = 1

    print all_users_tag

    users1 = users.copy()
    users2 = users.copy()

    print 'len of tag ', len(all_users_tag.keys()), 'len of word ', len(all_users_word.keys()), 'len of character ', \
        len(all_users_character.keys())
    for tag in all_users_tag:
        a = [0 for i in range(users.shape[0])]
        users['t' + str(tag)] = a
    '''
    for tag in all_users_word:
        a = [0 for i in range(users.shape[0])]
        users1['w' + str(tag)] = a

    for tag in all_users_character:
        a = [0 for i in range(users.shape[0])]
        users2['c' + str(tag)] = a

    # print users.columns.tolist()
    '''

    for i in range(users.shape[0]):
        tags_list = users.iloc[i, 1].split('/')
        for tag in tags_list:
            col = 't' + str(tag)
            users.at[i, col] = 1
    '''
    for i in range(users1.shape[0]):
        tags_list= users1.iloc[i,2].split('/')
        for tag in tags_list:
            col='w'+str(tag)
            users1.at[i,col]=1

    for i in range(users.shape[0]):
        tags_list= users2.iloc[i,1].split('/')
        for tag in tags_list:
            col='t'+str(tag)
            users2.at[i,col]=1
    '''
    users.to_csv("users_tag_matrix.csv", sep='\t')
    #users1.to_csv("users_word_matrix.csv", sep='\t')
    #users2.to_csv("users_character_matrix.csv", sep='\t')

def questions_process():
    for i in range(questions.shape[0]):
        #print questions.iloc[i, 1]
        tag = questions.iloc[i, 1]
        words_list = questions.iloc[i, 2].split('/')
        character_list = questions.iloc[i, 3].split('/')

        if tag in all_questions_tag:
            all_questions_tag[tag] += 1
        else:
            all_questions_tag[tag] = 1

        for word in words_list:
            if word in all_questions_word:
                all_questions_word[word] += 1
            else:
                all_questions_word[word] = 1

        for chr in character_list:
            if chr in all_questions_character:
                all_questions_character[chr] += 1
            else:
                all_questions_character[chr] = 1

    print all_questions_tag

    questions1 = questions.copy
    questions2 = questions.copy

    print 'len of tag ', len(all_questions_tag.keys()), 'len of word ', len(all_questions_word.keys()), 'len of character ', \
        len(all_questions_character.keys())

    for tag in all_questions_tag:
        a = [0 for i in range(questions.shape[0])]
        questions['qt' + str(tag)] = a
    '''
    for tag in all_questions_word:
        a = [0 for i in range(questions.shape[0])]
        questions1['w' + str(tag)] = a

    for tag in all_questions_character:
        a = [0 for i in range(questions.shape[0])]
        questions2['c' + str(tag)] = a

    # print questions.columns.tolist()
    '''
    for i in range(questions.shape[0]):
        tag = questions.iloc[i, 1]
        col = 'qt' + str(tag)
        questions.at[i, col] = 1
    '''
    for i in range(questions.shape[0]):
        tags_list= questions.iloc[i,2].split('/')
        for tag in tags_list:
            col='t'+str(tag)
            questions.at[i,col]=1

    for i in range(questions.shape[0]):
        tags_list= questions.iloc[i,1].split('/')
        for tag in tags_list:
            col='t'+str(tag)
            questions.at[i,col]=1
    '''
    questions.to_csv("questions_tag_matrix.csv", sep=',')
    # questions.to_csv("questions_word_matrix.csv", sep='\t')
    # questions.to_csv("questions_character_matrix.csv", sep='\t')

print 'questions',questions.shape
print 'users',users.shape
print 'invited',invited.shape
print 'validate',validate.shape
print users.shape[0],users.shape[1]

user_process()
questions_process()
users.rename(columns={0:'uid'}, inplace=True)
questions.rename(columns={0:'qid'}, inplace=True)

questions.drop(questions.columns[[1,2,3]],axis=1, inplace=True)
users.drop(users.columns[[1,2,3]],axis=1, inplace=True)

print 'Create train data'
invited.rename(columns={0:'qid',1:'uid',2:'y'}, inplace=True)
invited_rows=invited.shape[0]
y_tr = invited[[2]].copy()
#print y_tr
invited.drop(invited.columns[[2]],axis=1, inplace=True)
#print users.columns.tolist()
#print questions.columns.tolist()
#print invited.columns.tolist()
join_uid = pd.merge(invited, users, how='inner', on=['uid'])
#print 'join_uid',join_uid.shape,join_uid.columns.tolist()
train = pd.merge(join_uid, questions, how='inner', on=['qid'])
print train.shape,train.columns.tolist()
train.drop(train.columns[[0,1]],axis=1,inplace=True)
X_tr = train.copy()
train.to_csv("train.csv", sep=',')

print 'Create Validate data'
validate.rename(columns={0:'qid',1:'uid'}, inplace=True)
#print users.columns.tolist()
#print questions.columns.tolist()
#print validate.columns.tolist()
join_uid = pd.merge(validate, users, how='inner', on=['uid'])
#print 'join_uid',join_uid.shape,join_uid.columns.tolist()
test = pd.merge(join_uid, questions, how='inner', on=['qid'])
#### AFTER JOIN THE NO OF ROWS MAY DECREASE ? UNSEEN USER OR UNSEEN QUESTION ?
print test.shape,test.columns.tolist()
test.drop(test.columns[[0,1,2]],axis=1,inplace=True)

'''
validate_rows=test.shape[0]
test[test.shape[1]]=np.random.randint(2, size=validate_rows)
print test.shape,test.columns.tolist()
y_te = validate[[test.shape[1]-1]].copy()
test.drop(test.columns[test.shape[1]-1],1,inplace=True)
'''

#y_values=pd.DataFrame(y_values)
y_te=np.random.randint(2, size=test.shape[0])
print y_te.shape
X_te = test.copy()
test.to_csv("test.csv", sep=',')
X_tr, X_te=hw_utils.normalize(X_tr, X_te)

# to handle categorical represenatation
y_tr = to_categorical(y_tr.values)
y_te = to_categorical(y_te)

print X_tr.shape,X_te.shape,y_tr.shape,y_te.shape
din=X_tr.shape[1]
dout=2

del train,test,questions,users,invited,validate


print 'Try Neural Network'
print din,dout
time_e=hw_utils.start_time()
arch_list_e=[[din, din, dout],
             [din, din*10, dout],
             [din, din*10, din*7.5, dout],
             [din, din*15, din*10, din*7.5, dout],
             [din, din*15, din*1.5, din*10, din*7.5, dout]]
#arch_list_e=[[din, din*15, din*10, din*7.5, dout]]
arch_list_e=[[din, din, dout]]
y_out=hw_utils.testmodels(X_tr, y_tr, X_te, y_te, arch_list_e, actfn='relu', last_act='softmax', reg_coeffs=[0.0],
				num_epoch=30, batch_size=1000, sgd_lr=5e-4, sgd_decays=[0.0], sgd_moms=[0.0],sgd_Nesterov=False, EStop=False,
                verbose=0)
hw_utils.end_time(time_e)
validate[2]=y_out[0]
validate[3]=y_out[1]
validate.to_csv("validate_result.csv",index=False)
