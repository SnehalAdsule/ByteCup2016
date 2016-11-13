import numpy as np
import pandas as pd
questions=pd.read_csv("question_info.txt",delimiter="\t",header=None)
users=pd.read_csv("user_info.txt", delimiter="\t",header=None)
invited=pd.read_csv("invited_info_train.txt",delimiter="\t",header=None)
validate=pd.read_csv("validate_nolabel.txt",delimiter=",",header=None)
all_users_tag={} #[tag]=all_freq
all_users_word={} #[tag]=all_freq
all_users_character={} #[tag]=all_freq
all_question_tags={}
all_question_tags={}
all_question_words={}
all_question_character={}

print 'questions',questions.shape
print 'users',users.shape
print 'invited',invited.shape
print 'validate',validate.shape
print users.shape[0],users.shape[1]

for i in range(users.shape[0]):
    tags_list= users.iloc[i,1].split('/')
    words_list= users.iloc[i,2].split('/')
    character_list= users.iloc[i,3].split('/')
    for tag in tags_list:
        if tag in all_users_tag:
            all_users_tag[tag]+=1
        else:
            all_users_tag[tag]=1
#all_users_tag.sort()

for tag in all_users_tag:
    a=[0 for i in range(users.shape[0])]
    users['t'+str(tag)]=a

print users.columns.tolist()

for i in range(users.shape[0]):
    tags_list= users.iloc[i,1].split('/')
    for tag in tags_list:
        col='t'+str(tag)
        #print users.at[i,col]
        col='t'+tag
        users.at[i,col]=1
users.to_csv("users_tag_matrix.csv", sep='\t')
