import pandas as pd
from sklearn.utils import shuffle
names = ['qid', 'uid', 'rating']
invited=pd.read_csv("invited_info_train.txt",delimiter="\t",header=None, names=names)
print invited.shape
invited['key']=invited.uid+invited.qid
invited=invited.sort_values(by='key')
new_invited=invited.drop_duplicates('key',keep='last')
print new_invited.shape,new_invited.columns
#new_invited.drop('key',axis=1,inplace=True)
new_invited=new_invited.drop('key',axis=1)
new_invited=shuffle(new_invited)
new_invited.to_csv("invited_info_clean.csv",sep=',',index=False)
print new_invited.shape,new_invited.columns