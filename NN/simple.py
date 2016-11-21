import graphlab as gl
import pandas as pd
names = ['user_id', 'item_id', 'rating']
invited=pd.read_csv("invited_info_train.txt",delimiter="\t",header=None, names=names)
validate=pd.read_csv("validate_nolabel.txt",delimiter=",",header=0, names=names)
test=pd.read_csv("test_nolabel.txt",delimiter=",",header=0, names=names)

validate=gl.SFrame(validate)
test=gl.SFrame(test)

#print invited
sf = gl.SFrame(invited)
train, test = gl.recommender.util.random_split_by_user(sf)
m = gl.recommender.create(train, target='rating')
eval = m.evaluate(test)
val_output=m.predict(validate)
val_output.save("val_simple_reco.csv",format='csv')

test_output=m.predict(test)
test_output.save("test_simple_reco.csv",format='csv')