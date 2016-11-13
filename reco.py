import pandas as pd
import numpy as np
import sys

data = pd.read_csv('invited_info_train.txt' ,sep='\t',header =None)
qdict = {}; qArr = []
udict = {}; uArr = []
for i in data[0]:
    if i not in qdict:
        qdict[i]=1
for i in data[1]:
    if i not in udict:
        udict[i] = 1

for i in qdict:
    qArr.append(i)

for i in udict:
    uArr.append(i)


# pdMatrix = pd.DataFrame(index=qArr,columns=uArr)
pdMatrix1 = pd.DataFrame(index=qArr,columns=uArr)

for i in range(0,len(data[0])):
    qid = data.ix[i]
    k = qid[0]
    j =qid[1]
    val = qid[2]
    # print k
    # print j
    # print qid[2]
    # pdMatrix1.set_index(k)
    pdMatrix1[j][k] = val
    print pdMatrix1[j][k]


sys.stdout = open("matrix.txt", "w")
print pdMatrix1


