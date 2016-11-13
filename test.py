import pandas as pd
import numpy as np
from sklearn.decomposition import NMF

data = pd.read_csv('invited_info_train.txt' ,sep='\t',header =None)
# matrix = np.array(data)
print data[0][0]

# model = NMF(n_components=2, init='random', random_state=0)
# model.fit(matrix)
# print model.components_
# print model.reconstruction_err_










