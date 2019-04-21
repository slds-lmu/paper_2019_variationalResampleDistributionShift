import matplotlib.pyplot as plt
import json
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# # Cluster using VGMM

from sklearn import mixture
import itertools
import numpy as np
color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])

class VGMM(object):

    def cluster(self,X_train):

        dpgmm = mixture.BayesianGaussianMixture(n_components=5,
                                                covariance_type='full').fit(X_train)
        while dpgmm.converged_  ==False:

            max_iter = dpgmm.n_iter_ *2
            print("increase the number of iteration to {} to converge".format(max_iter))
            dpgmm = mixture.BayesianGaussianMixture(n_components=5, covariance_type='full',max_iter=max_iter).fit(X_train)
        X_prediction_vgmm = dpgmm.predict(X_train)
        dict = {}
        for i in range(5):
            # dict[i] istore the index of data belongs to cluster i
            dict[str(i)] = np.where(X_prediction_vgmm == i)[0].tolist()
        return dict,X_prediction_vgmm

    def save_dict(self,path,dict):
        with open(path, 'w') as f:
            json.dump(dict, f)


    def load_dict(self,path):
        with open(path) as f:
            my_dict = json.load(f)
        return my_dict


    def save_predict(self,path,predict):
        np.savetxt(path,predict,delimiter='')