#import seaborn as sns
import numpy as np
import json
from  data_manipulator import concatenate_data_from_dir
import config as config
# from utils import *
import utils_parent as utils_parent
from sklearn.decomposition import PCA
import argparse

def counting_label(num_labels,num_clusters):
    # load data
    _,y = utils_parent.load_mnist(config.dataset_name)
    global_index = np.load(config.data_path+config.global_index_name,allow_pickle=True)
    results = {}
    for i in range(num_clusters):
        index = global_index.item().get(str(i))
        temp_y = np.sum(y[index],axis=0)
        sum = np.sum(temp_y)
        temp_y = temp_y/sum
        results[str(i)]= temp_y

    with open("distribution_y.txt", 'a') as lf:
        lf.write(str(results))
    return results

#compute kernal density within one cluster
def kernel_density_estimation_single_Cluster(xs,result_path,img_name):
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as st
    from sklearn.datasets.samples_generator import make_blobs
    from mpl_toolkits.mplot3d import Axes3D
    # Extract x and y
    x = xs[:, 0]  # only works for 2d
    y = xs[:, 1]  # only works for 2d
    # Define the borders
    deltaX = (max(x) - min(x)) / 10
    deltaY = (max(y) - min(y)) / 10
    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY
    print(xmin, xmax, ymin, ymax)
    # Create meshgrid
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    fig = plt.figure(figsize=(13, 7))
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(xx, yy, f, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('PDF')
    ax.set_title('Surface plot of Gaussian 2D KDE')
    fig.colorbar(surf, shrink=0.5, aspect=5)  # add color bar indicating the PDF
    ax.view_init(60, 35)
    plt.savefig(result_path + "/KDE"+img_name+".jpg")
    plt.clf()

# compute kernal density within combined data from two cluster
def kernel_density_estimation(xs,xt,result_path,img_name):   # only for plotting
    import numpy as np
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')   # FIXME: add this line if executed on server.
    import scipy.stats as st
    from sklearn.datasets.samples_generator import make_blobs
    from mpl_toolkits.mplot3d import Axes3D
    n_samples = min(xs.shape[0], xt.shape[0])
    xs = xs[:n_samples]
    xt = xt[:n_samples]
    # concatenate the two datasets into the final training set
    X_train = np.vstack([xs, xt])

    # Extract x and y
    x = X_train[:, 0]
    y = X_train[:, 1]
    # Define the borders
    deltaX = (max(x) - min(x)) / 10
    deltaY = (max(y) - min(y)) / 10
    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY
    print(xmin, xmax, ymin, ymax)
    # Create meshgrid
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    fig = plt.figure(figsize=(13, 7))
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(xx, yy, f, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('PDF')
    ax.set_title('Surface plot of Gaussian 2D KDE')
    fig.colorbar(surf, shrink=0.5, aspect=5)  # add color bar indicating the PDF
    ax.view_init(60, 35)
    plt.savefig(result_path + "/KDE"+img_name+".jpg")
    plt.clf()


def density_estimation_GMM(xs,xt,result_path,img_name):
    # xs, xt: 2D array like data
    import numpy as np
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')   # FIXME: add this line if executed on server.
    from sklearn import mixture

    n_samples = min(xs.shape[0], xt.shape[0])
    xs = xs[:n_samples]
    xt = xt[:n_samples]

    # concatenate the two datasets into the final training set
    X_train = np.vstack([xs, xt])

    # fit a Gaussian Mixture Model with two components
    clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
    clf.fit(X_train)

    # display predicted scores by the model as a contour plot
    x = np.linspace(-75., 75.)
    y = np.linspace(-75., 75.)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -clf.score_samples(XX)
    Z = Z.reshape(X.shape)

    CS = plt.contour(X, Y, Z)
    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    plt.scatter(xs[:, 0], xs[:, 1], .8)
    plt.scatter(xt[:, 0], xt[:, 1], .9)

    plt.title('Negative log-likelihood predicted by a GMM')
    plt.axis('tight')
    plt.savefig(result_path + "/DE_GMM"+img_name+".jpg")
    plt.clf()


def gromov_wasserstein_distance(data_path,num_labels,num_clusters,result_path):
    import json
    import scipy as sp
    import numpy as np
    import matplotlib.pylab as pl
    import ot

    # z: global training, cluster according to label
    # z = np.load(data_path + "/L-1/z.npy")
    z = np.load(data_path + "/L-1" + config.z_name)

    # with open(data_path + "/L-1/cluster_dict.json") as f:
    with open(data_path + "/L-1"+config.cluster_index_json_name) as f:
        pos_index_dict = json.load(f)   # cluster index dictionary, not according to label. cluster on the whole space

    # dict: dictionary of data which training and clustering within label
    dict = concatenate_data_from_dir(data_path, num_labels,num_clusters)

    for i in range(num_clusters):
        # Compute distance kernels, normalize them and then display
        xs = dict[str(i)]    #FIXME: this is index or latent space?
        xt = z[pos_index_dict[str(i)]]
        n_samples = min(xs.shape[0], xt.shape[0])
        xs = xs[:n_samples]
        xt = xt[:n_samples]
        C1 = sp.spatial.distance.cdist(xs, xs)
        C2 = sp.spatial.distance.cdist(xt, xt)
        C1 /= C1.max()
        C2 /= C2.max()

        p = ot.unif(n_samples)
        q = ot.unif(n_samples)

        gw0, log0 = ot.gromov.gromov_wasserstein(
            C1, C2, p, q, 'square_loss', verbose=True, log=True)

        gw, log = ot.gromov.entropic_gromov_wasserstein(
            C1, C2, p, q, 'square_loss', epsilon=5e-4, log=True, verbose=True)

        print(" [*] cluster "+ str(i)+": ")
        print('Gromov-Wasserstein distances: ' + str(log0['gw_dist']))
        print('Entropic Gromov-Wasserstein distances: ' + str(log['gw_dist']))

        pl.figure(1, (10, 5))

        pl.subplot(1, 2, 1)
        pl.imshow(gw0, cmap='jet')
        pl.title('Gromov Wasserstein')

        pl.subplot(1, 2, 2)
        pl.imshow(gw, cmap='jet')
        pl.title('Entropic Gromov Wasserstein')
        pl.savefig(result_path + "/WD.jpg")



# computer gromov wasserstein distance on data fit and transformed by T_SNE
def gromov_wasserstein_distance_TSNE(data_path,num_labels,num_clusters,result_path):
    import scipy as sp
    import matplotlib.pylab as pl
    pl.switch_backend('agg')   # FIXME: add this line if executed on server.
    import ot
    # d_t = load_dict(data_path+ config.statistic_name4d_t)
    # d_s = load_dict(data_path+ config.statistic_name4d_s)
    d_t = np.load(data_path+ config.statistic_name4d_t)   # non-label based vae + vgmm+ t-sne(per-cluster)

    d_s = np.load(data_path+ config.statistic_name4d_s)   # label based vae + vgmm + t-sne(per-cluster, after cluster is merged)

    for i in range(num_clusters):
        # Compute distance kernels, normalize them and then display
        xs = d_s.item().get(str(i))
        xt = d_t.item().get(str(i))
        print(xt.shape)
        n_samples = min(100,xs.shape[0], xt.shape[0])
        xs = xs[:n_samples]
        xt = xt[:n_samples]
        C1 = sp.spatial.distance.cdist(xs, xs)
        C2 = sp.spatial.distance.cdist(xt, xt)
        C1 /= C1.max()
        C2 /= C2.max()

        p = ot.unif(n_samples)
        q = ot.unif(n_samples)

        gw0, log0 = ot.gromov.gromov_wasserstein(
            C1, C2, p, q, 'square_loss', verbose=True, log=True)

        gw, log = ot.gromov.entropic_gromov_wasserstein(
            C1, C2, p, q, 'square_loss', epsilon=5e-4, log=True, verbose=True)

        print('Gromov-Wasserstein distances: ' + str(log0['gw_dist']))
        print('Entropic Gromov-Wasserstein distances: ' + str(log['gw_dist']))

        pl.figure(1, (10, 5))

        pl.subplot(1, 2, 1)
        pl.imshow(gw0, cmap='jet')
        pl.title('Gromov Wasserstein')

        pl.subplot(1, 2, 2)
        pl.imshow(gw, cmap='jet')
        pl.title('Entropic Gromov Wasserstein')
        pl.savefig(result_path + "/WD_TSNE.jpg")

# computer gromov wasserstein distance on data fit and transformed by T_SNE
def gromov_wasserstein_distance_TSNE_test(data_path,num_labels,num_clusters,result_path):
    import scipy as sp
    import matplotlib.pylab as pl
    pl.switch_backend('agg')   # FIXME: add this line if executed on server.
    import ot
    d_t = np.load(data_path+ config.statistic_name4d_t)
    d_s = np.load(data_path+ config.statistic_name4d_s)
    # Compute distance kernels, normalize them and then display
    xs = d_s.item().get('0')
    xt = d_t.item().get('0')
    print(xt.shape)
    print(xs.shape)
    n_samples = min(100,xs.shape[0], xt.shape[0])
    xs = xs[:n_samples]
    xt = xt[:n_samples]
    C1 = sp.spatial.distance.cdist(xs, xs)
    C2 = sp.spatial.distance.cdist(xt, xt)
    C1 /= C1.max()
    C2 /= C2.max()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)

    gw0, log0 = ot.gromov.gromov_wasserstein(
        C1, C2, p, q, 'square_loss', verbose=True, log=True)

    gw, log = ot.gromov.entropic_gromov_wasserstein(
        C1, C2, p, q, 'square_loss', epsilon=5e-4, log=True, verbose=True)

    print('Gromov-Wasserstein distances: ' + str(log0['gw_dist']))
    print('Entropic Gromov-Wasserstein distances: ' + str(log['gw_dist']))
    pl.figure(1, (10, 5))
    pl.subplot(1, 2, 1)
    pl.imshow(gw0, cmap='jet')
    pl.title('Gromov Wasserstein')
    pl.subplot(1, 2, 2)
    pl.imshow(gw, cmap='jet')
    pl.title('Entropic Gromov Wasserstein')
    pl.savefig(result_path + "/WD_TSNE.jpg")


def kernel_density_estimation_on_latent_space(data_path,num_clusters):
    # z = np.load(data_path+"/L-1/z.npy")
    z = np.load(data_path+"/L-1" + config.z_name)
    # b = np.load(config.data_path+"/global_index_cluster_data.npy")
    b = np.load(config.data_path + config.global_index_name)

    for i in range(num_clusters):
       xs = z[b.item().get(str(i))]
       print(xs.shape)
       # pca_2 = PCA(n_components=2)
       # xs = pca_2.fit_transform(xs)
       kernel_density_estimation_single_Cluster(xs,config.result_path,str(i))
       for j in range(5):
           if i!=j:
               xt = z[b.item().get(str(j))]
               # xt = pca_2.fit_transform(xt)
               kernel_density_estimation(xs,xt,config.result_path,str(i)+str(j))

# computer gromov wasserstein distance on latent space z
# IMPORTANT: FIXME:
def gromov_wasserstein_distance_latent_space(data_path,num_labels,num_clusters,result_path):
    import scipy as sp
    import matplotlib.pylab as pl
    import ot
    # z = np.load(data_path+ "/L-1/z.npy")  # -1 means no discrimation for labelsa, the same vae transform , orthogonal concept to whether cluster on this z space or use other mehtod to split into clusters
    z = np.load(data_path+ "/L-1" + config.z_name)  # -1 means no discrimation for labelsa, the same vae transform , orthogonal concept to whether cluster on this z space or use other mehtod to split into clusters
    # index = np.load(data_path+"/global_index_cluster_data.npy")
    index = np.load(data_path + config.global_index_name)   # according to label, vae, vgmm, merge , cluster , per cluster-index(globally)
    d_t = z[index.item().get('0')]
    d_s = z[index.item().get('1')]
    # Compute distance kernels, normalize them and then display
    xs = d_s
    xt = d_t
    print(xt.shape)
    n_samples = min(100, xs.shape[0], xt.shape[0])
    xs = xs[:n_samples]
    xt = xt[:n_samples]
    C1 = sp.spatial.distance.cdist(xs, xs)
    C2 = sp.spatial.distance.cdist(xt, xt)
    C1 /= C1.max()
    C2 /= C2.max()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)

    gw0, log0 = ot.gromov.gromov_wasserstein(
        C1, C2, p, q, 'square_loss', verbose=True, log=True)

    gw, log = ot.gromov.entropic_gromov_wasserstein(
        C1, C2, p, q, 'square_loss', epsilon=5e-4, log=True, verbose=True)

    print('Gromov-Wasserstein distances: ' + str(log0['gw_dist']))
    print('Entropic Gromov-Wasserstein distances: ' + str(log['gw_dist']))

    pl.figure(1, (10, 5))

    pl.subplot(1, 2, 1)
    pl.imshow(gw0, cmap='jet')
    pl.title('Gromov Wasserstein')

    pl.subplot(1, 2, 2)
    pl.imshow(gw, cmap='jet')
    pl.title('Entropic Gromov Wasserstein')
    pl.savefig(result_path + "/WD_TSNE.jpg")


# computer gromov wasserstein distance on latent space z
def gromov_wasserstein_distance_latent_space_cluster(data_path,num_labels,num_clusters,result_path,args):
    import scipy as sp
    import matplotlib.pylab as pl
    import ot
    # z = np.load(data_path+ "/L-1/z.npy")  # -1 means no discrimation for labelsa, the same vae transform , orthogonal concept to whether cluster on this z space or use other mehtod to split into clusters
    z = np.load(data_path+ "/L-1" + config.z_name,allow_pickle=True)  # -1 means no discrimation for labelsa, the same vae transform , orthogonal concept to whether cluster on this z space or use other mehtod to split into clusters
    # index = np.load(data_path+"/global_index_cluster_data.npy")
    index = np.load(data_path + config.global_index_name,allow_pickle=True)   # according to label, vae, vgmm, merge , cluster , per cluster-index(globally)
    results = {}
    mat = np.zeros((num_clusters,num_clusters))
    for i in range(num_clusters):
        xs = z[index.item().get(str(i))]
        for j in range(num_clusters):
            xt = z[index.item().get(str(j))]
            # Compute distance kernels, normalize them and then display
            n_samples = min(xs.shape[0], xt.shape[0])
            if args.debug == True:
                n_samples = 100
            xs = xs[:n_samples]
            xt = xt[:n_samples]
            C1 = sp.spatial.distance.cdist(xs, xs)
            C2 = sp.spatial.distance.cdist(xt, xt)
            C1 /= C1.max()
            C2 /= C2.max()

            p = ot.unif(n_samples)
            q = ot.unif(n_samples)

            gw0, log0 = ot.gromov.gromov_wasserstein(
                C1, C2, p, q, 'square_loss', verbose=True, log=True)

            gw, log = ot.gromov.entropic_gromov_wasserstein(
                C1, C2, p, q, 'square_loss', epsilon=5e-4, log=True, verbose=True)

            print('Gromov-Wasserstein distances between {}_{} clusters: {} '.format(i,j,str(log0['gw_dist'])) )
            print('Entropic Gromov-Wasserstein distances between {}_{} clusters: {}'.format(i,j,str(log['gw_dist'])) )
            results[str(i)+str(j)]={"GW":log0['gw_dist'],"EGW":log['gw_dist']}
            mat[i,j] = log0['gw_dist']
            pl.figure(1, (10, 5))
            pl.subplot(1, 2, 1)
            pl.imshow(gw0, cmap='jet')
            pl.title('Gromov Wasserstein')

            pl.subplot(1, 2, 2)
            pl.imshow(gw, cmap='jet')
            pl.title('Entropic Gromov Wasserstein')
            pl.savefig(result_path + "/WD_TSNE{}_{}.jpg".format(i,j))
    # print(results)
    print(mat)
    with open("wd_vgmm.txt", 'a') as lf:
        lf.write(str(results))
    return results


def gromov_wasserstein_distance_latent_space_cluster_emd(data_path,num_labels,num_clusters,result_path,args):
    import scipy as sp
    import matplotlib.pylab as pl
    import ot
    # z = np.load(data_path+ "/L-1/z.npy")  # -1 means no discrimation for labelsa, the same vae transform , orthogonal concept to whether cluster on this z space or use other mehtod to split into clusters
    z = np.load(data_path+ "/L-1" + config.z_name,allow_pickle=True)  # -1 means no discrimation for labelsa, the same vae transform , orthogonal concept to whether cluster on this z space or use other mehtod to split into clusters
    # index = np.load(data_path+"/global_index_cluster_data.npy")
    index = np.load(data_path + config.global_index_name,allow_pickle=True)   # according to label, vae, vgmm, merge , cluster , per cluster-index(globally)
    results = {}
    mat = np.zeros((num_clusters,num_clusters))
    for i in range(num_clusters):
        xs = z[index.item().get(str(i))]
        for j in range(num_clusters):
            xt = z[index.item().get(str(j))]
            # Compute distance kernels, normalize them and then display
            n_samples = min(xs.shape[0], xt.shape[0])
            if args.debug == True:
                n_samples = 100
            xs = xs[:n_samples]
            xt = xt[:n_samples]
            M = sp.spatial.distance.cdist(xs, xt)
            ds, dt = np.ones((len(xs),)) / len(xs), np.ones((len(xt),)) / len(xt)
            M /= M.max()
            g0, loss = ot.emd(ds,dt, M, log = True)
            print('Gromov-Wasserstein distances between {}_{} clusters: {}--{} '.format(i,j,str(loss), str(np.sum(g0))) )
            #results[str(i)+str(j)]={"GW":log0['gw_dist'],"EGW":log['gw_dist']}
            results[str(i)+str(j)]= loss["cost"]
            mat[i,j] = loss["cost"]
            #pl.figure(1, (10, 5))
            #pl.subplot(1, 2, 1)
            #pl.imshow(gw0, cmap='jet')
            #pl.title('Gromov Wasserstein')

            #pl.subplot(1, 2, 2)
            #pl.imshow(gw, cmap='jet')
            #pl.title('Entropic Gromov Wasserstein')
            #pl.savefig(result_path + "/WD_TSNE{}_{}.jpg".format(i,j))
    # print(results)
    print(mat)
    with open("wd_vgmm.txt", 'a') as lf:
        lf.write(str(results))
    return results


#
# # computer gromov wasserstein distance on latent space z
# def gromov_wasserstein_distance_latent_space_rand(data_path,num_labels,num_clusters,result_path):
#     import scipy as sp
#     import matplotlib.pylab as pl
#     import ot
#     # z = np.load(data_path+ "/L-1/z.npy")  # -1 means no discrimation for labelsa, the same vae transform , orthogonal concept to whether cluster on this z space or use other mehtod to split into clusters
#     z = np.load(data_path+ "/L-1" + config.z_name)  # -1 means no discrimation for labelsa, the same vae transform , orthogonal concept to whether cluster on this z space or use other mehtod to split into clusters
#     # index = np.load(data_path+"/global_index_cluster_data.npy")
#     # index = np.load(data_path + config.global_index_name)   # according to label, vae, vgmm, merge , cluster , per cluster-index(globally)
#     np.random.shuffle(z)
#     results = {}
#     mat = np.zeros((num_clusters,num_clusters))
#     for i in range(num_clusters):
#         xs = z[i*100:i*100+100]
#         for j in range(num_clusters):
#             xt = z[j*100:j*100+100]
#             # Compute distance kernels, normalize them and then display
#             n_samples = min(xs.shape[0], xt.shape[0])
#             xs = xs[:n_samples]
#             xt = xt[:n_samples]
#             C1 = sp.spatial.distance.cdist(xs, xs)
#             C2 = sp.spatial.distance.cdist(xt, xt)
#             C1 /= C1.max()
#             C2 /= C2.max()
#
#             p = ot.unif(n_samples)
#             q = ot.unif(n_samples)
#
#             gw0, log0 = ot.gromov.gromov_wasserstein(
#                 C1, C2, p, q, 'square_loss', verbose=True, log=True)
#
#             gw, log = ot.gromov.entropic_gromov_wasserstein(
#                 C1, C2, p, q, 'square_loss', epsilon=5e-4, log=True, verbose=True)
#
#             print('Gromov-Wasserstein distances between {}_{} clusters: {} '.format(i,j,str(log0['gw_dist'])) )
#             print('Entropic Gromov-Wasserstein distances between {}_{} clusters: {}'.format(i,j,str(log['gw_dist'])) )
#             results[str(i)+str(j)]={"GW":log0['gw_dist'],"EGW":log['gw_dist']}
#             mat[i,j] = log0['gw_dist']
#             pl.figure(1, (10, 5))
#             pl.subplot(1, 2, 1)
#             pl.imshow(gw0, cmap='jet')
#             pl.title('Gromov Wasserstein')
#
#             pl.subplot(1, 2, 2)
#             pl.imshow(gw, cmap='jet')
#             pl.title('Entropic Gromov Wasserstein')
#             pl.savefig(result_path + "/WD_TSNE{}_{}.jpg".format(i,j))
#     # print(results)
#     print(mat)
#     with open("wd_rand.txt", 'a') as lf:
#         lf.write(str(results))
#     return results



# computer gromov wasserstein distance on latent space z
def gromov_wasserstein_distance_latent_space_rand(data_path,num_labels,num_clusters,result_path,args):
    import scipy as sp
    import matplotlib.pylab as pl
    import ot
    # z = np.load(data_path+ "/L-1/z.npy")  # -1 means no discrimation for labelsa, the same vae transform , orthogonal concept to whether cluster on this z space or use other mehtod to split into clusters
    z = np.load(data_path+ "/L-1" + config.z_name,allow_pickle=True)  # -1 means no discrimation for labelsa, the same vae transform , orthogonal concept to whether cluster on this z space or use other mehtod to split into clusters
    np.random.shuffle(z)
    results = {}
    mat = np.zeros((num_clusters,num_clusters))
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=num_clusters)
    i = 0
    cluster_idx = {}
    for train_eval_idx, test_idx in kf.split(z):
        cluster_idx[str(i)] = test_idx
        i = i +1

    i = 0
    print(z.shape)
    for i in range(num_clusters):
        xs = z[cluster_idx[str(i)]]
        print(xs.shape)
        for j in range(num_clusters):
            xt = z[cluster_idx[str(j)]]
            print(xt.shape)
            # Compute distance kernels, normalize them and then display
            n_samples = min(xs.shape[0], xt.shape[0])
            if args.debug == True:
                n_samples = 100
            xs = xs[:n_samples]
            xt = xt[:n_samples]
            C1 = sp.spatial.distance.cdist(xs, xs)
            C2 = sp.spatial.distance.cdist(xt, xt)
            C1 /= C1.max()
            C2 /= C2.max()

            p = ot.unif(n_samples)
            q = ot.unif(n_samples)

            gw0, log0 = ot.gromov.gromov_wasserstein(
                C1, C2, p, q, 'square_loss', verbose=True, log=True)

            gw, log = ot.gromov.entropic_gromov_wasserstein(
                C1, C2, p, q, 'square_loss', epsilon=5e-4, log=True, verbose=True)

            print('Gromov-Wasserstein distances between {}_{} clusters: {} '.format(i,j,str(log0['gw_dist'])) )
            print('Entropic Gromov-Wasserstein distances between {}_{} clusters: {}'.format(i,j,str(log['gw_dist'])) )
            results[str(i)+str(j)]={"GW":log0['gw_dist'],"EGW":log['gw_dist']}
            mat[i,j] = log0['gw_dist']
            pl.figure(1, (10, 5))
            pl.subplot(1, 2, 1)
            pl.imshow(gw0, cmap='jet')
            pl.title('Gromov Wasserstein')

            pl.subplot(1, 2, 2)
            pl.imshow(gw, cmap='jet')
            pl.title('Entropic Gromov Wasserstein')
            pl.savefig(result_path + "/WD_TSNE{}_{}.jpg".format(i,j))
    # print(results)
    print(mat)
    with open("wd_rand.txt", 'a') as lf:
        lf.write(str(results))
    return results



def gromov_wasserstein_distance_latent_space_rand_emd(data_path,num_labels,num_clusters,result_path,args):
    import scipy as sp
    import matplotlib.pylab as pl
    import ot
    # z = np.load(data_path+ "/L-1/z.npy")  # -1 means no discrimation for labelsa, the same vae transform , orthogonal concept to whether cluster on this z space or use other mehtod to split into clusters
    z = np.load(data_path+ "/L-1" + config.z_name,allow_pickle=True)  # -1 means no discrimation for labelsa, the same vae transform , orthogonal concept to whether cluster on this z space or use other mehtod to split into clusters
    np.random.shuffle(z)
    results = {}
    mat = np.zeros((num_clusters,num_clusters))
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=num_clusters)
    i = 0
    cluster_idx = {}
    for train_eval_idx, test_idx in kf.split(z):
        cluster_idx[str(i)] = test_idx
        i = i +1

    i = 0
    print(z.shape)
    for i in range(num_clusters):
        xs = z[cluster_idx[str(i)]]
        print(xs.shape)
        for j in range(num_clusters):
            xt = z[cluster_idx[str(j)]]
            print(xt.shape)
            # Compute distance kernels, normalize them and then display
            n_samples = min(xs.shape[0], xt.shape[0])
            if args.debug == True:
                n_samples = 100
            xs = xs[:n_samples]
            xt = xt[:n_samples]
            M = sp.spatial.distance.cdist(xt, xs)
            M /= M.max()
            ds, dt = np.ones((len(xs),)) / len(xs), np.ones((len(xt),)) / len(xt)
            g0, loss = ot.emd(ds,dt, M, log = True)
            print('Gromov-Wasserstein distances between {}_{} clusters: {}--{} '.format(i,j,str(loss), str(np.sum(g0))) )
            #results[str(i)+str(j)]={"GW":log0['gw_dist'],"EGW":log['gw_dist']}
            results[str(i)+str(j)]= loss["cost"]
            mat[i,j] = loss["cost"]

            #pl.figure(1, (10, 5))
            #pl.subplot(1, 2, 1)
            #pl.imshow(gw0, cmap='jet')
            #pl.title('Gromov Wasserstein')
            #pl.subplot(1, 2, 2)
            #pl.imshow(gw, cmap='jet')
            #pl.title('Entropic Gromov Wasserstein')
            #pl.savefig(result_path + "/WD_TSNE{}_{}.jpg".format(i,j))
    # print(results)
    print(mat)
    with open("wd_rand.txt", 'a') as lf:
        lf.write(str(results))
    return results





if __name__ == '__main__':
    desc = "statistic"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--method',type=str,default='wd_vgmm', choices=['wd_vgmm', 'wd_vgmm_emd', 'wd_rand', 'wd_rand_emd', 'kde','distribution_y'],help="method of statistic ")
    parser.add_argument('--debug',default=False,type=bool,help="debug mode has smaller data")
    args = parser.parse_args()

    if args.method =='wd_vgmm':
        gromov_wasserstein_distance_latent_space_cluster(config.data_path,config.num_labels,config.num_clusters,config.data_path,args)
    elif args.method == 'wd_vgmm_emd':
        gromov_wasserstein_distance_latent_space_cluster_emd(config.data_path,config.num_labels,config.num_clusters,config.data_path,args)
    elif args.method == 'wd_rand':
        gromov_wasserstein_distance_latent_space_rand(config.data_path, config.num_labels, config.num_clusters,
                                                      config.data_path,args)
    elif args.method == 'wd_rand_emd':
        gromov_wasserstein_distance_latent_space_rand_emd(config.data_path, config.num_labels, config.num_clusters,
                                                      config.data_path,args)

    elif args.method == 'kde':
        # code for density estimator
        b = np.load(config.data_path + "/TSNE_transformed_data_dict.npy",allow_pickle=True)

        for i in range(config.num_clusters):
            xs = b.item().get(str(i))
            kernel_density_estimation_single_Cluster(xs, config.result_path, str(i))
            for j in range(config.num_clusters):
                if i != j:  # compare cluster $i$'s KDE with all other cluters, in pairwised way, to see if the estimated KDE surface changed
                    xt = b.item().get(str(j))
                    kernel_density_estimation(xs, xt, config.result_path,
                                              str(i) + str(j))  # merge two clusters data then do KDE
        density_estimation_GMM(xs, xt, config.result_path, str(i) + str(j))

    elif args.method == "distribution_y":
        counting_label(config.num_labels,config.num_clusters)

    # gromov_wasserstein_distance_TSNE(config.data_path,config.num_labels,config.num_clusters,config.data_path)
    # gromov_wasserstein_distance_TSNE_test(config.data_path,config.num_labels,config.num_clusters,config.data_path)
    # gromov_wasserstein_distance_latent_space_cluster(config.data_path,config.num_labels,config.num_clusters,config.data_path)
    #


    # kernel_density_estimation_on_latent_space(config.data_path,config.num_clusters)
