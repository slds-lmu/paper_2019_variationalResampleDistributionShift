# Variational Gaussian Mixture model Cross Validation resampling of Bayesian and Frequest Neural Networks
- VAE code is adapted from this project
https://github.com/hwalsuklee/tensorflow-generative-model-collections.git
- Bayesian CNN and Frequenst ones are adapted from the following project
https://github.com/felix-laumann/Bayesian_CNN

## Dependencies
- pip install Cython
- pip install pot
- if you have conda, install pot with:  conda install -c conda-forge pot
- pip install -r requirements.txt

### Tips on generating dependencies file
pip freeze
https://medium.com/python-pandemonium/better-python-dependency-and-package-management-b5d8ea29dff1

 
## How to run or reproduce experiment

### Preparation
- clone a new repository and go to the root directory
- make build (30mins for the first run on fujitsu-celcius) # equivalent to python main.py --cluster True (train vae on all data and cluster), results could be stored in results/VAE_fashion-mnist_64_62/L-1 for example
- make label (1 hours on fujitsu-celcius) # equivalent to python main.py --labeled True --cluster True (train vae according to label and cluster each label, then merge), results could be stored in results/VAE_fashion-mnist_64_62/L0 unutil results/VAE_fashion-mnist_64_62/L9 for example
- results for the two steps are stored in results/VAE_fashion-mnist_64_62 for example, where 62 is the latent space dimension
of VAE (see configuration file named config.py), while data is stored in /data/FashionMNIST for example
### Evaluate Neural Network
- change directory to refactor_Bayesian_CNN
- make rand frand|vgmm|fvgmm_alexnet

### statistic
- before you run this command, you should previously run make build and make label
- change directory to root folder
- make wasser_cv_emd : compute wasserstein distance for random cross validation
- make wasser_vgmm_emd: compute wasserstein distance for vgmm-vae cross validation
- make t-SNE: generate t-SNE plot for all data divided by vgmm-vae  (results could be stored in /results/VAE_fashion-mnist_64_62 for example)
- make distribution_y: plot the histogram of class distribution for each cluster, result is store in distribution_y.txt

### Plotting
- go to  /plots and use the R code to generate the beautiful ggplot



## Guide to the code

### Configuration
in root folder and refactor_Bayesian_CNN, files start with config stores global configuration parameters.

### arguments for main.py in project root
  python main.py
  --cluster <True,False (default)>
  --dataset <'mnist', 'fashion-mnist' (default)>
  --z_dim <1-inf,62(default)>
  --labeled <True,False (default)>


## Misc Resources

### Semi-supervised vae
- https://pyro.ai/examples/ss-vae.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html
- https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_pdf.html#sphx-glr-auto-examples-mixture-plot-gmm-pdf-py
- https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
- https://github.com/hwalsuklee/tensorflow-generative-model-collections
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
- https://pot.readthedocs.io/en/stable/auto_examples/plot_gromov.html
- https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html

### python make
- https://pypi.org/project/py-make/
- https://snakemake.readthedocs.io/en/stable/
- https://sacred.readthedocs.io/en/latest/apidoc.html  decorator for reproducible experiment

### parallel
- https://github.com/horovod/horovod#pytorch
- https://skorch.readthedocs.io/en/stable/user/parallelism.html
- https://towardsdatascience.com/speed-up-your-algorithms-part-1-pytorch-56d8a4ae7051  **share_memory**
- https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
- https://medium.com/@iliakarmanov/multi-gpu-rosetta-stone-d4fa96162986
- https://github.com/pytorch/examples/blob/master/mnist_hogwild/main.py
- https://pytorch.org/docs/stable/notes/multiprocessing.html

#### to avoid no space on device problem when run parallel in pytorch
- The problem was resolved by setting the following env variable in our Dockerfile: ENV JOBLIB_TEMP_FOLDER=/tmp.
- https://stackoverflow.com/questions/44664900/oserror-errno-28-no-space-left-on-device-docker-but-i-have-space
- docker run --shm-size=512m <image-name>
- docker system prune -af
- https://stackoverflow.com/questions/40115043/no-space-left-on-device-error-while-fitting-sklearn-model 
- It seems, that your are running out of shared memory (/dev/shm when you run df -h). Try setting JOBLIB_TEMP_FOLDER environment variable to something different: e.g., to /tmp. 

%env JOBLIB_TEMP_FOLDER=/tmp

