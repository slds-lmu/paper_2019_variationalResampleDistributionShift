# Variational Gaussian Mixture Model Variational AutoEncoder Cross Validation Reampling onBayesian and Frequest Neural Networks

## Credit
- VAE code is adapted from this project
https://github.com/hwalsuklee/tensorflow-generative-model-collections.git
- Bayesian CNN and Frequenst ones are adapted from the following projects
https://github.com/felix-laumann/Bayesian_CNN
https://github.com/kumar-shridhar/PyTorch-BayesianCNN

 
## How to reproduce the experiment

clone this repository and navigate into the root directory

### Install Dependencies
- pip install -r requirements_cpu.txt
- pip install Cython
- pip install pot, if you have conda, install pot with:  conda install -c conda-forge pot

#### Tips on generating dependencies file from a project
- pip freeze
- see here: https://medium.com/python-pandemonium/better-python-dependency-and-package-management-b5d8ea29dff1

#### Testing if all dependencies met and the code could run properly
- make test
- make test_label
- The above command will use minimal epochs to test if the whole process below works

### Experiment Process

#### arguments for main.py in project root
  python main.py
  --cluster <True,False (default)>
  --dataset <'cifar10', 'mnist', 'fashion-mnist' (default)>
  --z_dim <1-inf,62(default)>
  --labeled <True,False (default)>

### Generating the embeding and assigning each image to its pseudo subdomain
- learn an embedding with respect to data from all classes
    - make build 
    - for fashion-mnist, it takes 30mins on a fujitsu-celcius workstation
    - equivalently you could do  'python main.py --cluster True'
    - cluster directly here won't be used since the cluster will most probably correspond to different classes, so we cluster with respect to each class label and merge them

- learn an embedding with respect to each class label and merge randomly
    - make label 
    - for fashion-mnist, it takes 1 hours on fujitsu-celcius workstation
    - equivalently you could do 'python main.py --labeled True --cluster True' 

- The result of the main routine (embed_cluster)  generate a file which stores the global index, which is a dictionary with key corresponding to cluster index, while value corresponding to the absolute index of the original data

### After the vae-vgmm subdomain assignment

#### Result files
- config.py is a volatile file storing information to retrieve results, which will be rewriten each time the embed_cluster routine is runned
- in folder 'results', for example, one possible folder name can be VAE-fashion-mnist-64-10 where 64 is
the batch size and 10 is the length of the latent dimension, inside which L0 to L9 stores the
results for each class label and L-1(not label-wise) stores the global embedding for all classes instances
- in folder checkpoint/VAE-fashion-mnist-64-10/L-1, VAE will store results for global embeding for all classes, delete this folder if you want to rerun experiment

#### Get access to the result
- after running the experiment, one could load config again to get a subset of the original merged train-test data from SubdomainDataset, which is inherited from torch.utils.data.dataset.Dataset

  ```
  import config
  from mdataset_class import SubdomainDataset
  subds = SubdomainDataset(config_volatile=config, list_idx=[0, 2], transform=None)
  ```

#### Evaluate different Neural Network Prediction Performance on the artificial sub-domains
- change directory to experiment_Bayesian_CNN
- make rand frand|vgmm|fvgmm_alexnet

#### Statistics and Visualization
- before you run this command, you should previously run 
    - make build
    - make label
- change directory to root folder
- 'make wasser_cv_emd' : compute wasserstein distance for random cross validation
- 'make wasser_vgmm_emd': compute wasserstein distance for vgmm-vae cross validation
- 'make t-SNE': generate t-SNE plot for all data divided by vgmm-vae  (results could be stored in /results/VAE_fashion-mnist_64_62 for example)
- 'make distribution_y': plot the histogram of class distribution for each cluster, result is store in distribution_y.txt

#### Reproduce Plotting in the paper
- go to  /plots and execute the R code to generate the beautiful ggplot

## Code structure 
- utils_parent.py is used in neural network classification for getting data and misc things
- config_manager.py took arguments from main()'s parser and also hard codedly defined some paths to
  store intermediate files

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

### parallel job in python
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

