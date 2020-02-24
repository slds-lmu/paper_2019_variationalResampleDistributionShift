# Variational Gaussian Mixture Model Variational AutoEncoder Cross Validation Reampling onBayesian and Frequest Neural Networks


## How to reproduce the experiment

clone this repository and navigate into the project directory

The following has been tested with Python version 3.7.4.

### Install Dependencies
- `pip install -r requirements_compact.txt`
- `pip install Cython`
- `pip install pot` or `conda install -c conda-forge pot`

#### Testing if all dependencies met and the code could run properly
- `make test`
- `make test_label`
- The above command will use minimal epochs to test if the whole process below works
- The above command will generate results with different folder names so will not affect the major
  result

### Experiment Process

#### arguments for main.py in project root
```
  python main.py
  --cluster <True, False(default)>
  --dataset <'cifar10', 'fashion-mnist' (default)>
  --z_dim <1-inf,62(default)>
  --labeled <True, False(default)>
  --result_dir<String, "results(default)>
  --persist_file_path<String, "ignore_flat_rst_meta_persist_FashionMNIST.py"(default)>
```
- 'persist_file_path': A volatile python file name(name must end with .py!) storing information to retrieve results, which will be overwritten each time the `embed_cluster` routine is runned
- `result_dir`: in folder 'result_dir' relative to the current directory, results will be stored.
For example, one possible folder name can be VAE-fashion-mnist-64-10 where 64 is the batch size and 10 is the length of the latent dimension, inside which L0 to L9 stores the results for each class label and L-1(-1 means for all classes) stores the global embedding for all classes instances. Note that these results won't be overwritten!
- in folder checkpoint/VAE-fashion-mnist-64-10/L-1, VAE will store results for global embeding for all classes, delete this folder if you want to rerun experiment



### Generating the embeding and assigning each image to its pseudo subdomain
- learn an embedding with respect to data from all classes: `python main.py --dataset fashion-mnist`
    - equivalently you could do `make common_embed`
    - cluster directly here won't be used since the cluster will most probably correspond to different classes, so we cluster with respect to each class label and merge them: `python main.py --cluster`, but this is not used in the experiment
    - for fashion-mnist, it takes 20 mins on titan gpu

- learn an embedding with respect to each class label and merge randomly: `python main.py --dataset fashion-mnist --labeled --cluster`
    - equivalently you could do `make label` 
    - for fashion-mnist, it takes 1 hours on fujitsu-celcius workstation, 20 mins on titan gpu

- The result of the main routine `embed_cluster()`  generate a file which stores the global index, which is a dictionary with key corresponding to cluster index, while value corresponding to the absolute index of the original data. The path of this result file is stored in a volatile python file, see below "Result files"

### After the vae-vgmm subdomain assignment

#### Evaluate different Neural Network Prediction Performance on the artificial sub-domains
- copy your "persist_file_path" for example "ignore_flat_rst_meta_persist_FashionMNIST.py" into
  "experiment_Bayesian_CNN" folder
- change directory to `experiment_Bayesian_CNN`
- test if code works by `make bdebugvgmm` and `make bdebugrand`
- check the Makefile for other tasks like  `make bvgmm_alexnet`, `make fvgmm_alexnet` etc

#### Statistics and Visualization
- before you run this command, you should finish the vae-vgmm subdomain assignment first with result
  files saved on disk
- change directory to root folder
- `make wasser_cv_emd` : compute wasserstein distance for random cross validation
- `make wasser_vgmm_emd`: compute wasserstein distance for vgmm-vae cross validation
- `make t-SNE`: generate t-SNE plot for all data divided by vgmm-vae  (results could be stored in ./results/VAE_fashion-mnist_64_62 for example)
- `make distribution_y`: plot the histogram of class distribution for each cluster, result is store in distribution_y.txt

#### Reproduce Plotting in the paper
- go to  ./Rsrc4plots and execute the R code to generate the beautiful ggplot

## Code structure 
- utils_parent.py is used in neural network classification for getting data and misc things
- config_manager.py took arguments from main()'s parser and also hard codedly defined some paths to
  store intermediate files

## Credit and Licences
- See folder ./licences
- VAE code is adapted from this project
https://github.com/hwalsuklee/tensorflow-generative-model-collections.git
- Bayesian CNN and Frequenst ones are adapted from the following projects
https://github.com/felix-laumann/Bayesian_CNN
https://github.com/kumar-shridhar/PyTorch-BayesianCNN

## Misc Resources

### Tips on generating dependencies file from a project
- `pip freeze` will print out all package versions on the computer
- see here: https://medium.com/python-pandemonium/better-python-dependency-and-package-management-b5d8ea29dff1

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
- %env JOBLIB_TEMP_FOLDER=/tmp

