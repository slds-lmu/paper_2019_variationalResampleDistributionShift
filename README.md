# paper_2019_rfms_representation
## Semi-supervised vae
- https://pyro.ai/examples/ss-vae.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html
- https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_pdf.html#sphx-glr-auto-examples-mixture-plot-gmm-pdf-py
- https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
- https://github.com/hwalsuklee/tensorflow-generative-model-collections
## python make
- https://pypi.org/project/py-make/
## arguments for main.py
  python main.py
  --cluster <True,False (default)>
  --dataset <'mnist', 'fashion-mnist' (default)>
  --z_dim <1-inf,62(default)>
  --labeled <True,False (default)>
  
## dependency
- pot: conda install -c conda-forge pot

## statistic
- conda install -c conda-forge pot
- python statistic.py
- before you run this command, you should previously run main.py and specifiy the data_path and result_path in config.py 