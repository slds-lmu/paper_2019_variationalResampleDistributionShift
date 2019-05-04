INPUT4STAT = ./results/VAE_fashion-mnist_64_62/L-1/TSNE_transformed_data_dict.npy
build:
	python main.py --cluster True  # the same vae to map all the data, then use vgmm. This is useful to calculate the wasserstein distance


label:
	 python main.py --labeled True --cluster True     # use different vae to each label, then cluster according to each label
convnet: 
	python convnet.py --epoch 10    # compare of cv and rfms
test:  # only run 1 epoch to see if the code works 
	python main.py --epoch 1 --z_dim 10 --cluster True --num_clusters 3
test_label:
	python main.py --labeled True --epoch 1 --z_dim 10 --cluster True --num_clusters 3
test_convnet:
	python convnet.py


statisic: ${INPUT4STAT}  # depends on build(make coordinate for each instance) and label(assign cluster to each point)
	 python statistic.py


#compute wasserstein distance of vgmm cluster
wasser_vgmm:
	python statistic.py --method wd_vgmm

wasser_vgmm_emd:
	python statistic.py --method wd_vgmm_emd

#compute wasserstein distance of random-pick 100 sample
wasser_cv:
	python statistic.py --method wd_rand

wasser_cv_emd:
	python statistic.py --method wd_rand_emd

#kde plot
kde:
	python statistic.py --method kde

# distribution of y:
distribution_y:
    python statistic.py -- method distribution_y
test_wasser_vgmm:
	python statistic.py --method wd_vgmm --debug True

#compute wasserstein distance of random-pick 100 sample
test_wasser_cv:
	python statistic.py --method wd_rand --debug True
#kde plot
test_kde:
	python statistic.py --method kde  --debug True

## for wasserstein: please run make wasser_vgmm_emd, make wasser_cv_emd

# t-SNE plot
t-SNE:
    python visualization.py