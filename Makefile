#python main.py
#--cluster <True,False (default)>
#--dataset <'cifar10', 'mnist', 'fashion-mnist' (default)>
#--z_dim <1-inf,62(default)>
#--labeled <True,False (default)>

# Test on a laptop, use all available options (cluster for example) to check if every function works
test:  # only run 1 epoch to see if the code works 
	python main.py --dataset "fashion-mnist" --epoch 1 --z_dim 10 --cluster True --num_clusters 3
test_label:
	python main.py --dataset "fashion-mnist" --labeled True --epoch 1 --z_dim 10 --cluster True --num_clusters 3
test_convnet:
	python convnet.py



build:
	python main.py --cluster True  # the same vae to map all the data, then use vgmm. This is useful to calculate the wasserstein distance


label:
	 python main.py --labeled True --cluster True     # use different vae to each label, then cluster according to each label

convnet: 
	python convnet.py --epoch 10    # compare of cv and rfms


# Statistic
## depends on build(make coordinate for each instance) and label(assign cluster to each point)

### compute wasserstein distance of vgmm cluster
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
	python statistic.py --method distribution_y
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
