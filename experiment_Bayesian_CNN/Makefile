# this Makefile contains all the commands needed to repeat the experiment
# rules: [b/f]bayesian-frequentist-[vgmm/rand]random cv or vgmm-vae-cv-[archname]
################
### 3conv3fc ###
################

bdebugvgmm:
	python main_resample_bayes.py --cv_type "vgmm" --debug  --net_type "3conv3fc"

bdebugrand:
	python main_resample_bayes.py --cv_type "rand" --debug  --net_type "3conv3fc"

bvgmm:
	python main_resample_bayes.py --cv_type "vgmm" --net_type "3conv3fc" --g "0" # default debug False

brand:
	python main_resample_bayes.py --cv_type "rand" --net_type "3conv3fc" --g "2" # default --debug False

fdebugrand:  # frequenst
	python refactor_main_Frequentist.py --cv_type "rand" --debug  --net_type "3conv3fc" # default --debug False

fdebugvgmm:  # frequenst
	python refactor_main_Frequentist.py --cv_type "vgmm" --debug  --net_type "3conv3fc" # default --debug False

frand:  # frequenst
	python refactor_main_Frequentist.py --cv_type "rand" --net_type "3conv3fc"   # default --debug False

fvgmm:  # frequenst
	python refactor_main_Frequentist.py --cv_type "vgmm" --net_type "3conv3fc"  # default --debug False

###############
### AlexNet ###
###############

bdebugrand_alexnet:
	python main_resample_bayes.py --cv_type "rand" --debug   --net_type "alexnet"

bvgmm_alexnet:
	python main_resample_bayes.py --cv_type "vgmm" --net_type "alexnet" # default debug False

brand_alexnet:
	python main_resample_bayes.py --cv_type "rand" --net_type "alexnet" # default --debug False

fdebugrand_alexnet:  # frequenst
	python refactor_main_Frequentist.py --cv_type "rand" --debug   --net_type "alexnet" # default --debug False

fdebugvgmm_alexnet:  # frequenst
	python refactor_main_Frequentist.py --cv_type "vgmm" --debug   --net_type "alexnet" # default --debug False

frand_alexnet:  # frequenst
	python refactor_main_Frequentist.py --cv_type "rand" --net_type "alexnet"  # default --debug False

fvgmm_alexnet:  # frequenst
	python refactor_main_Frequentist.py --cv_type "vgmm" --net_type "alexnet"  # default --debug False

#############
### LeNet ###
#############

bdebugvgmm_lenet:
	python main_resample_bayes.py --cv_type "vgmm" --debug   --net_type "lenet"

bdebugrand_lenet:
	python main_resample_bayes.py --cv_type "rand" --debug   --net_type "lenet"

bvgmm_lenet:
	python main_resample_bayes.py --cv_type "vgmm" --net_type "lenet" # default debug False

brand_lenet:
	python main_resample_bayes.py --cv_type "rand" --net_type "lenet" # default --debug False

fdebugrand_lenet:  # frequenst
	python refactor_main_Frequentist.py --cv_type "rand" --debug   --net_type "lenet" # default --debug False

fdebugvgmm_lenet:  # frequenst
	python refactor_main_Frequentist.py --cv_type "vgmm" --debug   --net_type "lenet" # default --debug False

frand_lenet:  # frequenst
	python refactor_main_Frequentist.py --cv_type "rand" --net_type "lenet"  # default --debug False

fvgmm_lenet:  # frequenst
	python refactor_main_Frequentist.py --cv_type "vgmm" --net_type "lenet"  # default --debug False


#################################
### original train-test split ###
#################################

fixed-frequenst-alex:
	python main_Frequentist.py --dataset fashion-mnist --net_type "alexnet"

fixed-bayesian-alex:
	python main_Bayes.py --dataset fashion-mnist --net_type "alexnet"

fixed-frequenst-lenet:
	python main_Frequentist.py --dataset fashion-mnist --net_type "lenet"

fixed-bayesian-lenet:
	python main_Bayes.py --dataset fashion-mnist --net_type "lenet"

fixed-frequenst-3conv3fc:
	python main_Frequentist.py --dataset fashion-mnist --net_type "3conv3fc"

fixed-bayesian-3conv3fc:
	python main_Bayes.py --dataset fashion-mnist --net_type "3conv3fc"
