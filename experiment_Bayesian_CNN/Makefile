# rules: [b/f]bayesian-frequentist-[vgmm/rand]random cv or vgmm-vae-cv-[archname]
################
### 3conv3fc ###
################

bdebugvgmm:
	python refactor_main_Bayes4.py --cv_type "vgmm" --debug True --net_type "3conv3fc"

bdebugrand:
	python refactor_main_Bayes4.py --cv_type "rand" --debug True --net_type "3conv3fc"

bvgmm:
	python refactor_main_Bayes4.py --cv_type "vgmm" --net_type "3conv3fc" --g "0" # default debug False

brand:
	python refactor_main_Bayes4.py --cv_type "rand" --net_type "3conv3fc" --g "2" # default --debug False

fdebugrand:  # frequenst
	python refactor_main_Frequentist.py --cv_type "rand" --debug True --net_type "3conv3fc" # default --debug False

fdebugvgmm:  # frequenst
	python refactor_main_Frequentist.py --cv_type "vgmm" --debug True --net_type "3conv3fc" # default --debug False

frand:  # frequenst
	python refactor_main_Frequentist.py --cv_type "rand" --net_type "3conv3fc"   # default --debug False

fvgmm:  # frequenst
	python refactor_main_Frequentist.py --cv_type "vgmm" --net_type "3conv3fc"  # default --debug False

#f_org_3conv3fc:  # frequenst for cifar-100
#	python main_Frequentist.py --net_type "3conv3fc"  # original


###############
### AlexNet ###
###############

bdebugrand_alexnet:
	python refactor_main_Bayes4.py --cv_type "rand" --debug True --net_type "alexnet"

bvgmm_alexnet:
	python refactor_main_Bayes4.py --cv_type "vgmm" --net_type "alexnet" # default debug False

brand_alexnet:
	python refactor_main_Bayes4.py --cv_type "rand" --net_type "alexnet" # default --debug False

fdebugrand_alexnet:  # frequenst
	python refactor_main_Frequentist.py --cv_type "rand" --debug True --net_type "alexnet" # default --debug False

fdebugvgmm_alexnet:  # frequenst
	python refactor_main_Frequentist.py --cv_type "vgmm" --debug True --net_type "alexnet" # default --debug False

frand_alexnet:  # frequenst
	python refactor_main_Frequentist.py --cv_type "rand" --net_type "alexnet"  # default --debug False

fvgmm_alexnet:  # frequenst
	python refactor_main_Frequentist.py --cv_type "vgmm" --net_type "alexnet"  # default --debug False

#############
### LeNet ###
#############

bdebugvgmm_lenet:
	python refactor_main_Bayes4.py --cv_type "vgmm" --debug True --net_type "lenet"

bdebugrand_lenet:
	python refactor_main_Bayes4.py --cv_type "rand" --debug True --net_type "lenet"

bvgmm_lenet:
	python refactor_main_Bayes4.py --cv_type "vgmm" --net_type "lenet" # default debug False

brand_lenet:
	python refactor_main_Bayes4.py --cv_type "rand" --net_type "lenet" # default --debug False

fdebugrand_lenet:  # frequenst
	python refactor_main_Frequentist.py --cv_type "rand" --debug True --net_type "lenet" # default --debug False

fdebugvgmm_lenet:  # frequenst
	python refactor_main_Frequentist.py --cv_type "vgmm" --debug True --net_type "lenet" # default --debug False

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
