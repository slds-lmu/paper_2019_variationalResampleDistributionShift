INPUT4STAT = ./results/VAE_fashion-mnist_64_62/L-1/TSNE_transformed_data_dict.npy
build:
	python main.py --cluster True


label:
	 python main.py --labeled True --cluster True

test:  # only run 1 epoch to see if the code works 
	python main.py --epoch 1

statisic: ${INPUT4STAT}
	 python statistic.py
