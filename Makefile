build:
	python main.py --cluster True


label:
	 python main.py --labeled True --cluster True


statisic: ./results/VAE_fashion-mnist_64_62/L-1/TSNE_transformed_data_dict.json
	 python statistic.py
