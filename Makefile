build:
	python main.py --cluster True


label: build
	 python main.py --labeled True --cluster True


statisic: label
	 python statistic.py
