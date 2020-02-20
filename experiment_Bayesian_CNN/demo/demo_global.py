# Python program showing a use of
# global in nested function

def add():
	x = 15
	def change():
		global x
		x = 20
	print("Before making changing: ", x)
	print("Making change")
	change()
	print("After making change: ", x)
add()
print("value of x",x)

global y


def add2():
	y = 10
	def change():
		global y
		y = 80
	print("Before making changing: ", y)
	print("Making change")
	change()
	print("After making change: ", y)
add2()
print("value of x",y)

