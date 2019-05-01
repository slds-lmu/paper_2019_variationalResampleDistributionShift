#pool = multiprocessing.Pool(4)
#out1, out2, out3 = zip(*pool.map(calc_stuff, range(0, 10 * offset, offset)))

from multiprocessing import Pool

def f(x):
    return x*x

if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3]))
