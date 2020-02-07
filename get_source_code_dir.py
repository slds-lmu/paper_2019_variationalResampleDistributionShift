from inspect import getsourcefile
import os.path
current_path = os.path.abspath(getsourcefile(lambda : 0))
current_dir = os.path.dirname(current_path)


