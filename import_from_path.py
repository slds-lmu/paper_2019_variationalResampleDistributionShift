import importlib.util
def load_config(path):
  spec = importlib.util.spec_from_file_location("", path)
  foo = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(foo)
  return foo

#m = load_config("/home/sunxd/vgmmvae/config.py")
#m.dataset_name
#m = load_config("../config.py")
#m.dataset_name
