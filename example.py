import inspect
import config
from mdataset_class import SubdomainDataset
subds = SubdomainDataset(config_volatile=config, list_idx=[0, 2], transform=None)
# the following code shows SubdomainDataset is a subclass of torch.utils.data.dataset.Dataset
type(subds)
inspect.getmro(SubdomainDataset)
#
