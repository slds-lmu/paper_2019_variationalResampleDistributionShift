import tensorflow as tf
class VGMMDataset(Dataset):
    """Dataset after VGMM clustering"""
    def __init__(self, pattern, root_dir, transform=None, list_idx):
        """
        Args:
            pattern (string): Path to the npy file.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            list_idx (list): the list of indexes of the cluster to choose as trainset or testset
        """
        if not tf.gfile.Exists(data_path+"/global_index_cluster_data.npy"):
        _,global_index = concatenate_data_from_dir(data_path,num_labels=num_labels,num_clusters=num_cluster)
        else:global_index = np.load(data_path+"/global_index_cluster_data.npy",allow_pickle=True)
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        global_index.item().get(str(self$cluster_index))

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

class WVGMMDataset(VGMMDataset):
    def __init__(self, conf_manager, list_idx):
        super(WVGMMDataset, self).__init__(pattern = conf_manager.pattern, root_dir = conf_manager.root_dir, list_idx = list_idx)

if __name__ == '__main__':
    trainset = WVGMMDataset(list_idx = [1, 2, 3, 4])
    testset = WVGMMDataset(list_idx = [5])
