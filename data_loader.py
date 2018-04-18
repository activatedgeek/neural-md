from torch.utils.data import Dataset

# @TODO add PyTorch loader for easy batching
class PDBData(Dataset):
    def __init__(self, cluster_path, chains_path):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError
