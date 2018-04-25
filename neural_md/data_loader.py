import os
import numpy as np
import torch
from torch.utils.data import Dataset
from . import FASTA_EXT, NPY_EXT, DATA_DIM


class PDBChainsDataLoader(Dataset):
    def __init__(self, chains_path):
        """
        Takes a folder which contains chain ".npy" files and ".fst" files
        :param chains_path: path to chains folder
        """
        self.chains_path = chains_path

        # Get a list of all chains from all FASTA files
        # self.seq_strings = []
        self.chains = []

        for pdb_id in os.listdir(self.chains_path):
            if os.path.isdir(os.path.join(self.chains_path, pdb_id)):
                faa_path = os.path.join(self.chains_path, pdb_id, pdb_id + FASTA_EXT)
                with open(faa_path, 'r') as fasta:
                    for line in fasta:
                        if line[0] == '>':
                            chain_id = line[1:].split('|')[0]
                            self.chains.append(chain_id)
                        # else:
                        #     self.seq_strings.append(line)

    def __len__(self):
        return len(self.chains)

    def __getitem__(self, item):
        """
        @NOTE: Only reading numpy file here to be memory efficient
        """
        chain_id = self.chains[item]
        npy_path = os.path.join(self.chains_path, chain_id.split('_')[0], chain_id + NPY_EXT)
        chain_data = np.load(npy_path)
        assert chain_data.shape == DATA_DIM
        chain_tensor = torch.from_numpy(chain_data).float()
        return chain_tensor
