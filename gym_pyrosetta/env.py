import os
import gym
from gym import spaces
import numpy as np
import pyrosetta


class PyRosettaEnv(gym.Env):
    def __init__(self, chain_id, chains_dir='data/chains'):
        super(PyRosettaEnv, self).__init__()

        self.chain_id = chain_id
        self.chains_dir = chains_dir

        # Gym params
        self.action_space = spaces.Discrete(180)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(109, 64), dtype=np.float32)

        # Internal pose to run the episode
        self.sf = pyrosetta.get_fa_scorefxn()
        self.pose = None
        self.state = None

    def reset(self):
        pdb_id = self.chain_id.split('_')[0]
        npy_file = os.path.join(self.chains_dir, pdb_id, self.chain_id + '.npy')
        self.state = np.load(npy_file)
        seq_string = self._get_seq_string()

        self.pose = pyrosetta.pose_from_sequence(seq_string)
        self._set_pose()

        return self.state

    def step(self, action):
        pass

    def render(self, mode='human'):
        raise NotImplementedError

    def _get_seq_string(self):
        pdb_id = self.chain_id.split('_')[0]
        faa_file = os.path.join(self.chains_dir, pdb_id, pdb_id + '.faa')
        with open(faa_file, 'r') as fasta:
            found = False
            for line in fasta:
                if line[0] == '>':
                    chain_id = line[1:].split('|')[0]
                    if chain_id == self.chain_id:
                        found = True
                else:
                    if found:
                        return line.rstrip()

        assert False, 'could not find sequence string'

    def _set_pose(self):
        nr = self.pose.total_residue()
        for r_i in range(1, nr + 1):
            data = self.state[:, r_i - 1]
            na = self.pose.residue(r_i).natoms()
            for a_i in range(na):
                coords = list(data[a_i * 3:(a_i + 1) * 3])
                self.pose.residue(r_i).set_xyz(a_i + 1, pyrosetta.rosetta.numeric.xyzVector_double_t(*coords))

            self.pose.set_phi(r_i, data[81] * 180.0)
            self.pose.set_psi(r_i, data[82] * 180.0)
            self.pose.set_omega(r_i, data[83] * 180.0)

            chi_length = len(self.pose.residue(r_i).chi())
            for i in range(chi_length):
                self.pose.set_chi(i + 1, r_i, data[84 + i] * 180.0)
