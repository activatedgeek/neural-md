import os
import gym
from gym import spaces
import numpy as np
import pyrosetta
from neural_md import aa_to_vector


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

    def reset(self):
        pdb_id = self.chain_id.split('_')[0]
        npy_file = os.path.join(self.chains_dir, pdb_id, self.chain_id + '.npy')
        state = np.load(npy_file)
        seq_string = self._get_seq_string()

        self.pose = pyrosetta.pose_from_sequence(seq_string)
        self._set_pose(state)

        return state

    def step(self, action):
        """
        :param action: 7 x chain_length numpy ndarray
        :return:
        """
        self._set_pose_angles(action)

        return self._get_pose_state_vector(), self.sf(self.pose), False, {}

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

    def _get_pose_state_vector(self):
        result = []
        for r_i in range(1, self.pose.total_residue() + 1):
            _, vec = aa_to_vector(self.pose, r_i)
            result.append(vec)

        return np.array(result)

    def _set_pose(self, state):
        nr = self.pose.total_residue()
        for r_i in range(1, nr + 1):
            data = state[:, r_i - 1]
            na = self.pose.residue(r_i).natoms()

            # Set all coordinates
            for a_i in range(na):
                coords = list(data[a_i * 3:(a_i + 1) * 3])
                self.pose.residue(r_i).set_xyz(a_i + 1, pyrosetta.rosetta.numeric.xyzVector_double_t(*coords))

            self.pose.set_phi(r_i, data[81] * 180.0)
            self.pose.set_psi(r_i, data[82] * 180.0)
            self.pose.set_omega(r_i, data[83] * 180.0)

            chi_length = len(self.pose.residue(r_i).chi())
            for i in range(chi_length):
                self.pose.set_chi(i + 1, r_i, data[84 + i] * 180.0)

    def _set_pose_angles(self, angles):
        nr = self.pose.total_residue()
        for r_i in range(1, nr + 1):
            data = angles[:, r_i - 1]

            self.pose.set_phi(r_i, data[0] * 180.0)
            self.pose.set_psi(r_i, data[1] * 180.0)
            self.pose.set_omega(r_i, data[2] * 180.0)

            chi_length = len(self.pose.residue(r_i).chi())
            for i in range(chi_length):
                self.pose.set_chi(i + 1, r_i, data[3 + i] * 180.0)
