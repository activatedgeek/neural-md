import os
import pyrosetta
from . import RESIDUE_LETTERS, MAX_CHI_COUNT, MAX_ATOM_COUNT


def aa_to_vector(pose, r_i):
    """
    Convert Amino Acids to vector of angles, atom coordinates and one-hot encoded letter type
    :param pose: PyRosetta Pose
    :param r_i: Residue number
    :return:
    """
    residue = pose.residue(r_i)

    # residue-type
    letter = residue.name1()
    assert letter in RESIDUE_LETTERS, '{} residue not supported'.format(residue.annotated_name())

    # angle parameters
    angles = [
        pose.phi(r_i),
        pose.psi(r_i),
        pose.omega(r_i),
    ]

    chi_angles = [
        *residue.chi(),
        # append extra zeros
        *([0.0] * (MAX_CHI_COUNT - len(residue.chi())))
    ]

    # normalize angles (-PI to PI)
    angles = list(map(lambda a: a / 180.0, angles))

    # atom coordinates, concatenation of x,y,z of each atom
    coords = []
    for atom in residue.atoms():
        coords.extend(atom.xyz())

    # append extra zeros for size consistency
    coords.extend([0.0, 0.0, 0.0] * (MAX_ATOM_COUNT - residue.natoms()))

    one_hot_type = [0] * len(RESIDUE_LETTERS)
    one_hot_type[RESIDUE_LETTERS.index(letter)] = 1

    feature_vector = [*coords, *angles, *chi_angles, *one_hot_type]
    return letter, feature_vector


def pre_compute_chains(pose, batch_size=None):
    """
    Precompute all the set of chains. Batch if needed. If a batched chain is
    of size smaller than batch, it is dropped. The output is a Python dictionary
    mapping from chain_id (string) to residue number limits (a tuple(int,int)),
    e.g. {'A1': (65,128)} represents a chain fragment of size 64 containing residues
    from atom number 65 to 128. This works because all residues of a chain occur in
    sequence in the PDB.
    :param pose:
    :param batch_size:
    :return: a dictionary mapping the chain_id to tuple of range of atom numbers
    """
    chains = {}
    for i in range(1, pose.total_residue() + 1):
        chain_id = pose.pdb_info().chain(i)
        if chain_id not in chains:
            chains[chain_id] = (i, i)
        chains[chain_id] = (chains[chain_id][0], i)

    if batch_size is not None:
        batched_chains = {}
        for k, (l, r) in chains.items():
            batch_count = (r - l + 1) // batch_size
            for b_id in range(batch_count):
                batched_chains[k + str(b_id)] = (l + b_id * batch_size, l + (b_id + 1) * batch_size - 1)

        return batched_chains

    return chains


def make_fasta(pdb_id, chain_id, seq):
    return '>{}_{}|PDBID|CHAIN|SEQUENCE\n{}\n'.format(pdb_id, chain_id, seq)


def pdb_to_chains(pdb_path):
    pdb_file = os.path.basename(pdb_path)
    pdb_id, _ = os.path.splitext(pdb_file)

    # Generate chains from PDB
    pose = pyrosetta.pose_from_pdb(pdb_path)
    chains = pre_compute_chains(pose, batch_size=64)

    for chain_id, (residue_l, residue_r) in chains.items():
        seq_string = ''
        chain_list = []
        valid_chain = True
        for r_id in range(residue_l, residue_r + 1):
            try:
                r_letter, r_vec = aa_to_vector(pose, r_id)
                seq_string += r_letter
                chain_list.append(r_vec)
            except AssertionError as e:
                print('Could not vectorize residue {}: {}, skipping chain {}'.format(r_id, str(e), chain_id))
                valid_chain = False
                break

        if valid_chain:
            yield chain_id, seq_string, chain_list
