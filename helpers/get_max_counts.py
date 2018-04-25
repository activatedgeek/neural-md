import pyrosetta
from neural_md import RESIDUE_LETTERS

ATOM_COUNT = [None] * len(RESIDUE_LETTERS)
CHI_COUNT = [None] * len(RESIDUE_LETTERS)


def main():
    pyrosetta.init()

    for id, c in enumerate(RESIDUE_LETTERS):
        atom_count = None
        chi_count = None
        try:
            pose = pyrosetta.pose_from_sequence(c, 'fa_standard')
            residue = pose.residue(1)
            atom_count = pose.residue(1).natoms()
            chi_count = len(residue.chi())
        except Exception as e:
            print('Error processing {}: {}'.format(c, e))

        ATOM_COUNT[id] = atom_count
        CHI_COUNT[id] = chi_count

    print('Max atom count: {}'.format(max(ATOM_COUNT)))
    print('Max chi angles count: {}'.format(max(CHI_COUNT)))


if __name__ == '__main__':
    main()
