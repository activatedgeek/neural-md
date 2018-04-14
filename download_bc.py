import os
import requests

F = os.path.expanduser('~/Downloads/bc-70.out')


def download_pdb(id):
    path = 'data/pdb/{}.pdb'.format(id)
    if os.path.isfile(path):
        print('{} already exists.'.format(path))
        return

    url = 'https://files.rcsb.org/download/{}.pdb'.format(id)

    r = requests.get(url)
    with open(path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print('Saved {}.'.format(path))


def main():
    with open(F, 'r') as f:
        for line in f:
            for chain in line.split():
                pdb_id = chain.split('_')[0]
                download_pdb(pdb_id)


if __name__ == '__main__':
    main()
