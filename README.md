# neural-md

* Use [`download-bc.py`](./helpers/download_bc.py) to download all the PDBs in a cluster file

* Use [`pdb2data.py`](./helpers/pdb2data.py) to generate and save numpy feature vectors for each chain.

* Use [`get_max_counts.py`](./helpers/get_max_counts.py) to get max counts for atoms and chi angles among supported residues

## Setup

Run the following from the root of the directory, if `neural_md` module is not found.

```shell
$ export PYTHONPATH=$(pwd):$PYTHONPATH
```

## Sample usage of the data loader


```python
import neural_md import PDBChainsDataLoader
import random

chains_data = PDBChainsDataLoader('data/chains')
count = len(chains_data)

chain_tensor, seq_string = chains_data[random.randrange(count)]

print(chain_tensor) # FloatTensor of shape (L, 109)
print(seq_string) # String of length L
```

