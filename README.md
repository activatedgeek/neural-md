# neural-md

* Use [`download-bc.py`](./download_bc.py) to download all the PDBs in a cluster file

* Use [`pdb2data.py`](./pdb2data.py) to generate feature vectors for each chain.

## Sample usage of the data loader


```python
import data_loader
import random

chains_data = data_loader.PDBChainsDataLoader('data/chains')
count = len(chains_data)

chain_tensor, seq_string = chains_data[random.randrange(count)]

print(chain_tensor) # FloatTensor of shape (L, 109)
print(seq_string) # String of length L
```

