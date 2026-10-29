from pathlib import Path

import anndata as ad
from scipy import sparse

FILES = Path(".").glob("*.h5ad")

for file in FILES:
    print(f"Processing {file}...")
    adata = ad.read_h5ad(file)
    if not sparse.issparse(adata.X):
        print(f"Converting dense matrix to sparse for {file}...")
        adata.X = sparse.csr_matrix(adata.X)
    adata.write_h5ad(file)
    print(f"Finished processing {file}.")
