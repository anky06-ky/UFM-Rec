"""Dense 256-d LSA of the train-fitted TF-IDF matrix, as an extra UFM item feature table.

Row 0 is padding (zeros), item row i+1 = catalog row i, L2-normalized, float16.
The SVD is fitted on train-visible items only (train_count > 0), as the TF-IDF vocabulary was.
"""
import csv, gzip, json, sys, time
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

data = Path(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=False)
t0 = time.time()
X = sparse.load_npz(data / 'content/tfidf_all.npz').tocsr().astype(np.float32)
counts = []
with gzip.open(data / 'content/items.csv.gz', 'rt', encoding='utf-8') as f:
    for r in csv.DictReader(f): counts.append(int(r['train_count']))
counts = np.asarray(counts)
fit = np.flatnonzero(counts > 0)
svd = TruncatedSVD(n_components=256, algorithm='randomized', n_iter=5, random_state=42)
svd.fit(X[fit])
print('fit done', round(time.time() - t0), 's, explained', float(svd.explained_variance_ratio_.sum()), flush=True)
Z = np.zeros((X.shape[0] + 1, 256), dtype=np.float32)
for s in range(0, X.shape[0], 100000):
    Z[1 + s:1 + s + 100000] = svd.transform(X[s:s + 100000])
norms = np.linalg.norm(Z, axis=1, keepdims=True)
present = norms[:, 0] > 1e-8
Z = np.where(present[:, None], Z / np.where(norms > 0, norms, 1), 0).astype(np.float16)
np.save(out / 'lsa.npy', Z); np.save(out / 'present.npy', present)
json.dump(dict(components=256, fit_items=int(len(fit)), rows=int(Z.shape[0]), present=int(present.sum()),
               explained_variance=float(svd.explained_variance_ratio_.sum()), seconds=round(time.time() - t0)),
          open(out / 'complete.json', 'w'), indent=2)
print('done', open(out / 'complete.json').read(), flush=True)
