"""Full-catalog inference for research demo, never reads evaluation/test labels."""
import argparse
import csv
import gzip
import heapq
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import numpy as np
from scipy import sparse
from evaluate_content_baseline import tie_noise

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/processed/toys_games_full_temporal'


def safe_image(url):
    p = urlsplit(url)
    return url if (p.scheme=='https' and p.hostname=='m.media-amazon.com' and p.port in (None,443)
                   and not p.username and not p.password and p.path.startswith('/images/I/')) else ''


class Catalog:
    def __init__(self,data):
        self.data = Path(data)
        self.asins,self.counts,self.titles,self.images = [],[],[],[]
        with gzip.open(self.data/'content/items.csv.gz','rt',encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if int(r['row_index']) != len(self.asins): raise ValueError('Non-contiguous catalog.')
                self.asins.append(r['parent_asin']); self.counts.append(int(r['train_count']))
        self.lookup = {a:i for i,a in enumerate(self.asins)}
        if not self.asins or len(self.lookup)!=len(self.asins) or min(self.counts)<0:
            raise ValueError('Invalid mapping/counts.')
        with gzip.open(self.data/'content/products_text.csv.gz','rt',encoding='utf-8') as f:
            for i,r in enumerate(csv.DictReader(f)):
                if i>=len(self.asins) or r['parent_asin'] != self.asins[i]: raise ValueError('Title mapping mismatch.')
                self.titles.append(r.get('title','')[:240] or r['parent_asin'])
        if len(self.titles)!=len(self.asins): raise ValueError('Title coverage mismatch.')
        self.images = ['']*len(self.asins)
        path = self.data/'foundation_catalog_v1/catalog.csv.gz'
        if path.exists():
            with gzip.open(path,'rt',encoding='utf-8') as f:
                n = 0
                for i,r in enumerate(csv.DictReader(f)):
                    if i>=len(self.asins) or int(r['row_index'])!=i or r['parent_asin']!=self.asins[i]:
                        raise ValueError('Image catalog mapping mismatch.')
                    try: self.images[i] = safe_image(r['image_url'])
                    except ValueError: self.images[i] = ''
                    n += 1
                if n!=len(self.asins): raise ValueError('Image mapping incomplete.')
        self.counts = np.asarray(self.counts,dtype=np.int64)

    def rows(self,asins):
        if len(asins)>20: raise ValueError('History limited to 20 items.')
        try: return [self.lookup[a] for a in asins]
        except KeyError as e: raise ValueError('Unknown ASIN: '+str(e.args[0])) from None

    def item(self,row):
        n = int(self.counts[row])
        regime = 'zero_shot' if n==0 else 'extreme_cold' if n<=5 else 'cold' if n<=20 else 'warm'
        return dict(asin=self.asins[row],title=self.titles[row],image=self.images[row],train_count=n,regime=regime)

    def search(self,q,limit=20):
        q = q.strip().lower()[:80]
        if len(q)<2: return []
        found = []
        for i,(asin,title) in enumerate(zip(self.asins,self.titles)):
            if q in asin.lower() or q in title.lower():
                found.append(self.item(i))
                if len(found)>=limit: break
        return found


class ContentBackend:
    name = 'TF-IDF content baseline'
    warning = 'Demo baseline TF-IDF, không phải UFM đã train. Điểm xếp hạng không phải xác suất mua hàng.'
    def __init__(self,catalog):
        self.catalog = catalog
        self.matrix = sparse.load_npz(catalog.data/'content/tfidf_all.npz').tocsr()
        if self.matrix.shape[0]!=len(catalog.asins) or not np.isfinite(self.matrix.data).all():
            raise ValueError('TF-IDF mapping/finite check failed.')

    def recommend(self,asins,k=10):
        history = self.catalog.rows(asins)
        if not 1<=k<=50: raise ValueError('Top K must be 1-50.')
        past = history[-10:]
        available = np.diff(self.matrix.indptr)>0
        past = [r for r in past if available[r]]
        popularity = np.log1p(self.catalog.counts)
        if past:
            profile = np.asarray(self.matrix[past].sum(0)).ravel()
            scores = np.asarray(self.matrix.dot(profile)).ravel().astype(np.float64)
            scores += popularity/(float(popularity.max()) or 1)*1e-7
        else: scores = popularity.astype(np.float64)
        # Full catalog, int64 ranks (do not reuse sampled uint8 rank helper).
        scores += tie_noise(np.arange(len(scores)))*1e-10
        scores[np.asarray(history,dtype=np.int64)] = -np.inf
        rows = np.flatnonzero(np.isfinite(scores))
        best = rows[np.argsort(-scores[rows],kind='stable')[:k]]
        return [dict(self.catalog.item(int(r)),score=float(scores[r])) for r in best]


class UFMBackend:
    name = 'UFM Rec'
    warning = 'Điểm và uncertainty là tín hiệu nghiên cứu, không phải xác suất mua hàng hay độ chắc chắn Bayesian.'
    def __init__(self,catalog,run,chunk=1024,_allow_technical_smoke=False):
        import torch
        from ufm_model import UFMConfig,UFMRec
        from train_ufm_recommender import feature_audit
        from prepare_ufm_catalog import sha256
        self.catalog,self.chunk,self.torch = catalog,chunk,torch
        run = Path(run)
        marker = json.loads((run/'completed.json').read_text())
        if marker['smoke_run'] and not _allow_technical_smoke:
            raise ValueError('Technical smoke cannot serve a production UFM demo.')
        config = json.loads((run/'config.json').read_text())
        self.features = Path(config['features'])
        if config['feature_marker_sha256'] != sha256(self.features/'complete.json'):
            raise ValueError('Feature marker changed since training.')
        feature_audit(SimpleNamespace(features=self.features,data=catalog.data),len(catalog.asins))
        if config['code_sha256']['ufm_model.py'] != sha256(ROOT/'src/ufm_model.py'):
            raise ValueError('Model code does not match training checkpoint.')
        state = torch.load(run/'best.pt',map_location='cpu',weights_only=True)
        if state['config'] != config: raise ValueError('Best checkpoint/config mismatch.')
        self.cfg = UFMConfig(**state['model_config'])
        if self.cfg.items != len(catalog.asins) or chunk<1: raise ValueError('Model/catalog mismatch.')
        self.model = UFMRec(self.cfg).eval()
        self.model.load_state_dict(state['model'])
        del state
        self.tables = tuple(torch.from_numpy(np.load(self.features/n)) for n in ['text.npy','image.npy','modalities.npy'])
        self.tables += (torch.from_numpy(np.concatenate([[0],catalog.counts]).astype(np.int64)),)
        # Deliberate CPU serving: do not compete with the long-running GPU suite.
        torch.set_num_threads(4)

    def recommend(self,asins,k=10):
        torch = self.torch
        history = self.catalog.rows(asins)
        if not 1<=k<=50: raise ValueError('Top K must be 1-50.')
        h = np.zeros((1,self.cfg.history_size),dtype=np.int64)
        past = history[-self.cfg.history_size:]
        h[0,:len(past)] = np.asarray(past,dtype=np.int64)+1
        excluded = set(history); best = []
        with torch.inference_mode():
            for start in range(0,len(self.catalog.asins),self.chunk):
                rows = np.array([r for r in range(start,min(start+self.chunk,len(self.catalog.asins))) if r not in excluded],dtype=np.int64)
                if not len(rows): continue
                output = self.model(torch.from_numpy(h),torch.from_numpy(rows[None]+1),*self.tables)
                scores = output['scores'][0].numpy().astype(np.float64)
                if not np.isfinite(scores).all(): raise FloatingPointError('Nonfinite inference scores.')
                adjusted = scores+tie_noise(rows)*1e-10
                for j in np.argsort(-adjusted,kind='stable')[:k]:
                    record = dict(self.catalog.item(int(rows[j])),score=float(scores[j]),
                        weights=output['weights'][0,j].tolist(),uncertainty=output['uncertainty'][0,j].tolist(),
                        modalities=self.tables[2][int(rows[j])+1].tolist())
                    heapq.heappush(best,(float(adjusted[j]),int(rows[j]),record))
                    if len(best)>k: heapq.heappop(best)
        return [record for _,_,record in sorted(best,key=lambda x:(-x[0],x[1]))]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=DATA)
    p.add_argument('--backend',choices=['content','ufm'],default='content')
    p.add_argument('--run',type=Path,default=ROOT/'runs/ufm_full_v1')
    p.add_argument('--history',nargs='*',default=[])
    p.add_argument('--top-k',type=int,default=10)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args(); catalog = Catalog(args.data)
    backend = ContentBackend(catalog) if args.backend=='content' else UFMBackend(catalog,args.run)
    print(json.dumps(dict(backend=backend.name,warning=backend.warning,items=backend.recommend(args.history,args.top_k)),ensure_ascii=False,indent=2))
