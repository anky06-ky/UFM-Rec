"""Full-catalog inference for research demo, never reads evaluation/test labels."""
import argparse
import csv
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit, urlencode
import re
import warnings

import numpy as np
from scipy import sparse
from evaluate_content_baseline import tie_noise
from demo_product_images import image_path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/processed/toys_games_full_temporal'
# Blend weight chosen on the earlier half of validation v3 for the served checkpoint
# (rebuilt epoch-4 checkpoint: reports/rebuild_20261108/hybrid_v3.json; published epoch 9 used 0.55).
HYBRID_ALPHA = 0.60
# Candidates taken from each source in the fast scope; UFM rescores their union.
FAST_CANDIDATES = dict(semantic=2000, behavior=2000, tfidf=1000, popularity=500)
# Fixed random sample of allowed items whose UFM scores estimate the blend statistics in the fast scope.
HYBRID_SAMPLE = 4096


def tfidf_scores(matrix,history,popularity):
    """TF-IDF profile of the latest 10 history items, popularity as fallback and tie-break."""
    available = np.diff(matrix.indptr)>0
    past = [r for r in history[-10:] if available[r]]
    if not past: return popularity.astype(np.float64)
    profile = np.asarray(matrix[past].sum(0)).ravel()
    scores = np.asarray(matrix.dot(profile)).ravel().astype(np.float64)
    return scores+popularity/(float(popularity.max()) or 1)*1e-7


def standardize(scores,reference=None):
    reference = scores if reference is None else reference
    std = reference.std()
    return (scores-reference.mean())/std if std>0 else np.zeros_like(scores)


def top_allowed(scores,mask,n):
    rows = np.flatnonzero(mask)
    if len(rows)<=n: return rows
    values = scores[rows]+tie_noise(rows)*1e-10
    return rows[np.argpartition(-values,n-1)[:n]]


def title_key(title):
    """Listings of one product often repeat the title with small variations; compare a normalized prefix."""
    return re.sub(r'[^0-9a-z]+',' ',title.lower()).strip()[:60]


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
        self.lower_titles = [t.lower() for t in self.titles]
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
        asin = self.asins[row]
        uploaded = image_path(self.data, asin).is_file()
        image = '/api/product-image?' + urlencode({'asin': asin}) if uploaded else self.images[row]
        return dict(asin=asin,title=self.titles[row],image=image,train_count=n,regime=regime)

    def provenance(self):
        simulated = (self.data/'demo_manifest.json').is_file()
        return dict(mode='synthetic_demo' if simulated else 'catalog_from_disk',
                    catalog=self.data.name, items=len(self.asins),
                    counts_simulated=simulated, display_images_affect_scores=False)

    def search(self,q,limit=20,scan=600):
        """Every word must appear in the title (or the query is an ASIN); most-interacted matches first."""
        q = q.strip().lower()[:80]
        if len(q)<2: return []
        exact = self.lookup.get(q.upper())
        words = q.split()
        found = [] if exact is None else [exact]
        for i,title in enumerate(self.lower_titles):
            if i!=exact and all(w in title for w in words):
                found.append(i)
                if len(found)>=scan: break
        found.sort(key=lambda i:(i!=exact,-int(self.counts[i])))
        return self.distinct(found,limit,[])

    def distinct(self,order,k,history):
        """First k rows of order, skipping repeated listings of one product (same normalized title)."""
        seen = {title_key(self.titles[r]) for r in history}
        picked = []
        for r in order:
            key = title_key(self.titles[int(r)])
            if key in seen: continue
            seen.add(key); picked.append(int(r))
            if len(picked)>=k: break
        return [self.item(r) for r in picked]

    def pick(self,order,k,history):
        """Positions in order of the first k distinct products (see distinct)."""
        seen = {title_key(self.titles[r]) for r in history}
        picked = []
        for position,r in enumerate(order):
            key = title_key(self.titles[int(r)])
            if key in seen: continue
            seen.add(key); picked.append(position)
            if len(picked)>=k: break
        return np.asarray(picked,dtype=np.int64)

    def regime_mask(self,regime='all'):
        n = self.counts
        masks = dict(all=np.ones(len(n),dtype=bool),zero_shot=n==0,
                     extreme_cold=(n>0)&(n<=5),cold=(n>5)&(n<=20),warm=n>20)
        if regime not in masks: raise ValueError('Unknown item regime.')
        return masks[regime]

    def examples(self,limit=12):
        rows = np.argsort(-self.counts,kind='stable')[:limit*3]
        return self.distinct(rows,limit,[])


class ContentBackend:
    name = 'TF-IDF content baseline'
    warning = 'Demo baseline TF-IDF, không phải UFM đã train. Điểm xếp hạng không phải xác suất mua hàng.'
    options = dict(models=['content'],scopes=['full'])
    def __init__(self,catalog):
        self.catalog = catalog
        if (catalog.data/'demo_manifest.json').is_file():
            self.warning += ' Catalog mẫu 30 sản phẩm; số tương tác được mô phỏng.'
        self.matrix = sparse.load_npz(catalog.data/'content/tfidf_all.npz').tocsr()
        if self.matrix.shape[0]!=len(catalog.asins) or not np.isfinite(self.matrix.data).all():
            raise ValueError('TF-IDF mapping/finite check failed.')
        self.last_stats = {}

    def recommend(self,asins,k=10,regime='all',model='content',scope='full'):
        history = self.catalog.rows(asins)
        if not 1<=k<=50: raise ValueError('Top K must be 1-50.')
        if model not in self.options['models'] or scope not in self.options['scopes']:
            raise ValueError('Unknown ranking model or scope.')
        scores = tfidf_scores(self.matrix,history,np.log1p(self.catalog.counts))
        # Full catalog, int64 ranks (do not reuse sampled uint8 rank helper).
        scores += tie_noise(np.arange(len(scores)))*1e-10
        scores[~self.catalog.regime_mask(regime)] = -np.inf
        scores[np.asarray(history,dtype=np.int64)] = -np.inf
        rows = np.flatnonzero(np.isfinite(scores))
        order = rows[np.argsort(-scores[rows],kind='stable')[:k*20]]
        best = order[self.catalog.pick(order,k,history)]
        self.last_stats = dict(model=model,scope=scope,candidates=int(len(rows)))
        return [dict(self.catalog.item(int(r)),score=float(scores[r])) for r in best]


class UFMBackend:
    name = 'UFM Rec'
    warning = ('UFM dùng lịch sử hành vi cùng CLIP embedding đã trích xuất từ mô tả và ảnh sản phẩm. '
               'JPEG trên màn hình chỉ để xem; ảnh tải lỗi không đổi embedding đã lưu. '
               'Điểm, trọng số và uncertainty là tín hiệu mô hình, không phải xác suất mua hàng.')
    def __init__(self,catalog,run,chunk=8192,_allow_technical_smoke=False,features=None,training_source=None):
        import torch
        from trained_ufm import model_classes
        from train_ufm_recommender import feature_audit
        from common import sha256
        self.catalog,self.chunk,self.torch = catalog,chunk,torch
        run = Path(run)
        marker = json.loads((run/'completed.json').read_text())
        if marker['smoke_run'] and not _allow_technical_smoke:
            raise ValueError('Technical smoke cannot serve a production UFM demo.')
        config = json.loads((run/'config.json').read_text())
        self.features = Path(features) if features is not None else Path(config['features'])
        if config['feature_marker_sha256'] != sha256(self.features/'complete.json'):
            raise ValueError('Feature marker changed since training.')
        feature_audit(SimpleNamespace(features=self.features,data=catalog.data),len(catalog.asins))
        state = torch.load(run/'best.pt',map_location='cpu',weights_only=True)
        if state['config'] != config: raise ValueError('Best checkpoint/config mismatch.')
        UFMConfig,UFMRec = model_classes(config,training_source)
        self.cfg = UFMConfig(**state['model_config'])
        if self.cfg.items != len(catalog.asins) or chunk<1: raise ValueError('Model/catalog mismatch.')
        self.model = UFMRec(self.cfg).eval()
        self.model.load_state_dict(state['model'])
        self.provenance = dict(checkpoint_sha256=sha256(run/'best.pt'),
            model_source_sha256=config['code_sha256']['ufm_model.py'],
            feature_marker_sha256=config['feature_marker_sha256'],
            variant=self.cfg.variant, checkpoint_epoch=state.get('epoch'),
            training_source=str(training_source) if training_source else str(ROOT/'src'))
        del state
        # Memory-map the CLIP tables: pages are shared with the file cache instead of copied.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='The given NumPy array is not writable')
            self.tables = tuple(torch.from_numpy(np.load(self.features/n,mmap_mode='r')) for n in ['text.npy','image.npy','modalities.npy'])
        self.tables += (torch.from_numpy(np.concatenate([[0],catalog.counts]).astype(np.int64)),)
        # Deliberate CPU serving: do not compete with the long-running GPU suite.
        torch.set_num_threads(4)
        path = catalog.data/'content/tfidf_all.npz'
        self.matrix = sparse.load_npz(path).tocsr() if path.is_file() else None
        if self.matrix is not None and self.matrix.shape[0]!=len(catalog.asins):
            raise ValueError('TF-IDF mapping check failed.')
        # The semantic item side depends only on the frozen CLIP tables and the adapter: compute it once
        # (row 0 is padding) and serve lookups, so requests skip the adapter and never page in the tables.
        with torch.inference_mode():
            ids = torch.arange(self.cfg.items+1)
            parts = [self.model.semantic_items(ids[i:i+self.chunk],*self.tables[:3]) for i in range(0,len(ids),self.chunk)]
        semantic,present = (torch.cat(x) for x in zip(*parts))
        self.model.semantic_items = lambda ids,*tables: (semantic[ids],present[ids])
        self.item_semantic = semantic[1:]  # also used to retrieve fast-scope candidates
        self.options = dict(models=['ufm','hybrid'] if self.matrix is not None else ['ufm'],scopes=['full','fast'],
                            hybrid_alpha=HYBRID_ALPHA,fast_candidates=FAST_CANDIDATES)
        self.last_stats = {}

    def retrieve(self,h,allowed,popularity,lexical):
        """Fast scope: union of the nearest items by UFM's own semantic and behaviour embeddings,
        TF-IDF and popularity; UFM then rescores only these candidates."""
        F,n = self.torch.nn.functional,FAST_CANDIDATES
        pools = [top_allowed(popularity,allowed,n['popularity'])]
        semantic,present = self.model.semantic_items(h,*self.tables[:3])
        if present.any():
            user = F.normalize(semantic.sum(1)/present.sum(1,keepdim=True).clamp_min(1),dim=-1)[0]
            pools.append(top_allowed((self.item_semantic@user).numpy(),allowed,n['semantic']))
        vectors,known = self.model.collaborative_items(h,self.tables[3])
        user,available = self.model.encode_cf_history(vectors,h,known)
        if available.any():
            behaviour = (self.model.item_ids.weight[1:]@user[0]).numpy()
            pools.append(top_allowed(behaviour,allowed&(self.catalog.counts>0),n['behavior']))
        if lexical is not None:
            pools.append(top_allowed(lexical,allowed,n['tfidf']))
        return np.unique(np.concatenate(pools))

    def score(self,h,rows):
        scores = np.empty(len(rows),dtype=np.float64)
        for start in range(0,len(rows),self.chunk):
            part = rows[start:start+self.chunk]
            scores[start:start+len(part)] = self.model(h,self.torch.from_numpy(part[None]+1),*self.tables)['scores'][0].numpy()
        if not np.isfinite(scores).all(): raise FloatingPointError('Nonfinite inference scores.')
        return scores

    def recommend(self,asins,k=10,regime='all',model='ufm',scope='full'):
        torch = self.torch
        history = self.catalog.rows(asins)
        if not 1<=k<=50: raise ValueError('Top K must be 1-50.')
        if model not in self.options['models'] or scope not in self.options['scopes']:
            raise ValueError('Unknown ranking model or scope.')
        h = np.zeros((1,self.cfg.history_size),dtype=np.int64)
        past = history[-self.cfg.history_size:]
        h[0,:len(past)] = np.asarray(past,dtype=np.int64)+1
        h = torch.from_numpy(h)
        allowed = self.catalog.regime_mask(regime).copy()
        allowed[np.asarray(history,dtype=np.int64)] = False
        popularity = np.log1p(self.catalog.counts)
        lexical = tfidf_scores(self.matrix,history,popularity) if self.matrix is not None and (
            model=='hybrid' or scope=='fast') else None
        with torch.inference_mode():
            pool = np.flatnonzero(allowed)
            rows = pool if scope=='full' else self.retrieve(h,allowed,popularity,lexical)
            scores = self.score(h,rows)
            ranking = scores
            if model=='hybrid' and len(rows):
                # Both scopes standardize against every allowed item, as the 100-candidate evaluation does
                # against mostly random items; the fast scope estimates the UFM statistics from a sample.
                reference = scores if scope=='full' else self.score(
                    h,np.random.default_rng(0).choice(pool,min(len(pool),HYBRID_SAMPLE),replace=False))
                ranking = HYBRID_ALPHA*standardize(scores,reference)+(1-HYBRID_ALPHA)*standardize(lexical[rows],lexical[pool])
            order = np.argsort(-(ranking+tie_noise(rows)*1e-10),kind='stable')[:k*20]
            best = order[self.catalog.pick(rows[order],k,history)]
            top = rows[best]
            # Branch weights and uncertainty only for the returned items.
            detail = self.model(h,torch.from_numpy(top[None]+1),*self.tables) if len(top) else None
        self.last_stats = dict(model=model,scope=scope,candidates=int(len(rows)))
        results = []
        for j,(i,row) in enumerate(zip(best,top)):
            record = dict(self.catalog.item(int(row)),score=float(ranking[i]),
                weights=detail['weights'][0,j].tolist(),uncertainty=detail['uncertainty'][0,j].tolist(),
                modalities=self.tables[2][int(row)+1].tolist())
            if model=='hybrid': record.update(ufm_score=float(scores[i]),tfidf_score=float(lexical[row]))
            results.append(record)
        return results


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=DATA)
    p.add_argument('--backend',choices=['content','ufm'],default='content')
    p.add_argument('--run',type=Path,default=ROOT/'runs/ufm_full_v1')
    p.add_argument('--features',type=Path,help='Relocated identical CLIP cache; hashes must still match.')
    p.add_argument('--training-source',type=Path,help='Archived source matching all recorded training hashes.')
    p.add_argument('--history',nargs='*',default=[])
    p.add_argument('--top-k',type=int,default=10)
    return p.parse_args()


if __name__ == '__main__':
    import sys
    if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    args = parse_args()
    if args.backend=='content' and args.data==DATA and not (args.data/'content/items.csv.gz').is_file():
        sys.path.insert(0,str(ROOT/'scripts'))
        from build_mock_demo_data import build_mock_catalog
        args.data = build_mock_catalog()
    catalog = Catalog(args.data)
    backend = ContentBackend(catalog) if args.backend=='content' else UFMBackend(
        catalog,args.run,features=args.features,training_source=args.training_source)
    print(json.dumps(dict(backend=backend.name,warning=backend.warning,items=backend.recommend(args.history,args.top_k)),ensure_ascii=False,indent=2))
