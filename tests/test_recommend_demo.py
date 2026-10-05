import contextlib
import csv
import gzip
import io
import json
from pathlib import Path
import sys
import shutil
import threading
import unittest
import urllib.request
import urllib.error

import numpy as np
from scipy import sparse

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ops'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import test_ufm_training as training_fixture
import train_ufm_recommender as trainer
from train_gpu_recommender import fingerprint
from prepare_ufm_catalog import sha256
from recommend_catalog import Catalog,ContentBackend,UFMBackend,safe_image
from demo_recommender import handler_for,ThreadingHTTPServer


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.fixture = training_fixture.TrainingTests(); self.fixture.setUp()
        self.data = self.fixture.data
        with gzip.open(self.data/'content/products_text.csv.gz','wt',encoding='utf-8',newline='') as f:
            writer = csv.writer(f); writer.writerow(['parent_asin','title','text'])
            writer.writerows([['ASIN'+str(i),'Synthetic product '+str(i),'Synthetic text'] for i in range(1,13)])
        featuremarkerpath = self.fixture.features/'complete.json'
        featuremarker = json.loads(featuremarkerpath.read_text())
        featuremarker['text_sha256'] = sha256(self.data/'content/products_text.csv.gz')
        featuremarkerpath.write_text(json.dumps(featuremarker))
        matrix = sparse.csr_matrix((np.ones(12,dtype=np.float32),(np.arange(12),np.arange(12)%3)),shape=(12,3))
        sparse.save_npz(self.data/'content/tfidf_all.npz',matrix)
        markerpath = self.fixture.legacy/'complete.json'
        marker = json.loads(markerpath.read_text()); marker['fingerprint'] = fingerprint(self.data)
        markerpath.write_text(json.dumps(marker))
        self.catalog = Catalog(self.data)

    def tearDown(self): self.fixture.tearDown()

    def test_content_seen_exclusion_empty_fallback_and_input(self):
        backend = ContentBackend(self.catalog)
        rows = backend.recommend(['ASIN1','ASIN2'],10)
        self.assertEqual(len(rows),10)
        self.assertFalse({'ASIN1','ASIN2'} & {r['asin'] for r in rows})
        self.assertEqual(backend.recommend([],1)[0]['asin'],'ASIN8')
        with self.assertRaisesRegex(ValueError,'Unknown'): backend.recommend(['not-an-asin'])
        with self.assertRaises(ValueError): backend.recommend(['ASIN1']*21)
        self.assertEqual(self.catalog.item(10)['regime'],'zero_shot')
        self.assertEqual(self.catalog.item(6)['regime'],'cold')
        unseen = backend.recommend(['ASIN11'],10,'zero_shot')
        self.assertTrue(unseen)
        self.assertTrue(all(r['train_count']==0 and r['asin']!='ASIN11' for r in unseen))
        with self.assertRaises(ValueError): backend.recommend([],10,'unknown')

    def test_image_host_guard(self):
        self.assertEqual(safe_image('https://m.media-amazon.com/images/I/test.jpg'),'https://m.media-amazon.com/images/I/test.jpg')
        for u in ['http://m.media-amazon.com/images/I/test.jpg','https://other.test/images/I/x','https://user:pw@m.media-amazon.com/images/I/x']:
            self.assertEqual(safe_image(u),'')

    def test_ufm_best_checkpoint_chunking_and_unknown_cf_mask(self):
        args = self.fixture.args('inference')
        with contextlib.redirect_stdout(io.StringIO()): trainer.train(args)
        with self.assertRaisesRegex(ValueError,'Technical smoke'): UFMBackend(self.catalog,args.output)
        a = UFMBackend(self.catalog,args.output,chunk=1,_allow_technical_smoke=True)
        b = UFMBackend(self.catalog,args.output,chunk=7,_allow_technical_smoke=True)
        one = a.recommend(['ASIN1','ASIN2'],10); seven = b.recommend(['ASIN1','ASIN2'],10)
        self.assertEqual([r['asin'] for r in one],[r['asin'] for r in seven])
        np.testing.assert_allclose([r['score'] for r in one],[r['score'] for r in seven],atol=1e-6)
        for r in one:
            if r['regime']=='zero_shot': self.assertEqual(r['weights'][0],0)
            self.assertTrue(np.isfinite(r['uncertainty']).all())
        self.assertEqual(a.recommend([],1)[0]['asin'],'ASIN8')
        relocated = self.data/'relocated_clip'
        shutil.copytree(self.fixture.features,relocated)
        moved = UFMBackend(self.catalog,args.output,features=relocated,_allow_technical_smoke=True)
        self.assertEqual(moved.features,relocated)
        self.assertEqual([r['asin'] for r in moved.recommend(['ASIN1','ASIN2'],10)],
                         [r['asin'] for r in one])
        (relocated/'complete.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'Feature marker changed'):
            UFMBackend(self.catalog,args.output,features=relocated,_allow_technical_smoke=True)

    def test_http_end_to_end_and_cross_origin_rejection(self):
        server = ThreadingHTTPServer(('127.0.0.1',0),handler_for(ContentBackend(self.catalog)))
        thread = threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        url = 'http://127.0.0.1:'+str(server.server_port)
        try:
            with urllib.request.urlopen(url+'/api/status') as r: status = json.load(r)
            self.assertIn('TF-IDF',status['backend'])
            request = urllib.request.Request(url+'/api/recommend',data=json.dumps({'history':['ASIN1'],'k':3}).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request) as r: response = json.load(r)
            self.assertEqual(len(response['results']),3)
            self.assertNotIn('ASIN1',[x['asin'] for x in response['results']])
            self.assertGreaterEqual(response['elapsed_ms'],0)
            with urllib.request.urlopen(url+'/api/examples') as r: examples = json.load(r)
            self.assertEqual(examples[0]['asin'],'ASIN8')
            for payload in [[],None,{'regime':[]},{'k':True},{'history':['not-in-catalog']},
                            {'history':['ASIN1']*21}]:
                request = urllib.request.Request(url+'/api/recommend',data=json.dumps(payload).encode())
                with self.assertRaises(urllib.error.HTTPError) as caught: urllib.request.urlopen(request)
                self.assertEqual(caught.exception.code,400)
            bad = urllib.request.Request(url+'/api/status',headers={'Origin':'https://untrusted.test'})
            with self.assertRaises(urllib.error.HTTPError) as caught: urllib.request.urlopen(bad)
            self.assertEqual(caught.exception.code,403)
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_reverse_proxy_prefix_and_exact_origin(self):
        origin = 'https://fitlab.example'
        prefix = '/proxy/8766'
        server = ThreadingHTTPServer(('127.0.0.1',0),handler_for(
            ContentBackend(self.catalog),allowed_origin=origin,base_path=prefix))
        thread = threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        url = 'http://127.0.0.1:'+str(server.server_port)
        headers = {'Host':'fitlab.example','Origin':origin}
        try:
            request = urllib.request.Request(url+prefix+'/',headers=headers)
            with urllib.request.urlopen(request) as response:
                page = response.read().decode()
            self.assertIn('<meta name="demo-base-path" content="/proxy/8766">',page)
            request = urllib.request.Request(url+prefix+'/api/status',headers={'Host':'fitlab.example'})
            with urllib.request.urlopen(request) as response: status = json.load(response)
            self.assertIn('TF-IDF',status['backend'])
            request = urllib.request.Request(url+prefix+'/api/status',headers=headers)
            with urllib.request.urlopen(request) as response: status = json.load(response)
            self.assertIn('TF-IDF',status['backend'])
            request = urllib.request.Request(url+prefix+'/api/recommend',data=json.dumps(
                {'history':['ASIN1'],'k':2}).encode(),headers={**headers,'Content-Type':'application/json'})
            with urllib.request.urlopen(request) as response: result = json.load(response)
            self.assertEqual(len(result['results']),2)
            # code-server forwards the root/API paths without /proxy/<port>.
            for path in ('/', '/api/status', '/api/search?q=ASIN'):
                with urllib.request.urlopen(urllib.request.Request(url+path,headers=headers)) as response:
                    self.assertEqual(response.status,200)
            request = urllib.request.Request(url+'/api/recommend',data=json.dumps(
                {'history':['ASIN1'],'k':2}).encode(),headers={**headers,'Content-Type':'application/json'})
            with urllib.request.urlopen(request) as response: result = json.load(response)
            self.assertEqual(len(result['results']),2)
            for path in ('/unrelated/api/status', '/api/status'):
                request = urllib.request.Request(url+path,headers={**headers,'Origin':'https://untrusted.example'})
                with self.assertRaises(urllib.error.HTTPError) as caught: urllib.request.urlopen(request)
                self.assertEqual(caught.exception.code,403)
            request = urllib.request.Request(url+prefix+'/api/recommend',data=json.dumps(
                {'history':['ASIN1'],'k':2}).encode(),headers={'Host':'fitlab.example','Content-Type':'application/json'})
            with self.assertRaises(urllib.error.HTTPError) as caught: urllib.request.urlopen(request)
            self.assertEqual(caught.exception.code,403)
            bad = urllib.request.Request(url+prefix+'/api/status',headers={
                'Host':'fitlab.example','Origin':'https://untrusted.example'})
            with self.assertRaises(urllib.error.HTTPError) as caught: urllib.request.urlopen(bad)
            self.assertEqual(caught.exception.code,403)
        finally:
            server.shutdown(); server.server_close(); thread.join()


if __name__=='__main__': unittest.main()
