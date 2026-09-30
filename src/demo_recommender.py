"""Loopback-only browser demo. CPU inference; no train or evaluation writes."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import argparse
import json
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs,urlsplit

from recommend_catalog import Catalog,ContentBackend,UFMBackend,DATA,ROOT


def handler_for(backend):
    lock = threading.Lock()
    page = (ROOT/'demo/index.html').read_bytes()
    class Handler(BaseHTTPRequestHandler):
        def send(self,status,value,content='application/json; charset=utf-8'):
            body = value if isinstance(value,bytes) else json.dumps(value,ensure_ascii=False).encode()
            self.send_response(status); self.send_header('Content-Type',content)
            self.send_header('Content-Length',str(len(body))); self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff'); self.end_headers(); self.wfile.write(body)
        def allowed(self):
            host = self.headers.get('Host','')
            origin = self.headers.get('Origin')
            return host in {f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'} and (
                not origin or origin in {f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'})
        def do_GET(self):
            if not self.allowed(): self.send(403,{'error':'Loopback origin required'}); return
            path = urlsplit(self.path)
            if path.path=='/': self.send(200,page,'text/html; charset=utf-8')
            elif path.path=='/api/status': self.send(200,dict(app='ufm-rec-demo',backend=backend.name,warning=backend.warning,
                items=len(backend.catalog.asins),max_history=20,profile_items=10 if isinstance(backend,ContentBackend) else backend.cfg.history_size))
            elif path.path=='/api/examples': self.send(200,backend.catalog.examples())
            elif path.path=='/api/search': self.send(200,backend.catalog.search(parse_qs(path.query).get('q',[''])[0]))
            else: self.send(404,{'error':'Not found'})
        def do_POST(self):
            if not self.allowed(): self.send(403,{'error':'Loopback origin required'}); return
            if self.path!='/api/recommend': self.send(404,{'error':'Not found'}); return
            try:
                size = int(self.headers.get('Content-Length','0'))
                if not 0<size<=4096: raise ValueError('Invalid payload size')
                value = json.loads(self.rfile.read(size))
                if not isinstance(value,dict): raise ValueError('Expected a JSON object')
                history,k = value.get('history',[]),value.get('k',10)
                regime = value.get('regime','all')
                if not isinstance(regime,str) or regime not in ('all','zero_shot','extreme_cold','cold','warm'):
                    raise ValueError('Invalid item regime')
                if not isinstance(history,list) or len(history)>20 or any(not isinstance(x,str) or len(x)>40 for x in history):
                    raise ValueError('Invalid history')
                if not isinstance(k,int) or isinstance(k,bool) or not 1<=k<=50: raise ValueError('Invalid Top K')
                if not lock.acquire(blocking=False): self.send(429,{'error':'Inference busy'}); return
                started = time.perf_counter()
                try: results = backend.recommend(history,k,regime)
                finally: lock.release()
                self.send(200,dict(backend=backend.name,warning=backend.warning,results=results,
                    elapsed_ms=round((time.perf_counter()-started)*1000,1),regime=regime,history_count=len(history)))
            except (ValueError,KeyError,TypeError) as e: self.send(400,{'error':str(e)})
            except Exception: self.send(500,{'error':'Inference failed; inspect local server log'})
        def log_message(self,*args): pass  # Do not persist entered histories.
    return Handler


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=DATA)
    p.add_argument('--backend',choices=['content','ufm'],default='content')
    p.add_argument('--run',type=Path,default=ROOT/'runs/ufm_full_v1')
    p.add_argument('--features',type=Path,help='Relocated identical CLIP cache; original hashes are verified.')
    p.add_argument('--port',type=int,default=8765)
    p.add_argument('--open-browser',action='store_true')
    args = p.parse_args()
    if not 1<=args.port<=65535: p.error('Port must be 1-65535')
    print('Loading catalog and model; please wait...',flush=True)
    catalog = Catalog(args.data)
    backend = ContentBackend(catalog) if args.backend=='content' else UFMBackend(catalog,args.run,features=args.features)
    server = ThreadingHTTPServer(('127.0.0.1',args.port),handler_for(backend))
    print(f'DEMO READY http://127.0.0.1:{args.port} | {backend.name} | CPU | {len(catalog.asins):,} items',flush=True)
    if args.open_browser:
        import webbrowser
        webbrowser.open(f'http://127.0.0.1:{args.port}')
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
