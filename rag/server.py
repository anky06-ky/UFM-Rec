"""Chat demo server for the RAG layer (local LLM + catalog retrieval + UFM v2)."""
import argparse, json, sys, threading, time, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'rag'))
sys.path.insert(0, str(ROOT / 'src'))


def handler_for(get_pipeline, allowed_origin='', base_path=''):
    lock = threading.Lock()
    allowed_origin = allowed_origin.rstrip('/')
    origin_host = urlsplit(allowed_origin).netloc
    page = (ROOT / 'rag/chat.html').read_bytes().replace(
        b'<meta name="demo-base-path" content="">', f'<meta name="demo-base-path" content="{base_path}">'.encode())

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, value, content='application/json; charset=utf-8'):
            body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def allowed(self):
            host, origin = self.headers.get('Host', ''), self.headers.get('Origin')
            port = self.server.server_port
            if host in {f'127.0.0.1:{port}', f'localhost:{port}'}:
                return not origin or origin in {f'http://127.0.0.1:{port}', f'http://localhost:{port}', allowed_origin}
            return bool(allowed_origin and host == origin_host and
                        (origin == allowed_origin or (origin is None and self.command == 'GET')))

        def route(self):
            path = urlsplit(self.path).path
            if base_path and path == base_path:
                return '/'
            if base_path and path.startswith(base_path + '/'):
                return path[len(base_path):]
            return path

        def do_GET(self):
            if not self.allowed():
                return self.send(403, {'error': 'Origin not allowed'})
            path = self.route()
            p = get_pipeline()
            if path == '/':
                self.send(200, page, 'text/html; charset=utf-8')
            elif path == '/api/status':
                if p is None:
                    return self.send(200, {'ready': False})
                self.send(200, dict(ready=True, llm=p.llm.name, llm_device=p.llm.device, answer_llm=getattr(getattr(p, 'answer_llm', None), 'name', None), pipeline=type(p).__name__,
                                    items=len(p.retriever.catalog.asins),
                                    ufm=f'UFM v2 epoch {p.retriever.ufm.epoch}' if p.retriever.ufm else None))
            elif path == '/api/search':
                if p is None:
                    return self.send(503, {'error': 'Đang khởi động'})
                q = parse_qs(urlsplit(self.path).query).get('q', [''])[0]
                self.send(200, p.retriever.catalog.search(q, limit=8))
            else:
                self.send(404, {'error': 'Not found'})

        def do_POST(self):
            if not self.allowed():
                return self.send(403, {'error': 'Origin not allowed'})
            route = self.route()
            if route not in ('/api/chat', '/api/chat_stream'):
                return self.send(404, {'error': 'Not found'})
            p = get_pipeline()
            if p is None:
                return self.send(503, {'error': 'Mô hình đang khởi động, thử lại sau ít phút.'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 16384:
                    raise ValueError('Invalid payload size')
                v = json.loads(self.rfile.read(size))
                msg, state, history = v.get('message', ''), v.get('state') or {}, v.get('history') or []
                if not isinstance(msg, str) or not msg.strip() or len(msg) > 500:
                    raise ValueError('Tin nhắn trống hoặc quá dài (tối đa 500 ký tự).')
                if not isinstance(state, dict) or not isinstance(history, list) or len(history) > 20 \
                        or any(not isinstance(x, str) or len(x) > 40 for x in history):
                    raise ValueError('Invalid state/history')
                if not lock.acquire(blocking=False):
                    return self.send(429, {'error': 'Đang trả lời câu khác, vui lòng đợi.'})
                try:
                    if route == '/api/chat_stream' and hasattr(p, 'run'):
                        return self.stream(p.run(msg, state, history))
                    out = p.chat(msg, state, history)
                finally:
                    lock.release()
                self.send(200, out)
            except (ValueError, KeyError, TypeError) as e:
                self.send(400, {'error': str(e)})
            except Exception:
                traceback.print_exc()
                self.send(500, {'error': 'Lỗi khi sinh câu trả lời; xem log máy chủ.'})

        def stream(self, events):
            # NDJSON, one event per line, flushed as soon as each stage finishes.
            self.send_response(200)
            self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Accel-Buffering', 'no')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            try:
                for ev in events:
                    self.wfile.write((json.dumps(ev, ensure_ascii=False) + '\n').encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception:
                traceback.print_exc()
                self.wfile.write((json.dumps(dict(type='error', error='Lỗi khi sinh câu trả lời.'), ensure_ascii=False) + '\n').encode())
            self.close_connection = True

        def log_message(self, *a):
            pass  # do not persist user messages

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port', type=int, default=8767)
    ap.add_argument('--allowed-origin', default='')
    ap.add_argument('--base-path', default='')
    ap.add_argument('--ufm-run', default=str(ROOT / 'runs/ufm_v2'))
    ap.add_argument('--llm-device', default='auto')
    ap.add_argument('--llm-model', default='auto', help='auto picks 3B or 1.5B by free GPU memory')
    ap.add_argument('--llm-adapter', default=None, help='LoRA adapter.pt from rag/distill_train.py (merged at load)')
    ap.add_argument('--answer-model', default=None, help='optional separate LLM for the answer step (v2)')
    ap.add_argument('--answer-adapter', default=None)
    ap.add_argument('--pipeline', choices=['v1', 'v2'], default='v2')
    args = ap.parse_args()
    holder = {}

    def load():
        from retriever import QueryRetriever
        from llm import LocalLLM
        from pipeline import RAGPipeline
        llm = LocalLLM(model=args.llm_model, device=args.llm_device, adapter=args.llm_adapter)
        retriever = QueryRetriever(ufm_run=args.ufm_run or None)
        retriever.encode('warm up')
        if args.pipeline == 'v2':
            from pipeline_v2 import AdvancedRAG
            ans = LocalLLM(model=args.answer_model, device=args.llm_device, adapter=args.answer_adapter) if args.answer_model else None
            holder['p'] = AdvancedRAG(retriever, llm, answer_llm=ans)
        else:
            holder['p'] = RAGPipeline(retriever, llm)
        print('RAG ready', flush=True)

    threading.Thread(target=load, daemon=True).start()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(lambda: holder.get('p'), args.allowed_origin, args.base_path))
    print(f'Serving on http://127.0.0.1:{args.port}{args.base_path}/', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
