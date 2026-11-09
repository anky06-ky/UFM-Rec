"""Local instruction LLM (Qwen2.5-Instruct via transformers; weights only, no remote code)."""
import json, os, re, threading, time
from pathlib import Path

HF_HOME = Path(os.environ.get('UFM_LLM_HOME', Path(__file__).resolve().parents[1] / 'tmp/hf_llm'))
DEFAULT_MODEL = 'Qwen/Qwen2.5-3B-Instruct'
SMALL_MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
REVISIONS = {'Qwen/Qwen2.5-3B-Instruct': 'aa8e72537993ba99e69dfaafa59ed015b17504d1',
             'Qwen/Qwen2.5-1.5B-Instruct': '989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
             'Qwen/Qwen2.5-0.5B-Instruct': '7ae557604adf67be50417f59c2c2f167def9a775'}
# bf16 weights + KV cache headroom on the shared GPU.
NEEDS = {'Qwen/Qwen2.5-3B-Instruct': 7.5e9, 'Qwen/Qwen2.5-1.5B-Instruct': 4.0e9, 'Qwen/Qwen2.5-0.5B-Instruct': 1.6e9}


class LocalLLM:
    def __init__(self, model='auto', device='auto', log=print, adapter=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        free = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
        if model == 'auto':
            # The GPU is shared with other jobs: 3B when it fits, else 1.5B (3B on CPU is ~3 min/answer).
            model = DEFAULT_MODEL if free > NEEDS[DEFAULT_MODEL] else SMALL_MODEL
        if device == 'auto':
            device = 'cuda' if free > NEEDS.get(model, 7.5e9) else 'cpu'
        self.torch, self.name = torch, model
        kw = dict(revision=REVISIONS.get(model), cache_dir=HF_HOME / 'hub', trust_remote_code=False)
        self.tok = AutoTokenizer.from_pretrained(model, **kw)
        dtype = torch.bfloat16 if device == 'cuda' else torch.float32
        t0 = time.monotonic()
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, **kw).to(device).eval()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            device, dtype = 'cpu', torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, **kw).eval()
        self.device = device
        if adapter:
            from distill_train import merge_adapter
            blob = merge_adapter(self.model, adapter)
            if blob['student'] != model:
                raise ValueError('Adapter was trained for ' + blob['student'])
            self.name = f"{model} + distilled (epoch {blob['epoch']})"
        if device == 'cpu':
            torch.set_num_threads(16)
        self.lock = threading.Lock()
        log(f'LLM {model} on {device} ({dtype}) loaded in {time.monotonic() - t0:.0f}s')

    def generate(self, messages, max_new_tokens=400, temperature=0.0):
        torch = self.torch
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok([text], return_tensors='pt').to(self.device)
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.eos_token_id)
        kw.update(do_sample=True, temperature=temperature, top_p=0.9) if temperature > 0 else kw.update(do_sample=False)
        with self.lock, torch.inference_mode():
            t0 = time.monotonic()
            out = self.model.generate(**inputs, **kw)
            dt = time.monotonic() - t0
        new = out[0, inputs['input_ids'].shape[1]:]
        return self.tok.decode(new, skip_special_tokens=True).strip(), dict(
            seconds=round(dt, 2), prompt_tokens=int(inputs['input_ids'].shape[1]), new_tokens=int(len(new)))

    def generate_json(self, messages, max_new_tokens=200):
        text, stats = self.generate(messages, max_new_tokens)
        m = re.search(r'\{.*\}', text, re.S)
        try:
            return (json.loads(m.group(0)) if m else None), text, stats
        except ValueError:
            return None, text, stats


def parse_json(text):
    m = re.search(r'\{.*\}', text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except ValueError:
        return None


def generate_batch(llm, conversations, max_new_tokens=200, temperature=0.0, batch=8):
    """Left-padded batched generation (offline data building); returns list of texts."""
    torch, tok = llm.torch, llm.tok
    tok.padding_side = 'left'
    outs = []
    for s in range(0, len(conversations), batch):
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in conversations[s:s + batch]]
        enc = tok(texts, return_tensors='pt', padding=True).to(llm.device)
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
        kw.update(do_sample=True, temperature=temperature, top_p=0.9) if temperature > 0 else kw.update(do_sample=False)
        with llm.lock, torch.inference_mode():
            out = llm.model.generate(**enc, **kw)
        outs += [tok.decode(o[enc['input_ids'].shape[1]:], skip_special_tokens=True).strip() for o in out]
    return outs
