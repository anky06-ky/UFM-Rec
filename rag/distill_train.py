"""Distil the teacher's filtered outputs into a small student with LoRA (no extra dependencies).

Student: Qwen2.5-0.5B-Instruct (default). LoRA rank r on every attention and MLP projection;
loss only on the assistant tokens (the JSON). One adapter learns both steps (intent plan, answer picks).
The adapter is saved separately and merged into the base weights when served (rag/llm.py --adapter).
"""
import argparse, json, math, random, sys, time
from pathlib import Path
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'rag'))
from llm import HF_HOME, REVISIONS  # noqa: E402

DATA = ROOT / 'data/processed/toys_games_full_temporal/rag_distill_v1'
TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


class LoRA(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.base(x)
        return y + (self.drop(x).float() @ self.A.T @ self.B.T * self.scale).to(y.dtype)


def add_lora(model, r, alpha):
    names = []
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and child_name in TARGETS:
                setattr(mod, child_name, LoRA(child, r, alpha))
                names.append(f'{name}.{child_name}')
    return names


def lora_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if n.endswith('.A') or n.endswith('.B')}


def merge_adapter(model, path):
    """Merge a saved adapter into a plain (non-LoRA) model in place."""
    blob = torch.load(path, map_location='cpu', weights_only=True)
    scale = blob['alpha'] / blob['r']
    mods = dict(model.named_modules())
    for key in [k for k in blob['state'] if k.endswith('.A')]:
        lin_name = key[:-2].replace('.base', '')
        lin = mods[lin_name]
        A, B = blob['state'][key].float(), blob['state'][key[:-2] + '.B'].float()
        with torch.no_grad():
            lin.weight += (B @ A * scale).to(lin.weight.dtype).to(lin.weight.device)
    return blob


def encode(tok, ex, max_len):
    prompt = tok.apply_chat_template(ex['messages'], tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)['input_ids']
    t_ids = tok(ex['target'] + tok.eos_token, add_special_tokens=False)['input_ids']  # <|im_end|> for Qwen
    ids = (p_ids + t_ids)[-max_len:]
    labels = ([-100] * len(p_ids) + t_ids)[-max_len:]
    return ids, labels


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--student', default='Qwen/Qwen2.5-0.5B-Instruct')
    ap.add_argument('--out', type=Path, default=ROOT / 'runs/rag_student_0.5b')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--r', type=int, default=16)
    ap.add_argument('--alpha', type=int, default=32)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--accum', type=int, default=4)
    ap.add_argument('--max-len', type=int, default=1536)
    ap.add_argument('--holdout', type=float, default=0.05)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0); random.seed(0)
    kw = dict(revision=REVISIONS[args.student], cache_dir=HF_HOME / 'hub', trust_remote_code=False)
    tok = AutoTokenizer.from_pretrained(args.student, **kw)
    model = AutoModelForCausalLM.from_pretrained(args.student, dtype=torch.bfloat16, **kw).cuda()
    model.requires_grad_(False)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    names = add_lora(model, args.r, args.alpha)
    model.cuda()
    params = [p for n, p in model.named_parameters() if n.endswith('.A') or n.endswith('.B')]
    for p in params:
        p.requires_grad_(True)
    examples = [dict(json.loads(l), task='intent') for l in (DATA / 'intent_sft.jsonl').open()] + \
               [dict(json.loads(l), task='answer') for l in (DATA / 'answer_sft.jsonl').open()]
    random.shuffle(examples)
    n_hold = max(20, int(len(examples) * args.holdout))
    hold, train = examples[:n_hold], examples[n_hold:]
    enc_train = [encode(tok, e, args.max_len) for e in train]
    enc_hold = [encode(tok, e, args.max_len) for e in hold]
    steps = math.ceil(len(enc_train) / (args.batch * args.accum)) * args.epochs
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 30) * max(0.05, 0.5 * (1 + math.cos(math.pi * s / steps))))
    print(f'lora modules {len(names)}, trainable {sum(p.numel() for p in params) / 1e6:.1f}M, train {len(train)}, holdout {len(hold)}, steps {steps}', flush=True)

    def batches(data, bs, shuffle):
        idx = list(range(len(data)))
        if shuffle:
            random.shuffle(idx)
        idx.sort(key=lambda i: len(data[i][0]) // 64 + (random.random() if shuffle else 0))  # bucket by length
        groups = [idx[i:i + bs] for i in range(0, len(idx), bs)]
        if shuffle:
            random.shuffle(groups)
        for g in groups:
            L = max(len(data[i][0]) for i in g)
            ids = torch.full((len(g), L), tok.pad_token_id or 0)
            lab = torch.full((len(g), L), -100)
            att = torch.zeros((len(g), L), dtype=torch.long)
            for k, i in enumerate(g):
                a, b = data[i]
                ids[k, :len(a)] = torch.tensor(a); lab[k, :len(b)] = torch.tensor(b); att[k, :len(a)] = 1
            yield ids.cuda(), lab.cuda(), att.cuda()

    def evaluate():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for ids, lab, att in batches(enc_hold, args.batch, False):
                out = model(input_ids=ids, attention_mask=att, labels=lab)
                k = int((lab[:, 1:] != -100).sum())
                tot += float(out.loss) * k; n += k
        model.train()
        return tot / max(n, 1)

    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / 'history.jsonl').open('a')
    t0, step, best = time.monotonic(), 0, float('inf')
    base_loss = evaluate()
    print('holdout loss before', round(base_loss, 4), flush=True)
    model.train()
    for ep in range(args.epochs):
        run, cnt = 0.0, 0
        for bi, (ids, lab, att) in enumerate(batches(enc_train, args.batch, True)):
            out = model(input_ids=ids, attention_mask=att, labels=lab)
            (out.loss / args.accum).backward()
            run += float(out.loss); cnt += 1
            if (bi + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 20 == 0:
                    print(f'ep {ep + 1} step {step}/{steps} loss {run / cnt:.4f} lr {sched.get_last_lr()[0]:.2e} '
                          f'{(time.monotonic() - t0) / 60:.1f}min vram {torch.cuda.max_memory_allocated() / 2**30:.2f}GiB', flush=True)
                    run, cnt = 0.0, 0
        hl = evaluate()
        rec = dict(epoch=ep + 1, step=step, holdout_loss=round(hl, 4), minutes=round((time.monotonic() - t0) / 60, 1))
        log.write(json.dumps(rec) + '\n'); log.flush(); print(rec, flush=True)
        if hl < best:
            best = hl
            torch.save(dict(state=lora_state(model), r=args.r, alpha=args.alpha, student=args.student,
                            revision=REVISIONS[args.student], epoch=ep + 1, holdout_loss=hl), args.out / 'adapter.pt')
    (args.out / 'completed.json').write_text(json.dumps(dict(best_holdout_loss=round(best, 4), holdout_loss_before=round(base_loss, 4),
                                                             train=len(train), holdout=len(hold), steps=step, args={k: str(v) for k, v in vars(args).items()}), indent=2))


if __name__ == '__main__':
    main()
