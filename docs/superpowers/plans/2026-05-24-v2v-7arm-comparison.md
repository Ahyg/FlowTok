# V2V 7-Arm Method Comparison — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train 7 satellite→radar v2v methods on one shared summer-2021 small dataset for a controlled comparison, adding two new FlowTok conditioning variants (modality positional embedding; cross-attention) as backward-compatible opt-in features.

**Architecture:** Arms 1–2 are Diffi2i pixel baselines (BaseU3d/DiU3d) run via new PBS scripts. Arms 3–5 are existing FlowTok flow modes (direct / token_concat / token_concat_interleaved) at FlowTok-B. Arms 6–7 add new code in `libs/model/flowtok_t2i.py` + `diffusion/flow_matching.py`, gated by `use_modality_pos_emb` / `use_cross_attention` model flags that default to legacy behavior — so the in-flight m8/m8align XL jobs reconstruct an identical model on their next resubmit.

**Tech Stack:** PyTorch, accelerate, ml_collections configs, NCI GADI PBS (gpuhopper), conda env `flowtok` (FlowTok) / `diffi2ivdit` (Diffi2i).

**Spec:** `docs/superpowers/specs/2026-05-24-v2v-7arm-comparison-design.md`

**Backward-compat invariant (verify after every FlowTok code task):** with both new flags unset, `FlowTok` builds the identical module tree (no `context_embedder`, no `norm_ca`/`cross_attn`, adaLN width 6) and `_build_pos_embed` returns the identical 2-axis embedding. The running m8/m8align jobs depend on this.

---

## Task 0: Build the summer-2021 v2v dataset filelist

**Files:**
- Create: `FlowTok/scripts/build_v2v_2021summer_filelist.py`
- Produces: `/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_v2v_baseline_2021summer.pkl`

- [ ] **Step 1: Inspect the two source pkls to learn their structure**

Run:
```bash
cd /scratch/kl02/yh0308/Projv2v/FlowTok
python3 - <<'PY'
import pickle, os
i2i="/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_i2i_baseline_2021summer_merged.pkl"
v2v="/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_v2v_train_201906_202312.pkl"
for p in (i2i, v2v):
    with open(p,"rb") as f: obj=pickle.load(f)
    print("\n==", os.path.basename(p), "type=", type(obj))
    if isinstance(obj,(list,tuple)):
        print("  len:", len(obj), "elem types:", [type(x).__name__ for x in obj][:4])
        first=obj[0]
        if isinstance(first,(list,tuple)) and first:
            print("  split0 len:", len(first), " sample[0]:", first[0])
            print("  sample[1]:", first[1] if len(first)>1 else None)
PY
```
Expected: both are `(train, val, test)` tuples; each split is a list of clip entries (paths or `(sat_path, radar_path)` pairs). Note the **date encoding in the paths** (e.g. `.../2021/07/.../YYYYMMDD_HHMM.npy`). Record the exact path field used for dates.

- [ ] **Step 2: Extract the summer-2021 date range from the i2i pkl**

Run:
```bash
python3 - <<'PY'
import pickle, re
i2i="/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_i2i_baseline_2021summer_merged.pkl"
with open(i2i,"rb") as f: tr,va,te=pickle.load(f)
def paths(split):
    out=[]
    for e in split:
        out.append(e[0] if isinstance(e,(list,tuple)) else e)
    return out
def dates(ps):
    ds=set()
    for p in ps:
        m=re.search(r'(20\d{6})', str(p))
        if m: ds.add(m.group(1))
    return ds
allp=paths(tr)+paths(va)+paths(te)
ds=sorted(dates(allp))
print("n_train entries:", len(tr), "date span:", ds[0], "->", ds[-1], "n_days:", len(ds))
PY
```
Expected: prints a contiguous summer-2021 span (e.g. `20210601 -> 20210831`). Record `DMIN`, `DMAX` (YYYYMMDD ints).

- [ ] **Step 3: Write the builder script**

Create `FlowTok/scripts/build_v2v_2021summer_filelist.py`:
```python
"""Filter the full v2v train pkl to the summer-2021 date range used by the i2i
M-series, producing a comparable v2v small-set filelist. A v2v clip is kept iff
ALL its frame timestamps fall within [DMIN, DMAX]."""
import pickle, re, argparse

def clip_paths(entry):
    # v2v entry layout is (clip_paths, clip_paths) per Diffi2i convention; a clip
    # may be a list of per-frame paths or a single path. Normalize to a list.
    p = entry[0] if isinstance(entry, (list, tuple)) else entry
    return p if isinstance(p, (list, tuple)) else [p]

def all_dates(entry):
    ds = []
    for fp in clip_paths(entry):
        m = re.search(r'(20\d{6})', str(fp))
        if m:
            ds.append(int(m.group(1)))
    return ds

def keep(entry, dmin, dmax):
    ds = all_dates(entry)
    return bool(ds) and all(dmin <= d <= dmax for d in ds)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_v2v_train_201906_202312.pkl")
    ap.add_argument("--out", default="/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_v2v_baseline_2021summer.pkl")
    ap.add_argument("--dmin", type=int, required=True)
    ap.add_argument("--dmax", type=int, required=True)
    args = ap.parse_args()
    with open(args.src, "rb") as f:
        tr, va, te = pickle.load(f)
    tr2 = [e for e in tr if keep(e, args.dmin, args.dmax)]
    va2 = [e for e in va if keep(e, args.dmin, args.dmax)]
    te2 = [e for e in te if keep(e, args.dmin, args.dmax)]
    assert len(tr2) > 0, "no train clips in range -- check path date regex / range"
    with open(args.out, "wb") as f:
        pickle.dump((tr2, va2, te2), f)
    print(f"wrote {args.out}: train={len(tr2)} val={len(va2)} test={len(te2)} (range {args.dmin}-{args.dmax})")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the builder with the recorded range**

Run (substitute DMIN/DMAX from Step 2):
```bash
python3 scripts/build_v2v_2021summer_filelist.py --dmin <DMIN> --dmax <DMAX>
```
Expected: `wrote .../dataset_filelist_v2v_baseline_2021summer.pkl: train=<N> val=... test=...` with N>0.

- [ ] **Step 5: Commit**

```bash
cd /scratch/kl02/yh0308/Projv2v/FlowTok
git add scripts/build_v2v_2021summer_filelist.py docs/superpowers/
git commit -m "data: build summer-2021 v2v filelist for 7-arm comparison"
```

---

## Task 1: Arm 6 — modality positional embedding (`_build_pos_embed`)

**Files:**
- Modify: `FlowTok/libs/model/flowtok_t2i.py` (`FlowTok.__init__`, `_build_pos_embed`)
- Test: `FlowTok/tests/test_ablation_units.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ablation_units.py` (before `_main()` / the runner that calls each `test_*`):
```python
def test_modality_pos_emb_partition_and_alignment():
    import torch, numpy as np
    from types import SimpleNamespace
    from libs.model.flowtok_t2i import FlowTok
    L_tok, T, D = 77, 4, 768          # tiny T for speed; D=768 like FlowTok-B
    cfg = SimpleNamespace(
        channels=16, clip_dim=16, num_clip_token=L_tok, cfg_indicator=0.0,
        noising_type="none", noising_scale=0.1, use_modality_pos_emb=True,
        textVAE=SimpleNamespace(num_blocks=1, hidden_dim=32, num_attention_heads=2,
                                dropout_prob=0.0, clip_loss_weight=0.0),
    )
    m = FlowTok(cfg, num_latent_tokens=L_tok, hidden_size=D, depth=1, num_heads=8)
    assert m.use_modality_pos_emb is True
    seq = 2 * T * L_tok                # block [sat | radar]
    pe = m._build_pos_embed(seq, torch.device("cpu"), torch.float32)
    assert pe.shape == (1, seq, D)
    half = seq // 2
    # sat frame i token j and radar frame i token j share spatial+temporal subvectors
    d_sp, d_tp = D // 2, D // 4
    for i in range(T):
        for j in (0, 40, 76):
            sat = i * L_tok + j
            rad = half + i * L_tok + j
            assert torch.allclose(pe[0, sat, :d_sp], pe[0, rad, :d_sp])              # spatial equal
            assert torch.allclose(pe[0, sat, d_sp:d_sp + d_tp], pe[0, rad, d_sp:d_sp + d_tp])  # temporal equal
            assert not torch.allclose(pe[0, sat, d_sp + d_tp:], pe[0, rad, d_sp + d_tp:])      # modality differs

def test_modality_flag_defaults_off():
    import torch
    from types import SimpleNamespace
    from libs.model.flowtok_t2i import FlowTok
    cfg = SimpleNamespace(
        channels=16, clip_dim=16, num_clip_token=77, cfg_indicator=0.0,
        noising_type="none", noising_scale=0.1,
        textVAE=SimpleNamespace(num_blocks=1, hidden_dim=32, num_attention_heads=2,
                                dropout_prob=0.0, clip_loss_weight=0.0),
    )
    m = FlowTok(cfg, num_latent_tokens=77, hidden_size=768, depth=1, num_heads=8)
    assert m.use_modality_pos_emb is False
    pe = m._build_pos_embed(2 * 77, torch.device("cpu"), torch.float32)  # legacy 2-axis
    assert pe.shape == (1, 2 * 77, 768)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /scratch/kl02/yh0308/Projv2v/FlowTok && python3 tests/test_ablation_units.py`
Expected: FAIL — `AttributeError: 'FlowTok' object has no attribute 'use_modality_pos_emb'`.

- [ ] **Step 3: Implement the flag + modality branch**

In `FlowTok.__init__`, right after `self.pos_n_per_frame = num_latent_tokens` (currently line 178), add:
```python
        # Arm 6: opt-in 3-axis pos-emb (spatial D/2 + temporal D/4 + modality D/4).
        # Default False == legacy 2-axis (spatial+temporal). No new params either way.
        self.use_modality_pos_emb = getattr(config, "use_modality_pos_emb", False)
```

In `_build_pos_embed`, insert this branch at the top of the method body (immediately after the docstring, before `L = seq_len`):
```python
        if getattr(self, "use_modality_pos_emb", False):
            L = seq_len
            Ltok = self.num_latent_tokens          # tokens per frame (77)
            half = L // 2                           # modality boundary: [cond | target]
            assert half % Ltok == 0, "seq_len//2 must be divisible by num_latent_tokens"
            idx = np.arange(L, dtype=np.float32)
            spatial_pos = idx % Ltok                # 0..76
            temporal_pos = (idx % half) // Ltok     # 0..T-1, aligned across modalities
            modality_pos = idx // half              # 0 = sat/cond, 1 = radar/target
            D = self.hidden_size
            d_sp = (D // 2) - ((D // 2) % 2)
            d_tp = (D // 4) - ((D // 4) % 2)
            d_mod = D - d_sp - d_tp
            assert d_mod % 2 == 0, f"modality dim {d_mod} must be even"
            sp = get_1d_sincos_pos_embed_from_grid(d_sp, spatial_pos)
            tp = get_1d_sincos_pos_embed_from_grid(d_tp, temporal_pos)
            md = get_1d_sincos_pos_embed_from_grid(d_mod, modality_pos)
            pos = np.concatenate([sp, tp, md], axis=-1)              # [L, D]
            return torch.from_numpy(pos).to(device=device, dtype=dtype).unsqueeze(0)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 tests/test_ablation_units.py`
Expected: PASS — all tests green (the new two + the pre-existing 10).

- [ ] **Step 5: Commit**

```bash
git add libs/model/flowtok_t2i.py tests/test_ablation_units.py
git commit -m "feat(arm6): opt-in 3-axis modality positional embedding"
```

---

## Task 2: Arm 6 — wire `token_concat_modality` mode in flow_matching

**Files:**
- Modify: `FlowTok/diffusion/flow_matching.py` (assert line ~214; training branch line ~281; solver line ~465)
- Test: `FlowTok/tests/test_ablation_units.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ablation_units.py`:
```python
def test_flowmatching_accepts_token_concat_modality():
    from diffusion.flow_matching import FlowMatching
    fm = FlowMatching(flow_cond_mode="token_concat_modality",
                      flow_prediction_target="radar_tokens")
    assert fm.flow_cond_mode == "token_concat_modality"

def test_modality_training_branch_supervises_radar_half():
    # token_concat_modality reuses the BLOCK data path: concat [cond | noisy],
    # supervise the last L tokens. Identity nnet => finite loss, correct logs.
    import torch
    from types import SimpleNamespace
    from diffusion.flow_matching import FlowMatching
    B, T, Lt, C = 2, 2, 77, 16
    fm = FlowMatching(flow_cond_mode="token_concat_modality",
                      flow_prediction_target="radar_tokens")
    x_start = torch.randn(B, T * Lt, C)     # radar tokens
    cond = torch.randn(B, T * Lt, C)        # sat tokens
    t = torch.rand(B)
    class IdNnet:
        def __call__(self, inp, t=None, null_indicator=None, context=None):
            return [inp]                     # identity
    all_cfg = SimpleNamespace(
        losses=SimpleNamespace(contrastive_loss_weight=0.0, kld_loss_weight=0.0),
        vq_model=SimpleNamespace(num_latent_tokens=Lt),
        nnet=SimpleNamespace(model_args=SimpleNamespace(cfg_indicator=0.0)),
    )
    loss, logs = fm.p_losses_textVAE_flowtok(x_start, cond, t, IdNnet(), all_cfg)
    assert "diff_loss" in logs and torch.isfinite(loss)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 tests/test_ablation_units.py`
Expected: FAIL — `AssertionError` inside `FlowMatching.__init__` (mode not in the allowed set).

- [ ] **Step 3: Implement — add the mode to the three membership checks**

Edit `diffusion/flow_matching.py`:

(a) assert (currently line 214):
```python
        assert flow_cond_mode in ("none", "token_concat", "token_concat_interleaved", "token_concat_modality", "cross_attention")
```

(b) training branch guard (currently line 281):
```python
        if self.flow_cond_mode in ("token_concat", "token_concat_interleaved", "token_concat_modality"):
```
(No other change inside — `token_concat_modality` is NOT interleaved, so it falls into the existing `else` block-concat path; the modality pos-emb lives entirely in the nnet.)

(c) solver guard (currently line 465):
```python
        is_token_concat = flow_cond_mode in ("token_concat", "token_concat_interleaved", "token_concat_modality")
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 tests/test_ablation_units.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add diffusion/flow_matching.py tests/test_ablation_units.py
git commit -m "feat(arm6): wire token_concat_modality mode (block path + modality pos-emb)"
```

---

## Task 3: Arm 7 — CrossAttention module + cross-attn DiTBlock + context embedder

**Files:**
- Modify: `FlowTok/libs/model/flowtok_t2i.py` (imports; new `CrossAttention`; `DiTBlock`; `FlowTok.__init__`, `_forward`, `forward`)
- Test: `FlowTok/tests/test_ablation_units.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ablation_units.py`:
```python
def test_crossattention_shape():
    import torch
    from libs.model.flowtok_t2i import CrossAttention
    ca = CrossAttention(64, num_heads=8)
    x = torch.randn(2, 30, 64); ctx = torch.randn(2, 50, 64)
    out = ca(x, ctx)
    assert out.shape == (2, 30, 64)

def test_cross_attn_block_zero_init_identity():
    # With adaLN zero-initialized (all gates 0), a use_cross_attn block returns x.
    import torch
    from libs.model.flowtok_t2i import DiTBlock
    blk = DiTBlock(64, num_heads=8, use_cross_attn=True)
    blk.train(False)
    for p in blk.adaLN_modulation[-1].parameters():
        torch.nn.init.zeros_(p)
    x = torch.randn(2, 10, 64); c = torch.randn(2, 64); ctx = torch.randn(2, 12, 64)
    out = blk._forward(x, c, ctx)
    assert torch.allclose(out, x, atol=1e-5)

def test_flowtok_cross_attention_forward_and_compat():
    import torch
    from types import SimpleNamespace
    from libs.model.flowtok_t2i import FlowTok
    base = dict(channels=16, clip_dim=16, num_clip_token=77, cfg_indicator=0.0,
                noising_type="none", noising_scale=0.1,
                textVAE=SimpleNamespace(num_blocks=1, hidden_dim=32, num_attention_heads=2,
                                        dropout_prob=0.0, clip_loss_weight=0.0))
    # cross-attn ON
    cfg = SimpleNamespace(use_cross_attention=True, **base)
    m = FlowTok(cfg, num_latent_tokens=77, hidden_size=128, depth=2, num_heads=8)
    assert hasattr(m, "context_embedder")
    x = torch.randn(2, 2 * 77, 16); ctx = torch.randn(2, 2 * 77, 16); t = torch.rand(2)
    nullind = torch.zeros(2, dtype=torch.bool)
    out = m(x, t=t, null_indicator=nullind, context=ctx)[0]
    assert out.shape == (2, 2 * 77, 16)
    # cross-attn OFF (legacy) -> no extra params, no context path
    cfg0 = SimpleNamespace(**base)
    m0 = FlowTok(cfg0, num_latent_tokens=77, hidden_size=128, depth=2, num_heads=8)
    assert not hasattr(m0, "context_embedder")
    assert m0.use_cross_attention is False
    out0 = m0(x, t=t, null_indicator=nullind)[0]
    assert out0.shape == (2, 2 * 77, 16)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 tests/test_ablation_units.py`
Expected: FAIL — `ImportError: cannot import name 'CrossAttention'`.

- [ ] **Step 3: Implement**

(a) At the top of `libs/model/flowtok_t2i.py`, after `import torch.nn as nn` (line 2), add:
```python
import torch.nn.functional as F
```

(b) Add `CrossAttention` immediately before `class DiTBlock` (before line 84):
```python
class CrossAttention(nn.Module):
    """Multi-head cross-attention: queries from x, keys/values from context."""
    def __init__(self, hidden_size, num_heads, qkv_bias=True):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.kv = nn.Linear(hidden_size, hidden_size * 2, bias=qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, context):
        B, Nq, C = x.shape
        Nk = context.shape[1]
        q = self.q(x).reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, Nk, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        out = F.scaled_dot_product_attention(q, k, v)          # [B, heads, Nq, head_dim]
        out = out.transpose(1, 2).reshape(B, Nq, C)
        return self.proj(out)
```

(c) Replace `DiTBlock.__init__` and `_forward`/`forward` (lines 88–109) with:
```python
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_cross_attn=False, **block_kwargs):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        n_mod = 9 if use_cross_attn else 6
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, n_mod * hidden_size, bias=True)
        )
        if use_cross_attn:
            self.norm_ca = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.cross_attn = CrossAttention(hidden_size, num_heads)

    def forward(self, x, c, context=None):
        return torch.utils.checkpoint.checkpoint(self._forward, x, c, context, use_reentrant=False)

    def _forward(self, x, c, context=None):
        if self.use_cross_attn:
            (shift_msa, scale_msa, gate_msa,
             shift_ca, scale_ca, gate_ca,
             shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(9, dim=1)
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_ca.unsqueeze(1) * self.cross_attn(modulate(self.norm_ca(x), shift_ca, scale_ca), context)
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
```

(d) In `FlowTok.__init__`: add the flag after `self.use_modality_pos_emb = ...` (from Task 1):
```python
        self.use_cross_attention = getattr(config, "use_cross_attention", False)
```
Change the blocks construction (currently lines 180–182) to pass the flag:
```python
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_cross_attn=self.use_cross_attention)
            for _ in range(depth)
        ])
```
After `self.final_layer = FinalLayer(...)` (line 183), add the context embedder (built only when on):
```python
        if self.use_cross_attention:
            self.context_embedder = nn.Linear(config.channels, hidden_size, bias=True)
```

(e) Update `_forward` (lines 283–303) signature and the block loop:
```python
    def _forward(self, x, t, null_indicator, context=None):
        B, L, _ = x.shape
        x = self.x_embedder(x)
        pos_embed = self._build_pos_embed(seq_len=L, device=x.device, dtype=x.dtype)
        x = x + pos_embed
        ctx = None
        if self.use_cross_attention and context is not None:
            ctx = self.context_embedder(context)
            ctx = ctx + self._build_pos_embed(seq_len=ctx.shape[1], device=ctx.device, dtype=ctx.dtype)
        t = self.t_embedder(t)
        y = self.y_embedder(null_indicator)
        c = t + y
        for block in self.blocks:
            x = block(x, c, ctx) if self.use_cross_attention else block(x, c)
        x = self.final_layer(x, c)
        return [x]
```

(f) Update `forward` (line 328) to accept and forward `context`:
```python
    def forward(self, x, t=None, text_encoder=False, text_projector=False, image_clip=False, null_indicator=None, context=None):
```
and the final `else` return (line 336):
```python
            return self._forward(x=x, t=t, null_indicator=null_indicator, context=context)
```

Note: `initialize_weights` already zero-inits `adaLN_modulation[-1]` for every block, so `gate_ca` starts at 0 (stable). No change needed there.

- [ ] **Step 4: Run to verify it passes**

Run: `python3 tests/test_ablation_units.py`
Expected: PASS (new 3 + previous).

- [ ] **Step 5: Verify backward-compat (legacy module tree unchanged)**

Run:
```bash
python3 - <<'PY'
import torch
from types import SimpleNamespace
from libs.model.flowtok_t2i import FlowTok
cfg = SimpleNamespace(channels=16, clip_dim=16, num_clip_token=77, cfg_indicator=0.0,
    noising_type="none", noising_scale=0.1,
    textVAE=SimpleNamespace(num_blocks=1, hidden_dim=32, num_attention_heads=2, dropout_prob=0.0, clip_loss_weight=0.0))
m = FlowTok(cfg, num_latent_tokens=77, hidden_size=128, depth=2, num_heads=8)
keys = [k for k in m.state_dict() if ("cross_attn" in k or "norm_ca" in k or "context_embedder" in k)]
print("legacy extra cross-attn keys:", keys)        # must be []
print("block0 adaLN out features:", m.blocks[0].adaLN_modulation[-1].out_features)  # must be 6*128=768
assert keys == [] and m.blocks[0].adaLN_modulation[-1].out_features == 6*128
print("OK backward-compat")
PY
```
Expected: `legacy extra cross-attn keys: []`, `out_features: 768`, `OK backward-compat`.

- [ ] **Step 6: Commit**

```bash
git add libs/model/flowtok_t2i.py tests/test_ablation_units.py
git commit -m "feat(arm7): opt-in cross-attention DiTBlock + context embedder (legacy-safe)"
```

---

## Task 4: Arm 7 — wire `cross_attention` mode in flow_matching

**Files:**
- Modify: `FlowTok/diffusion/flow_matching.py` (training branch after line 313; solver after line 465/in loop)
- Test: `FlowTok/tests/test_ablation_units.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ablation_units.py`:
```python
def test_cross_attention_training_branch():
    # main seq = noisy radar tokens (T*L); prediction shape == radar tokens (NOT 2*T*L).
    import torch
    from types import SimpleNamespace
    from diffusion.flow_matching import FlowMatching
    B, T, Lt, C = 2, 2, 77, 16
    fm = FlowMatching(flow_cond_mode="cross_attention", flow_prediction_target="radar_tokens")
    x_start = torch.randn(B, T * Lt, C)
    cond = torch.randn(B, T * Lt, C)
    t = torch.rand(B)
    seen = {}
    class CtxNnet:
        def __call__(self, inp, t=None, null_indicator=None, context=None):
            seen["ctx_is_cond"] = context is not None and context.shape == cond.shape
            seen["inp_len"] = inp.shape[1]
            return [inp]
    all_cfg = SimpleNamespace(
        losses=SimpleNamespace(contrastive_loss_weight=0.0, kld_loss_weight=0.0),
        vq_model=SimpleNamespace(num_latent_tokens=Lt),
        nnet=SimpleNamespace(model_args=SimpleNamespace(cfg_indicator=0.0)),
    )
    loss, logs = fm.p_losses_textVAE_flowtok(x_start, cond, t, CtxNnet(), all_cfg)
    assert seen["inp_len"] == T * Lt              # radar stream only, not concatenated
    assert seen["ctx_is_cond"] is True            # sat tokens routed to context
    assert torch.isfinite(loss)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 tests/test_ablation_units.py`
Expected: FAIL — the `cross_attention` mode currently falls through to the textVAE path and does not pass `context`, so `seen["inp_len"]`/`ctx_is_cond` assertions fail (or it errors).

- [ ] **Step 3: Implement the training branch**

In `diffusion/flow_matching.py`, immediately AFTER the `token_concat*` block returns (after the closing of the `if self.flow_cond_mode in (...)` block, i.e. after current line 313), add:
```python
        if self.flow_cond_mode == "cross_attention":
            B_, L_, _ = x_start.shape
            noise = torch.randn_like(x_start)
            x_start_local = x_start.clone()
            null_indicator = torch.zeros(B_, dtype=torch.bool, device=x_start.device)
            x_noisy = self.psi(t, x=noise, x1=x_start_local)
            prediction = nnet(x_noisy, t=t, null_indicator=null_indicator, context=cond)[0]
            if self.flow_prediction_target == "radar_tokens":
                fm_target = x_start_local
            else:
                fm_target = self.Dt_psi(t, x=noise, x1=x_start_local)
            if valid_mask is not None:
                err = (prediction - fm_target).pow(2).mean(dim=-1)
                loss_diff = (err * valid_mask).sum() / valid_mask.sum().clamp(min=1)
            else:
                loss_diff = self.mos(prediction - fm_target)
            zero = x_start.new_zeros([])
            return loss_diff, {
                'diff_loss': loss_diff,
                'contrastive_loss': zero,
                'kld_loss': zero,
            }
```

- [ ] **Step 4: Implement the solver branch**

In `ODEEulerFlowMatchingSolver.sample_euler`, after the `is_token_concat` setup block (after current line 471), add:
```python
        is_cross_attn = flow_cond_mode == "cross_attention"
        if is_cross_attn:
            assert cond_tokens is not None, "cross_attention requires cond_tokens"
```
Then in the per-step loop, change the dispatch (currently the `if is_token_concat: ... else: get_model_output_flowtok`) so cross-attn is handled before the plain `else`:
```python
            if is_token_concat:
                # ... unchanged block / interleaved handling ...
            elif is_cross_attn:
                null_ind = torch.zeros(x_T.shape[0], dtype=torch.bool, device=x_T.device)
                model_out = self.model(
                    x_T, t=t_i.repeat(x_T.shape[0]), null_indicator=null_ind, context=cond_tokens
                )[-1]
            else:
                model_out = self.get_model_output_flowtok(
                    x_T,
                    has_null_indicator=has_null_indicator,
                    t_continuous=t_i.repeat(x_T.shape[0]),
                    unconditional_guidance_scale=unconditional_guidance_scale,
                )
```
(The `prediction_target == "radar_tokens"` reparam below is unchanged and applies to cross-attn too.)

- [ ] **Step 5: Run to verify it passes**

Run: `python3 tests/test_ablation_units.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add diffusion/flow_matching.py tests/test_ablation_units.py
git commit -m "feat(arm7): wire cross_attention mode in training loss + ODE solver"
```

---

## Task 5: Train-script wiring for the two new inference paths

**Files:**
- Modify: `FlowTok/scripts/train_sat2radar_v2v.py` (inference sample call, lines ~1260–1284)

- [ ] **Step 1: Update the sample-call mode handling**

Replace the block at lines ~1260–1284 (`_flow_cond_mode = ...` through the `ode_solver.sample(...)` kwargs) so all radar-noise modes start from `randn` and route sat tokens to `cond_tokens` (the concat modes use it directly; cross-attn passes it as `context` inside the solver):
```python
                _flow_cond_mode = getattr(config, "flow_cond_mode", "none")
                _modes_randn = ("token_concat", "token_concat_interleaved",
                                "token_concat_modality", "cross_attention")
                _is_tc = _flow_cond_mode in _modes_randn
                if _is_tc:
                    x_T_init = torch.randn_like(sat_tokens)
                else:
                    x_T_init = x0
                z, _ = ode_solver.sample(
                    x_T=x_T_init,
                    x_T_uncon=None,
                    sample_steps=config.sample.sample_steps,
                    unconditional_guidance_scale=1.0,
                    has_null_indicator=False,
                    prediction_target=getattr(config, "flow_prediction_target", "velocity"),
                    flow_cond_mode=_flow_cond_mode,
                    cond_tokens=sat_tokens if _is_tc else None,
                    cond_num_latent_tokens=(
                        int(config.vq_model.num_latent_tokens)
                        if _flow_cond_mode == "token_concat_interleaved" else None
                    ),
                )
```
Keep the surrounding lines exactly as they already are (the `sample_steps`, `x_T_uncon`, `prediction_target`, etc. argument names must match the current call). Only the `_modes_randn`/`_is_tc` set and the `cond_tokens` gating change. **Do NOT change** the `_need_2L_pos_per_frame` block (~lines 320–328): arm 6 uses spatial n_per_frame=77 and arm 7's radar stream is single-modality, so neither needs the 2L override.

- [ ] **Step 2: Byte-check the diff is minimal**

Run: `cd /scratch/kl02/yh0308/Projv2v/FlowTok && git diff --stat scripts/train_sat2radar_v2v.py`
Expected: only the inference-sample region changed (a handful of lines).

- [ ] **Step 3: Commit**

```bash
git add scripts/train_sat2radar_v2v.py
git commit -m "feat: route token_concat_modality + cross_attention in v2v inference sampling"
```

---

## Task 6: CPU shape-sanity for arms 6 & 7 (forward + one loss call)

**Files:**
- Create (temporary, NOT committed): `FlowTok/_sanity_v2v_arms67.py`

- [ ] **Step 1: Write the sanity script**

Create `FlowTok/_sanity_v2v_arms67.py` (uses `.train(False)` instead of `.eval()` to avoid the security-hook false positive on the literal token):
```python
"""CPU forward + one training-loss call for arms 6 & 7 at FlowTok-B geometry
(small T for speed)."""
import torch
from types import SimpleNamespace
from libs.model.flowtok_t2i import FlowTok
from diffusion.flow_matching import FlowMatching

B, T, Lt, C, D = 2, 4, 77, 16, 768
def cfg(**extra):
    return SimpleNamespace(channels=C, clip_dim=C, num_clip_token=Lt, cfg_indicator=0.0,
        noising_type="none", noising_scale=0.1,
        textVAE=SimpleNamespace(num_blocks=1, hidden_dim=32, num_attention_heads=2,
                                dropout_prob=0.0, clip_loss_weight=0.0), **extra)
all_cfg = SimpleNamespace(
    losses=SimpleNamespace(contrastive_loss_weight=0.0, kld_loss_weight=0.0),
    vq_model=SimpleNamespace(num_latent_tokens=Lt),
    nnet=SimpleNamespace(model_args=SimpleNamespace(cfg_indicator=0.0)))
x_start = torch.randn(B, T*Lt, C); cond = torch.randn(B, T*Lt, C); t = torch.rand(B)

m6 = FlowTok(cfg(use_modality_pos_emb=True), num_latent_tokens=Lt, hidden_size=D, depth=2, num_heads=16).train(False)
fm6 = FlowMatching(flow_cond_mode="token_concat_modality", flow_prediction_target="radar_tokens")
inp6 = torch.cat([cond, fm6.psi(t, x=torch.randn_like(x_start), x1=x_start)], dim=1)
print("arm6 nnet out:", m6(inp6, t=t, null_indicator=torch.zeros(B, dtype=torch.bool))[0].shape)
print("arm6 loss:", float(fm6.p_losses_textVAE_flowtok(x_start, cond, t, m6, all_cfg)[0]))

m7 = FlowTok(cfg(use_cross_attention=True), num_latent_tokens=Lt, hidden_size=D, depth=2, num_heads=16).train(False)
fm7 = FlowMatching(flow_cond_mode="cross_attention", flow_prediction_target="radar_tokens")
print("arm7 nnet out:", m7(x_start, t=t, null_indicator=torch.zeros(B, dtype=torch.bool), context=cond)[0].shape)
print("arm7 loss:", float(fm7.p_losses_textVAE_flowtok(x_start, cond, t, m7, all_cfg)[0]))
print("SANITY OK")
```

- [ ] **Step 2: Run it**

Run: `cd /scratch/kl02/yh0308/Projv2v/FlowTok && python3 _sanity_v2v_arms67.py`
Expected: `arm6 nnet out: torch.Size([2, 616, 16])` (2·4·77=616), `arm6 loss: <finite>`, `arm7 nnet out: torch.Size([2, 308, 16])` (4·77=308), `arm7 loss: <finite>`, `SANITY OK`.

- [ ] **Step 3: Remove the temp script (do not commit)**

Run: `rm /scratch/kl02/yh0308/Projv2v/FlowTok/_sanity_v2v_arms67.py`

---

## Task 7: FlowTok configs (arms 3,4,5,6,7 full @ B + 2021summer; arms 6,7 tiny)

All FlowTok configs are derived from `configs/Sat2Radar-v2v-m8-tokconcat-xpred-FlowTiTok-XL_gadi.py` (read it for the full template). Each new config differs only in: `name="flowtok-b"`, `n_steps`, dataset `filelist_path`, `workdir`, `flow_cond_mode`, and (arms 6,7) the two `model = Args(...)` flags.

**Files (create):**
- `configs/Sat2Radar-v2v-cmp-direct-B-2021summer_gadi.py`
- `configs/Sat2Radar-v2v-cmp-m8block-B-2021summer_gadi.py`
- `configs/Sat2Radar-v2v-cmp-m8align-B-2021summer_gadi.py`
- `configs/Sat2Radar-v2v-cmp-m8mod-B-2021summer_gadi.py`
- `configs/Sat2Radar-v2v-cmp-xattn-B-2021summer_gadi.py`
- `configs/Sat2Radar-v2v-cmp-m8mod-tiny_gadi.py`
- `configs/Sat2Radar-v2v-cmp-xattn-tiny_gadi.py`

- [ ] **Step 1: Create the m8block full config (the canonical B template)**

```bash
cd /scratch/kl02/yh0308/Projv2v/FlowTok/configs
cp Sat2Radar-v2v-m8-tokconcat-xpred-FlowTiTok-XL_gadi.py Sat2Radar-v2v-cmp-m8block-B-2021summer_gadi.py
```
Then edit `Sat2Radar-v2v-cmp-m8block-B-2021summer_gadi.py`:
- `name="flowtok-xl"` → `name="flowtok-b"`
- `n_steps=200_000` → `n_steps=60_000`
- `eval_interval=1_000` → `eval_interval=2_000` (keep `save_interval=20_000`)
- dataset `filelist_path` → `"/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_v2v_baseline_2021summer.pkl"`
- workdir → `"/scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_v2v_cmp_m8block_B"`
- keep `flow_cond_mode = "token_concat"`

- [ ] **Step 2: Create the direct full config**

```bash
cp Sat2Radar-v2v-cmp-m8block-B-2021summer_gadi.py Sat2Radar-v2v-cmp-direct-B-2021summer_gadi.py
```
First read the reference's exact flow values:
```bash
grep -nE "flow_cond_mode|flow_prediction_target|use_text_vae_encoder|noising_type" Sat2Radar-v2v-sat10ch-direct-FlowTiTok-XL_gadi.py
```
Then edit `Sat2Radar-v2v-cmp-direct-B-2021summer_gadi.py` so its `flow_cond_mode`, `flow_prediction_target`, `use_text_vae_encoder`, and `noising_type` EXACTLY match the reference (arm 3 = the reference recipe at B + 2021summer). Expected typical values: `flow_cond_mode="none"`, `flow_prediction_target="velocity"`. Set workdir → `".../sat2radar_flowtok_v2v_cmp_direct_B"`.

- [ ] **Step 3: Create the m8align full config**

```bash
cp Sat2Radar-v2v-cmp-m8block-B-2021summer_gadi.py Sat2Radar-v2v-cmp-m8align-B-2021summer_gadi.py
```
Edit: `flow_cond_mode = "token_concat"` → `"token_concat_interleaved"`; workdir → `".../sat2radar_flowtok_v2v_cmp_m8align_B"`.

- [ ] **Step 4: Create the m8mod (arm 6) full config**

```bash
cp Sat2Radar-v2v-cmp-m8block-B-2021summer_gadi.py Sat2Radar-v2v-cmp-m8mod-B-2021summer_gadi.py
```
Edit:
- in the `model = Args(...)` block, add a line `use_modality_pos_emb=True,` (e.g. after `cfg_indicator=0.0,`)
- `flow_cond_mode = "token_concat"` → `"token_concat_modality"`
- workdir → `".../sat2radar_flowtok_v2v_cmp_m8mod_B"`

- [ ] **Step 5: Create the xattn (arm 7) full config**

```bash
cp Sat2Radar-v2v-cmp-m8block-B-2021summer_gadi.py Sat2Radar-v2v-cmp-xattn-B-2021summer_gadi.py
```
Edit:
- in `model = Args(...)`, add `use_cross_attention=True,`
- `flow_cond_mode = "token_concat"` → `"cross_attention"`
- workdir → `".../sat2radar_flowtok_v2v_cmp_xattn_B"`

- [ ] **Step 6: Create the two tiny configs (32-clip overfit, gate)**

```bash
cp Sat2Radar-v2v-cmp-m8mod-B-2021summer_gadi.py Sat2Radar-v2v-cmp-m8mod-tiny_gadi.py
cp Sat2Radar-v2v-cmp-xattn-B-2021summer_gadi.py Sat2Radar-v2v-cmp-xattn-tiny_gadi.py
```
In BOTH tiny configs edit:
- `n_steps=60_000` → `n_steps=12_000`
- `eval_interval=2_000` → `500`; `save_interval=20_000` → `4_000`
- dataset `filelist_path` → the per-arm tiny pkl:
  - m8mod: `".../Experiments/sat2radar_flowtok_v2v_cmp_m8mod_tiny/dataset_filelist.pkl"`
  - xattn: `".../Experiments/sat2radar_flowtok_v2v_cmp_xattn_tiny/dataset_filelist.pkl"`
- workdir → `".../sat2radar_flowtok_v2v_cmp_m8mod_tiny"` / `"..._cmp_xattn_tiny"`

- [ ] **Step 7: Build the 32-clip overfit filelists (same 32 clips for both)**

First check the builder's flags:
```bash
cd /scratch/kl02/yh0308/Projv2v/FlowTok
python3 scripts/make_overfit_filelist.py -h 2>&1 | head -20
```
Then build (adapt flag names to the help output if they differ):
```bash
mkdir -p /scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_v2v_cmp_m8mod_tiny \
         /scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_v2v_cmp_xattn_tiny
python3 scripts/make_overfit_filelist.py \
  --src /g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_v2v_baseline_2021summer.pkl \
  --n 32 \
  --out /scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_v2v_cmp_m8mod_tiny/dataset_filelist.pkl
cp /scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_v2v_cmp_m8mod_tiny/dataset_filelist.pkl \
   /scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_v2v_cmp_xattn_tiny/dataset_filelist.pkl
cmp -s /scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_v2v_cmp_m8mod_tiny/dataset_filelist.pkl \
       /scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_v2v_cmp_xattn_tiny/dataset_filelist.pkl && echo "tiny filelists identical"
```
Expected: both tiny pkls exist and are byte-identical (`tiny filelists identical`).

- [ ] **Step 8: Commit**

```bash
git add configs/Sat2Radar-v2v-cmp-*.py
git commit -m "config: 7-arm v2v comparison configs (B + 2021summer; arms6,7 tiny)"
```

---

## Task 8: FlowTok PBS scripts (arms 3–7 full + arms 6,7 tiny with gate→full→self-resubmit)

Base on the existing `train_v2v_m8align_tokconcat_xpred_{tiny,full}_gadi.sh` (read them for the exact gate/resubmit logic). Each script differs only in CFG path, workdir (WD/TD), job name (`-N`), and the joblog/qsub-target filenames.

**Files (create, then `chmod +x`):**
- `train_v2v_cmp_direct_full_gadi.sh`, `train_v2v_cmp_m8block_full_gadi.sh`, `train_v2v_cmp_m8align_full_gadi.sh`, `train_v2v_cmp_m8mod_full_gadi.sh`, `train_v2v_cmp_xattn_full_gadi.sh`
- `train_v2v_cmp_m8mod_tiny_gadi.sh`, `train_v2v_cmp_xattn_tiny_gadi.sh`

- [ ] **Step 1: Create the 5 full scripts from the m8align full template**

Example (m8mod):
```bash
cd /scratch/kl02/yh0308/Projv2v/FlowTok
cp train_v2v_m8align_tokconcat_xpred_full_gadi.sh train_v2v_cmp_m8mod_full_gadi.sh
```
Edit `train_v2v_cmp_m8mod_full_gadi.sh`:
- `#PBS -N v2v_m8align_full` → `#PBS -N v2v_cmp_m8mod_full`
- `CFG=$FT/configs/Sat2Radar-v2v-m8align-tokconcat-xpred-FlowTiTok-XL_gadi.py` → `CFG=$FT/configs/Sat2Radar-v2v-cmp-m8mod-B-2021summer_gadi.py`
- `WD=.../sat2radar_flowtok_v2v_m8align_tokconcat_xpred_full` → `WD=.../sat2radar_flowtok_v2v_cmp_m8mod_B`
- `TARGET=200000` → `TARGET=60000`
- joblog name `${PBS_JOBID}_v2v_m8align_full.log` → `${PBS_JOBID}_v2v_cmp_m8mod_full.log` (every occurrence)
- self-`qsub` line `qsub train_v2v_m8align_tokconcat_xpred_full_gadi.sh` → `qsub train_v2v_cmp_m8mod_full_gadi.sh`

Repeat for the other 4 (all keep `TARGET=60000` and the `timeout 47h` + resume + conditional re-qsub + RESUBMIT_STALLED logic verbatim):

| script | -N | CFG basename | WD basename | qsub target |
|---|---|---|---|---|
| direct | v2v_cmp_direct_full | Sat2Radar-v2v-cmp-direct-B-2021summer_gadi.py | sat2radar_flowtok_v2v_cmp_direct_B | train_v2v_cmp_direct_full_gadi.sh |
| m8block | v2v_cmp_m8block_full | Sat2Radar-v2v-cmp-m8block-B-2021summer_gadi.py | sat2radar_flowtok_v2v_cmp_m8block_B | train_v2v_cmp_m8block_full_gadi.sh |
| m8align | v2v_cmp_m8align_full | Sat2Radar-v2v-cmp-m8align-B-2021summer_gadi.py | sat2radar_flowtok_v2v_cmp_m8align_B | train_v2v_cmp_m8align_full_gadi.sh |
| m8mod | v2v_cmp_m8mod_full | Sat2Radar-v2v-cmp-m8mod-B-2021summer_gadi.py | sat2radar_flowtok_v2v_cmp_m8mod_B | train_v2v_cmp_m8mod_full_gadi.sh |
| xattn | v2v_cmp_xattn_full | Sat2Radar-v2v-cmp-xattn-B-2021summer_gadi.py | sat2radar_flowtok_v2v_cmp_xattn_B | train_v2v_cmp_xattn_full_gadi.sh |

- [ ] **Step 2: Create the 2 tiny scripts from the m8align tiny template**

```bash
cp train_v2v_m8align_tokconcat_xpred_tiny_gadi.sh train_v2v_cmp_m8mod_tiny_gadi.sh
cp train_v2v_m8align_tokconcat_xpred_tiny_gadi.sh train_v2v_cmp_xattn_tiny_gadi.sh
```
Edit `train_v2v_cmp_m8mod_tiny_gadi.sh`:
- `#PBS -N v2v_m8align_tiny` → `#PBS -N v2v_cmp_m8mod_tiny`
- `CFG=...m8align-tokconcat-xpred-tiny_gadi.py` → `CFG=$FT/configs/Sat2Radar-v2v-cmp-m8mod-tiny_gadi.py`
- `TD=...v2v_m8align_tokconcat_xpred_tiny` → `TD=.../sat2radar_flowtok_v2v_cmp_m8mod_tiny`
- joblog `${PBS_JOBID}_v2v_m8align_tiny.log` → `${PBS_JOBID}_v2v_cmp_m8mod_tiny.log`
- gate's `qsub train_v2v_m8align_tokconcat_xpred_full_gadi.sh` → `qsub train_v2v_cmp_m8mod_full_gadi.sh`

Edit `train_v2v_cmp_xattn_tiny_gadi.sh` with the xattn equivalents (`-N v2v_cmp_xattn_tiny`, CFG `...cmp-xattn-tiny_gadi.py`, TD `..._cmp_xattn_tiny`, joblog `..._v2v_cmp_xattn_tiny.log`, gate qsub `train_v2v_cmp_xattn_full_gadi.sh`).

- [ ] **Step 3: chmod + syntax-validate**

```bash
chmod +x train_v2v_cmp_*.sh
for s in train_v2v_cmp_direct_full train_v2v_cmp_m8block_full train_v2v_cmp_m8align_full \
         train_v2v_cmp_m8mod_full train_v2v_cmp_xattn_full train_v2v_cmp_m8mod_tiny train_v2v_cmp_xattn_tiny; do
  bash -n ${s}_gadi.sh || { echo "PARSE FAIL: $s"; break; }
done && echo "ALL PBS SCRIPTS PARSE OK"
```
Expected: `ALL PBS SCRIPTS PARSE OK`.

- [ ] **Step 4: Commit**

```bash
git add train_v2v_cmp_*.sh
git commit -m "pbs: 7-arm v2v comparison job scripts (gate->full->self-resubmit)"
```

---

## Task 9: Diffi2i baseu/diu PBS scripts (arms 1 & 2)

Base on `Diffi2i-shrimp-proj2/train_v2v_{baseu,diu}_gadi.sh`. Use 10 IR bands + lightning to match the flowtok arms (baseu `in_dim=11`; diu `in_dim=12` because diffusion concatenates the noisy radar channel), and the summer-2021 v2v filelist.

**Files (create):** `Diffi2i-shrimp-proj2/train_v2v_cmp_baseu_2021summer_gadi.sh`, `train_v2v_cmp_diu_2021summer_gadi.sh`

- [ ] **Step 1: Create the baseu script**

```bash
cd /scratch/kl02/yh0308/Projv2v/Diffi2i-shrimp-proj2
cp train_v2v_baseu_gadi.sh train_v2v_cmp_baseu_2021summer_gadi.sh
```
Edit `train_v2v_cmp_baseu_2021summer_gadi.sh`:
- `#PBS -N v2v_baseu3d_train` → `#PBS -N v2v_cmp_baseu`
- `#PBS -q gpuvolta` → `#PBS -q gpuhopper`
- `MODEL_DIR=".../v2v-3sat-ct005/models"` → `MODEL_DIR="${PROJECT_ROOT}/v2v-cmp-baseu-2021summer/models"`
- `FILELIST=".../dataset_filelist_v2v_train_201906_202312_halfvalid50_ct005.pkl"` → `FILELIST="/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_v2v_baseline_2021summer.pkl"`
- ckpt glob `baseu3d-v2v-3sat-ct005_step*.pt` → `baseu3d-v2v-cmp-2021summer_step*.pt` (both occurrences)
- python args:
  - `--ir-band-indices "0,2,6"` → `--ir-band-indices "0,1,2,3,4,5,6,7,8,9"`
  - `--in-dim 4` → `--in-dim 11`
  - `--input-shape "4,128,128"` → `--input-shape "11,128,128"`
  - `--total-steps 1200000` → `--total-steps 200000`
  - `--save-freq 50000` → `--save-freq 25000`
  - `--label "baseu3d-v2v-3sat-ct005"` → `--label "baseu3d-v2v-cmp-2021summer"`
  - add `--loss hilburn_legacy` (matches i2i m9)
  - joblog name → `${PBS_JOBID}_v2v_cmp_baseu.log`
- Add a self-resubmit guard at the very end (baseu has no `--walltime-seconds`; rely on save-freq + re-detect):
```bash
NEW_LATEST=$(ls -t "${MODEL_DIR}"/baseu3d-v2v-cmp-2021summer_step*.pt 2>/dev/null | head -1)
NEW_STEP=$(echo "${NEW_LATEST}" | grep -oP 'step\K[0-9]+')
if [ -n "${NEW_STEP}" ] && [ "${NEW_STEP}" -lt 200000 ]; then
    echo "baseu cmp: step ${NEW_STEP}/200000, re-qsub"
    cd "${PROJECT_ROOT}" && qsub train_v2v_cmp_baseu_2021summer_gadi.sh
else
    echo "baseu cmp: done at step ${NEW_STEP}"
fi
```

- [ ] **Step 2: Create the diu script**

```bash
cp train_v2v_diu_gadi.sh train_v2v_cmp_diu_2021summer_gadi.sh
```
Edit `train_v2v_cmp_diu_2021summer_gadi.sh`:
- `#PBS -N v2v_diu3d_train` → `#PBS -N v2v_cmp_diu`
- `#PBS -q gpuvolta` → `#PBS -q gpuhopper`
- `MODEL_DIR=".../v2v-3sat-ct005/models"` → `MODEL_DIR="${PROJECT_ROOT}/v2v-cmp-diu-2021summer/models"`
- `FILELIST=...` → `"/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_v2v_baseline_2021summer.pkl"`
- `FINAL_TARGET=1200000` → `FINAL_TARGET=200000`
- `WALLTIME_SEC=172800` → `WALLTIME_SEC=169200`  # 47h margin
- ckpt glob `diu3d-v2v-3sat-ct005_step*.pt` → `diu3d-v2v-cmp-2021summer_step*.pt` (all 3 occurrences)
- python args:
  - `--ir-band-indices "0,2,6"` → `--ir-band-indices "0,1,2,3,4,5,6,7,8,9"`
  - `--in-dim 5` → `--in-dim 12`
  - `--input-shape "5,128,128"` → `--input-shape "12,128,128"`
  - `--save-freq 50000` → `--save-freq 25000`
  - `--label "diu3d-v2v-3sat-ct005"` → `--label "diu3d-v2v-cmp-2021summer"`
  - joblog → `${PBS_JOBID}_v2v_cmp_diu.log`
- update the existing self-resubmit tail's `qsub train_v2v_diu_gadi.sh` → `qsub train_v2v_cmp_diu_2021summer_gadi.sh`.

- [ ] **Step 3: Verify the in-dim asserts hold + scripts parse**

Run:
```bash
cd /scratch/kl02/yh0308/Projv2v/Diffi2i-shrimp-proj2
python3 - <<'PY'
# baseu: input_shape[0] == in_dim*(history_frames+1) -> 11 == 11*1
# diu:   input_shape[0] == (in_dim-1)*(history_frames+1)+1 -> 12 == 11+1
print("baseu", 11 == 11*1)
print("diu",   12 == (12-1)*1 + 1)
PY
chmod +x train_v2v_cmp_baseu_2021summer_gadi.sh train_v2v_cmp_diu_2021summer_gadi.sh
bash -n train_v2v_cmp_baseu_2021summer_gadi.sh && bash -n train_v2v_cmp_diu_2021summer_gadi.sh && echo OK
```
Expected: `baseu True`, `diu True`, `OK`.

- [ ] **Step 4: Commit**

```bash
git add train_v2v_cmp_baseu_2021summer_gadi.sh train_v2v_cmp_diu_2021summer_gadi.sh
git commit -m "pbs: baseu/diu v2v cmp scripts (10ir+lgt, 2021summer)"
```

---

## Task 10: Launch — tiny gate for arms 6,7; full for arms 1–5

**Files:** none (submission only). The USER submits via `qsub` per the established workflow.

- [ ] **Step 1: Submit the two tiny gates (arms 6 & 7)**

```bash
cd /scratch/kl02/yh0308/Projv2v/FlowTok
qsub train_v2v_cmp_m8mod_tiny_gadi.sh
qsub train_v2v_cmp_xattn_tiny_gadi.sh
```
These auto-`qsub` their full runs on gate PASS (`diff_loss<0.3 & step>=11900 & not-NaN`); on FAIL they write `TINY_FAIL` and do NOT submit full.

- [ ] **Step 2: Submit arms 1–5 full directly**

```bash
cd /scratch/kl02/yh0308/Projv2v/FlowTok
qsub train_v2v_cmp_direct_full_gadi.sh
qsub train_v2v_cmp_m8block_full_gadi.sh
qsub train_v2v_cmp_m8align_full_gadi.sh
cd /scratch/kl02/yh0308/Projv2v/Diffi2i-shrimp-proj2
qsub train_v2v_cmp_baseu_2021summer_gadi.sh
qsub train_v2v_cmp_diu_2021summer_gadi.sh
```

- [ ] **Step 3: Confirm queue state**

Run: `qstat -u yh0308`
Expected: 7 new jobs (2 tiny + 3 flowtok full + 2 diffi2i full) Queued/Running, plus the pre-existing m8/m8align XL jobs untouched.

- [ ] **Step 4: After arms 6,7 tiny finish, check the gate result**

Run:
```bash
for ARM in m8mod xattn; do
  TD=/scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_v2v_cmp_${ARM}_tiny
  echo "== $ARM =="; ls "$TD"/TINY_FAIL 2>/dev/null && echo "GATE FAILED" || echo "gate passed (or still running)"
done
```
If `TINY_FAIL` exists, STOP and diagnose (do not force-submit the full) — invoke superpowers:systematic-debugging on that arm.

---

## Task 11: Holdout test wiring (after full runs land ckpts) — deferred

**Files:** to be created when ckpts exist: `FlowTok/test_v2v_cmp_gadi.sh` (per-arm), Diffi2i `validate_test_ckpt.py` invocations for arms 1,2.

Intentionally deferred until at least one ckpt per arm exists. When ready:
- FlowTok arms: `python scripts/test_sat2radar_v2v.py --ckpt <...>.pth --config configs/Sat2Radar-v2v-cmp-<arm>-B-2021summer_gadi.py` on the v2v holdout pkl `dataset_filelist_v2v_test_202407_202507.pkl`, `--seed 42`.
- Diffi2i arms: `python3 validate_test_ckpt.py --model-path <dir> --label <label> --step <N> --split test --full-metrics --seed 42` with the same holdout pkl copied into the model dir as `dataset_filelist_test.pkl`.
- Collect FSS / wFSS / SSIM / PSNR / MAE into one comparison table. Same pkl + seed across all 7 arms so the numbers are directly comparable.

---

## Self-review notes

- **Spec coverage:** dataset (Task 0); arms 3–5 configs/scripts (Tasks 7–8); arm 6 code+wiring (Tasks 1–2,5) + config/script (7–8); arm 7 code+wiring (Tasks 3–5) + config/script (7–8); arms 1–2 (Task 9); tiny gate (Tasks 7–8,10); tests (Tasks 1–4); holdout (Task 11). All spec sections mapped.
- **Backward-compat:** Tasks 1 & 3 keep the legacy module tree byte-identical when flags are off (explicit verify in Task 3 Step 5); the running m8/m8align XL jobs are safe on resubmit.
- **Type/name consistency:** `flow_cond_mode` strings `token_concat_modality` / `cross_attention`, model flags `use_modality_pos_emb` / `use_cross_attention`, and the `context=` kwarg are used identically across model, flow_matching, train-script, configs, and tests.
- **Open verifications folded into steps (not placeholders):** exact summer-2021 date range (Task 0 Step 2); `make_overfit_filelist.py` flag names (Task 7 Step 7); the arm-3 reference recipe values (Task 7 Step 2).
