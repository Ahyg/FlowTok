# Arm 8 — Satellite Context Self-Attention Encoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, timestep-independent self-attention encoder that refines satellite K/V tokens before arm-7's per-block cross-attention, gated by a new flag that defaults to exact arm-7 behavior.

**Architecture:** A new `SatContextEncoder` (N pre-norm self-attention layers @ hidden_size, reusing timm `Attention`/`Mlp`) is inserted in `FlowTok._forward`'s ctx branch, right after `context_embedder` + the existing 768-d positional embedding. Full self-attention over all T·L sat tokens (spatiotemporal). New flag `sat_context_encoder_layers` (default 0) means no module is built and the forward branch is skipped → bit-identical to arm 7. Experiment scripts are clones of the arm-7 (xattn-B) config / train / tiny-gate / holdout flow.

**Tech Stack:** PyTorch, timm (`Attention`, `Mlp`), ml_collections configs, NCI GADI PBS (gpuhopper), conda env `flowtok`.

**Branch constraint:** Both repos MUST stay on `v2v-7arm-comparison` — the cluster reads the live working tree; switching branches would break the in-flight XL/H/m8 jobs.

**Spec:** `docs/superpowers/specs/2026-05-26-arm8-sat-context-encoder-design.md`

---

## File Structure

- **Modify** `libs/model/flowtok_t2i.py` — add `SatEncoderLayer` + `SatContextEncoder` classes; construct in `FlowTok.__init__` (inside the `use_cross_attention` block); insert call in `_forward` ctx branch.
- **Modify** `tests/test_ablation_units.py` — add 3 arm-8 tests (custom runner auto-discovers `test_*`).
- **Create** `configs/Sat2Radar-v2v-cmp-satenc-B-2021summer_gadi.py` — clone of xattn-B + flag.
- **Create** `configs/Sat2Radar-v2v-cmp-satenc-tiny_gadi.py` — clone of xattn-tiny + flag.
- **Create** `train_v2v_cmp_satenc_full_gadi.sh` — clone of xattn-full + auto-qsub holdout test at 60k.
- **Create** `train_v2v_cmp_satenc_tiny_gadi.sh` — clone of xattn-tiny gate → qsub satenc full on pass.
- **Create** `holdout_test_v2v_cmp_satenc_gadi.sh` — clone of xattn holdout (gen-metrics ON).

**Conventions for all commands below:**
- `PY=/scratch/kl02/yh0308/miniconda3/envs/flowtok/bin/python`
- `FT=/scratch/kl02/yh0308/Projv2v/FlowTok`
- Run all `python`/`git`/`qsub` commands from `$FT`.

---

### Task 1: `SatContextEncoder` module + wiring (TDD)

**Files:**
- Modify: `libs/model/flowtok_t2i.py` (classes after `CrossAttention` ~line 105; construction ~line 229; `_forward` ctx branch ~line 366)
- Test: `tests/test_ablation_units.py` (append new tests)

- [ ] **Step 1: Write the failing tests** — append to `tests/test_ablation_units.py`:

```python
def test_satenc_defaults_off():
    import torch
    from types import SimpleNamespace
    from libs.model.flowtok_t2i import FlowTok
    base = dict(channels=16, clip_dim=16, num_clip_token=77, cfg_indicator=0.0,
                noising_type="none", noising_scale=0.1,
                textVAE=SimpleNamespace(num_blocks=1, hidden_dim=32, num_attention_heads=2,
                                        dropout_prob=0.0, clip_loss_weight=0.0))
    cfg = SimpleNamespace(use_cross_attention=True, **base)  # no sat_context_encoder_layers
    m = FlowTok(cfg, num_latent_tokens=77, hidden_size=128, depth=2, num_heads=8)
    assert getattr(m, "use_sat_context_encoder", False) is False
    assert not hasattr(m, "sat_context_encoder")


def test_satenc_construction_and_forward():
    import torch
    from types import SimpleNamespace
    from libs.model.flowtok_t2i import FlowTok, SatContextEncoder
    base = dict(channels=16, clip_dim=16, num_clip_token=77, cfg_indicator=0.0,
                noising_type="none", noising_scale=0.1,
                textVAE=SimpleNamespace(num_blocks=1, hidden_dim=32, num_attention_heads=2,
                                        dropout_prob=0.0, clip_loss_weight=0.0))
    cfg = SimpleNamespace(use_cross_attention=True, sat_context_encoder_layers=6, **base)
    m = FlowTok(cfg, num_latent_tokens=77, hidden_size=128, depth=2, num_heads=8)
    assert m.use_sat_context_encoder is True
    assert isinstance(m.sat_context_encoder, SatContextEncoder)
    assert len(m.sat_context_encoder.layers) == 6
    x = torch.randn(2, 2 * 77, 16); ctx = torch.randn(2, 2 * 77, 16); t = torch.rand(2)
    nullind = torch.zeros(2, dtype=torch.bool)
    out = m(x, t=t, null_indicator=nullind, context=ctx)[0]
    assert out.shape == (2, 2 * 77, 16)
    assert torch.isfinite(out).all()
    # i2i length too (L = 77)
    xi = torch.randn(2, 77, 16); ci = torch.randn(2, 77, 16); ti = torch.rand(2)
    ni = torch.zeros(2, dtype=torch.bool)
    oi = m(xi, t=ti, null_indicator=ni, context=ci)[0]
    assert oi.shape == (2, 77, 16) and torch.isfinite(oi).all()


def test_satenc_off_state_dict_matches_arm7():
    from types import SimpleNamespace
    from libs.model.flowtok_t2i import FlowTok
    base = dict(channels=16, clip_dim=16, num_clip_token=77, cfg_indicator=0.0,
                noising_type="none", noising_scale=0.1,
                textVAE=SimpleNamespace(num_blocks=1, hidden_dim=32, num_attention_heads=2,
                                        dropout_prob=0.0, clip_loss_weight=0.0))
    m_arm7 = FlowTok(SimpleNamespace(use_cross_attention=True, **base),
                     num_latent_tokens=77, hidden_size=128, depth=2, num_heads=8)
    m_off = FlowTok(SimpleNamespace(use_cross_attention=True, sat_context_encoder_layers=0, **base),
                    num_latent_tokens=77, hidden_size=128, depth=2, num_heads=8)
    assert not hasattr(m_off, "sat_context_encoder")
    assert set(m_off.state_dict().keys()) == set(m_arm7.state_dict().keys())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd $FT && $PY -c "import tests.test_ablation_units as t; t.test_satenc_construction_and_forward()"`
Expected: FAIL — `ImportError: cannot import name 'SatContextEncoder'`.

- [ ] **Step 3: Add the two classes** — in `libs/model/flowtok_t2i.py`, immediately AFTER the `CrossAttention` class (after its `return self.proj(out)`, ~line 105), insert:

```python
class SatEncoderLayer(nn.Module):
    """One pre-norm self-attention + MLP block for the satellite context encoder.
    No adaLN, no timestep conditioning, no cross-attn — the satellite condition is
    clean input, independent of the diffusion timestep (Arm 8)."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = Mlp(in_features=hidden_size,
                       hidden_features=int(hidden_size * mlp_ratio),
                       act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SatContextEncoder(nn.Module):
    """Timestep-independent self-attention encoder that refines satellite K/V
    tokens before per-block cross-attention (Arm 8). Full self-attention over the
    whole sat sequence (T*L tokens) => spatiotemporal, not axial."""
    def __init__(self, hidden_size, num_heads, depth=6, mlp_ratio=4.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [SatEncoderLayer(hidden_size, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
```

- [ ] **Step 4: Construct the encoder** — in `FlowTok.__init__`, find the existing cross-attention block (currently):

```python
        if self.use_cross_attention:
            self.context_embedder = nn.Linear(config.channels, hidden_size, bias=True)
        self.initialize_weights()
```

Replace it with (note: still BEFORE `self.initialize_weights()` so the new module gets standard init):

```python
        if self.use_cross_attention:
            self.context_embedder = nn.Linear(config.channels, hidden_size, bias=True)
            self.sat_ctx_layers = getattr(config, "sat_context_encoder_layers", 0)
            self.use_sat_context_encoder = self.sat_ctx_layers > 0
            if self.use_sat_context_encoder:
                self.sat_context_encoder = SatContextEncoder(
                    hidden_size, num_heads, depth=self.sat_ctx_layers, mlp_ratio=mlp_ratio)
        self.initialize_weights()
```

- [ ] **Step 5: Insert the encoder call in `_forward`** — find the existing ctx branch (currently):

```python
        ctx = None
        if self.use_cross_attention and context is not None:
            ctx = self.context_embedder(context)
            ctx = ctx + self._build_pos_embed(seq_len=ctx.shape[1], device=ctx.device, dtype=ctx.dtype)
```

Append one line so it becomes:

```python
        ctx = None
        if self.use_cross_attention and context is not None:
            ctx = self.context_embedder(context)
            ctx = ctx + self._build_pos_embed(seq_len=ctx.shape[1], device=ctx.device, dtype=ctx.dtype)
            if getattr(self, "use_sat_context_encoder", False):
                ctx = self.sat_context_encoder(ctx)
```

- [ ] **Step 6: Run the new tests to verify they pass**

Run:
```bash
cd $FT && $PY -c "import tests.test_ablation_units as t; t.test_satenc_defaults_off(); t.test_satenc_construction_and_forward(); t.test_satenc_off_state_dict_matches_arm7(); print('ARM8 UNIT OK')"
```
Expected: prints `ARM8 UNIT OK` (no assertion error).

- [ ] **Step 7: Run the FULL ablation suite (no regression on arms 1–7)**

Run: `cd $FT && $PY tests/test_ablation_units.py`
Expected: final line `N/N passed` with N = previous count + 3, exit code 0. Every line `PASS test_*`.

- [ ] **Step 8: Commit**

```bash
cd $FT && git add libs/model/flowtok_t2i.py tests/test_ablation_units.py
git commit -m "feat(arm8): satellite context self-attention encoder

SatContextEncoder (N pre-norm self-attn layers @ hidden_size, reusing timm
Attention/Mlp) refines sat K/V before arm7 cross-attention. Flag
sat_context_encoder_layers (default 0) => bit-identical arm7. +3 unit tests
(defaults-off, construction+forward i2i/v2v, off-state_dict==arm7).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: Config clones (satenc-B + satenc-tiny)

**Files:**
- Create: `configs/Sat2Radar-v2v-cmp-satenc-B-2021summer_gadi.py`
- Create: `configs/Sat2Radar-v2v-cmp-satenc-tiny_gadi.py`

- [ ] **Step 1: Clone the B config**

Run: `cd $FT && cp configs/Sat2Radar-v2v-cmp-xattn-B-2021summer_gadi.py configs/Sat2Radar-v2v-cmp-satenc-B-2021summer_gadi.py`

- [ ] **Step 2: Edit satenc-B — add the flag** (in the `model = Args(...)` block, after the `use_cross_attention=True,` line):

Change:
```python
    use_cross_attention=True,   # Arm 7: sat tokens as cross-attention KV.
    noising_type="none",        # cross_attention mode uses randn for radar noise.
```
to:
```python
    use_cross_attention=True,   # Arm 7: sat tokens as cross-attention KV.
    sat_context_encoder_layers=6,  # Arm 8: refine sat KV with N=6 self-attn layers.
    noising_type="none",        # cross_attention mode uses randn for radar noise.
```

- [ ] **Step 3: Edit satenc-B — change the workdir** (so it does not collide with arm 7):

Change:
```python
    config.workdir = (
        "/scratch/kl02/yh0308/Projv2v/Experiments/"
        "sat2radar_flowtok_v2v_cmp_xattn_B"
    )
```
to:
```python
    config.workdir = (
        "/scratch/kl02/yh0308/Projv2v/Experiments/"
        "sat2radar_flowtok_v2v_cmp_satenc_B"
    )
```

(Leave `config.nnet.name = "flowtok-b"` and everything else unchanged.)

- [ ] **Step 4: Clone + edit the tiny config**

Run: `cd $FT && cp configs/Sat2Radar-v2v-cmp-xattn-tiny_gadi.py configs/Sat2Radar-v2v-cmp-satenc-tiny_gadi.py`

Then apply THREE edits to `configs/Sat2Radar-v2v-cmp-satenc-tiny_gadi.py`:

(a) add the flag after `use_cross_attention=True,`:
```python
    use_cross_attention=True,   # Arm 7: sat tokens as cross-attention KV.
    sat_context_encoder_layers=6,  # Arm 8: refine sat KV with N=6 self-attn layers.
```

(b) reuse arm-7's tiny overfit dataset (avoid building a new one) — change the dataset `filelist_path`:
```python
            "sat2radar_flowtok_v2v_cmp_xattn_tiny/dataset_filelist.pkl"
```
(leave this pointing at the EXISTING xattn tiny pkl — the overfit clips are arm-agnostic).

(c) change the workdir:
```python
        "sat2radar_flowtok_v2v_cmp_satenc_tiny"
```

- [ ] **Step 5: Compile-check both configs**

Run:
```bash
cd $FT && $PY -m py_compile configs/Sat2Radar-v2v-cmp-satenc-B-2021summer_gadi.py configs/Sat2Radar-v2v-cmp-satenc-tiny_gadi.py && echo CONFIGS_OK
```
Expected: `CONFIGS_OK`.

- [ ] **Step 6: Param-count sanity (delta ≈ +42M vs arm 7)**

Run:
```bash
cd $FT && $PY -c "
from types import SimpleNamespace
from libs.model.flowtok_t2i import FlowTok
base=dict(channels=16,clip_dim=16,num_clip_token=77,cfg_indicator=0.0,noising_type='none',noising_scale=0.1,use_cross_attention=True,
          textVAE=SimpleNamespace(num_blocks=6,hidden_dim=256,num_attention_heads=4,dropout_prob=0.1,clip_loss_weight=0.0,align_quantized=False,use_pretrained=False,tokenizer_checkpoint='',freeze_encoder=False))
p=lambda m:sum(x.numel() for x in m.parameters())/1e6
m7=FlowTok(SimpleNamespace(**base),num_latent_tokens=77,hidden_size=768,depth=12,num_heads=12)
m8=FlowTok(SimpleNamespace(sat_context_encoder_layers=6,**base),num_latent_tokens=77,hidden_size=768,depth=12,num_heads=12)
d=p(m8)-p(m7); print('arm7',round(p(m7),1),'M  arm8',round(p(m8),1),'M  delta',round(d,1),'M'); assert 38<d<46, d; print('PARAM_DELTA_OK')
"
```
Expected: `delta ~42 M` and `PARAM_DELTA_OK`.

- [ ] **Step 7: Commit**

```bash
cd $FT && git add configs/Sat2Radar-v2v-cmp-satenc-B-2021summer_gadi.py configs/Sat2Radar-v2v-cmp-satenc-tiny_gadi.py
git commit -m "feat(arm8): satenc-B + satenc-tiny configs (clone of xattn + N=6 flag)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: PBS scripts (full train + tiny gate + holdout)

**Files:**
- Create: `train_v2v_cmp_satenc_full_gadi.sh`
- Create: `train_v2v_cmp_satenc_tiny_gadi.sh`
- Create: `holdout_test_v2v_cmp_satenc_gadi.sh`

- [ ] **Step 1: Full train script** — clone xattn-full and rewrite arm-specific tokens + ADD auto-qsub of the holdout test in the TARGET-reached branch:

```bash
cd $FT && cp train_v2v_cmp_xattn_full_gadi.sh train_v2v_cmp_satenc_full_gadi.sh
```

Then edit `train_v2v_cmp_satenc_full_gadi.sh`:

(a) `#PBS -N v2v_cmp_xattn_full` → `#PBS -N v2v_cmp_satenc_full`
(b) `CFG=$FT/configs/Sat2Radar-v2v-cmp-xattn-B-2021summer_gadi.py` → `CFG=$FT/configs/Sat2Radar-v2v-cmp-satenc-B-2021summer_gadi.py`
(c) `WD=.../sat2radar_flowtok_v2v_cmp_xattn_B` → `WD=.../sat2radar_flowtok_v2v_cmp_satenc_B`
(d) `JOBLOG=...${PBS_JOBID}_v2v_cmp_xattn_full.log` → `..._v2v_cmp_satenc_full.log`
(e) every remaining log string `v2v cmp xattn full` → `v2v cmp satenc full`
(f) the two `qsub train_v2v_cmp_xattn_full_gadi.sh` → `qsub train_v2v_cmp_satenc_full_gadi.sh`
(g) in the TARGET-reached branch, after `rm -f "$WD/RESUBMIT_STALLED"`, add the auto-test submit. The branch must read:

```bash
if [ "$NEW" -ge "$TARGET" ]; then
  echo "[$(date '+%F %T')] v2v cmp satenc full: TARGET reached at $NEW. Done."
  rm -f "$WD/RESUBMIT_STALLED"
  echo "[$(date '+%F %T')] auto-submitting holdout test for satenc"
  cd $FT && qsub holdout_test_v2v_cmp_satenc_gadi.sh
elif [ "$RC" = "124" ] || [ "$NEW" -gt "$STEP" ]; then
```

- [ ] **Step 2: Tiny gate script** — clone xattn-tiny and rewrite arm-specific tokens (gate logic unchanged):

```bash
cd $FT && cp train_v2v_cmp_xattn_tiny_gadi.sh train_v2v_cmp_satenc_tiny_gadi.sh
```

Then edit `train_v2v_cmp_satenc_tiny_gadi.sh`:

(a) `#PBS -N v2v_cmp_xattn_tiny` → `#PBS -N v2v_cmp_satenc_tiny`
(b) `CFG=$FT/configs/Sat2Radar-v2v-cmp-xattn-tiny_gadi.py` → `CFG=$FT/configs/Sat2Radar-v2v-cmp-satenc-tiny_gadi.py`
(c) `TD=.../sat2radar_flowtok_v2v_cmp_xattn_tiny` → `TD=.../sat2radar_flowtok_v2v_cmp_satenc_tiny`
(d) `JOBLOG=...${PBS_JOBID}_v2v_cmp_xattn_tiny.log` → `..._v2v_cmp_satenc_tiny.log`
(e) every remaining log string `v2v cmp xattn tiny` → `v2v cmp satenc tiny`
(f) `qsub train_v2v_cmp_xattn_full_gadi.sh` → `qsub train_v2v_cmp_satenc_full_gadi.sh`

- [ ] **Step 3: Holdout test script** — clone xattn holdout and rewrite arm-specific tokens (gen-metrics already ON in this script — do NOT add any skip flag):

```bash
cd $FT && cp holdout_test_v2v_cmp_xattn_gadi.sh holdout_test_v2v_cmp_satenc_gadi.sh
```

Then edit `holdout_test_v2v_cmp_satenc_gadi.sh`:

(a) `#PBS -N htest_v2v_xattn` → `#PBS -N htest_v2v_satenc`
(b) `CFG=$FT/configs/Sat2Radar-v2v-cmp-xattn-B-2021summer_gadi.py` → `CFG=$FT/configs/Sat2Radar-v2v-cmp-satenc-B-2021summer_gadi.py`
(c) `WD=.../sat2radar_flowtok_v2v_cmp_xattn_B` → `WD=.../sat2radar_flowtok_v2v_cmp_satenc_B`
(d) `JOBLOG=...${PBS_JOBID}_htest_v2v_xattn.log` → `..._htest_v2v_satenc.log`
(e) the final echo `xattn v2v holdout test done` → `satenc v2v holdout test done`

(`CKPT=$WD/ckpts/60000.ckpt`, `OUT=$WD/test_holdout_60000`, `TEST_PKL`, and the python invocation stay as-is — they key off `$WD`/`$CFG`.)

- [ ] **Step 4: Syntax-check all three scripts**

Run:
```bash
cd $FT && for s in train_v2v_cmp_satenc_full_gadi.sh train_v2v_cmp_satenc_tiny_gadi.sh holdout_test_v2v_cmp_satenc_gadi.sh; do bash -n "$s" && echo "OK $s"; done
```
Expected: `OK` for all three.

- [ ] **Step 5: Verify no stale `xattn` tokens remain in the clones**

Run:
```bash
cd $FT && ! grep -n "xattn" train_v2v_cmp_satenc_full_gadi.sh train_v2v_cmp_satenc_tiny_gadi.sh holdout_test_v2v_cmp_satenc_gadi.sh && echo "NO_STALE_XATTN"
```
Expected: `NO_STALE_XATTN` (grep finds nothing). If it prints lines, fix those occurrences and re-run.

- [ ] **Step 6: Commit**

```bash
cd $FT && git add train_v2v_cmp_satenc_full_gadi.sh train_v2v_cmp_satenc_tiny_gadi.sh holdout_test_v2v_cmp_satenc_gadi.sh
git commit -m "feat(arm8): satenc PBS scripts (tiny gate -> full -> auto holdout test)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4: Launch the tiny gate

**Files:** none (execution + bookkeeping).

- [ ] **Step 1: Confirm branch (hard constraint)**

Run: `cd $FT && git rev-parse --abbrev-ref HEAD`
Expected: `v2v-7arm-comparison`. If not, STOP — do not switch; investigate.

- [ ] **Step 2: Submit the tiny overfit gate**

Run: `cd $FT && qsub train_v2v_cmp_satenc_tiny_gadi.sh`
Expected: a job id like `NNNNNNNN.gadi-pbs`. (On pass — diff_loss < 0.3 & step ≥ 11900 — it auto-qsubs `train_v2v_cmp_satenc_full_gadi.sh`, which at 60k auto-qsubs the holdout test.)

- [ ] **Step 3: Record the job + update memory**

Append the satenc job id and the arm-8 pipeline to memory `v2v-7arm-comparison.md` (and note arm 9 = deferred factorized design). Confirm in qstat:

Run: `qstat -u yh0308 | grep -i satenc`
Expected: the tiny job listed (Q or R).

---

## Self-Review

**Spec coverage:**
- §3.1 module → Task 1 Step 3. §3.2 construction → Task 1 Step 4. §3.3 forward → Task 1 Step 5.
- §4 locked params (N=6, width=768, no VAE head, no timestep, full attn, B) → encoded in classes (no VAE head / no adaLN) + configs (flag=6, name=flowtok-b). ✓
- §5 backward-compat → Task 1 Steps 1/6/7 (defaults-off test, state_dict==arm7 test, full-suite no-regression). ✓
- §6 experiment setup (config/tiny/train/test) → Tasks 2 & 3; auto-qsub chain → Task 3 Step 1(g) + tiny Step 2(f). ✓
- §7 testing (backward-compat, construction, shape, i2i+v2v forward, param-count) → Task 1 tests + Task 2 Step 6. ✓ Tiny gate + holdout → Task 4 + Task 3.

**Placeholder scan:** none — every code/edit step shows exact content or exact old→new strings; every command has expected output.

**Type/name consistency:** `SatEncoderLayer`, `SatContextEncoder`, attribute `sat_context_encoder` / `use_sat_context_encoder` / `sat_ctx_layers`, flag `sat_context_encoder_layers`, workdir `sat2radar_flowtok_v2v_cmp_satenc_B`, scripts `*_satenc_*` — used identically across Tasks 1–4. ✓
