# Arm 9 — Factorized Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the radar DiT's full self-attention with factorized frame-local spatial + axial temporal self-attention (keeping arm7's cross-attention + MLP), behind a flag that defaults to exact arm1-8 behavior.

**Architecture:** A new `FactorizedDiTBlock` (Approach B — `DiTBlock` left byte-identical so running jobs are unaffected) mirrors the cross-attn `DiTBlock` but swaps the single full self-attention for two sub-layers: frame-local spatial SA (`[B,T·L,D]→[B·T,L,D]`) and axial temporal SA (`→[B·L,T,D]`), adaLN-Zero over 4 sub-layers (chunk 12). `FlowTok.__init__` builds these blocks when `use_factorized_attn=True`; `_forward` is unchanged. Experiment scripts clone the arm-7 (xattn-B) flow.

**Tech Stack:** PyTorch, timm (`Attention`, `Mlp`), ml_collections configs, NCI GADI PBS (gpuhopper), conda env `flowtok`.

**Branch constraint:** Stay on `v2v-7arm-comparison` in both repos — the cluster reads the live working tree; in-flight XL/H/m8 jobs depend on it. Do NOT switch branches or merge.

**Spec:** `docs/superpowers/specs/2026-05-26-arm9-factorized-attention-design.md`

**Conventions for all commands:**
- `PY=/scratch/kl02/yh0308/miniconda3/envs/flowtok/bin/python`
- `FT=/scratch/kl02/yh0308/Projv2v/FlowTok`
- Run all commands from `$FT`. The test runner is `$PY tests/test_ablation_units.py` (custom `_main()` auto-discovers `test_*`; single test via `$PY -c "import tests.test_ablation_units as t; t.NAME()"`).

---

## File Structure

- **Modify** `libs/model/flowtok_t2i.py` — add `FactorizedDiTBlock` class (after `DiTBlock`); change the `self.blocks` construction in `FlowTok.__init__` to a flag switch. `DiTBlock` and `_forward` are NOT changed.
- **Modify** `tests/test_ablation_units.py` — add 4 arm-9 tests.
- **Create** `configs/Sat2Radar-v2v-cmp-fact-B-2021summer_gadi.py`, `configs/Sat2Radar-v2v-cmp-fact-tiny_gadi.py`.
- **Create** `train_v2v_cmp_fact_full_gadi.sh`, `train_v2v_cmp_fact_tiny_gadi.sh`, `holdout_test_v2v_cmp_fact_gadi.sh`.

---

### Task 1: `FactorizedDiTBlock` + block-list switch (TDD)

**Files:**
- Modify: `libs/model/flowtok_t2i.py` (add class after `DiTBlock` ~line 180; change block-list construction ~lines 259-262)
- Test: `tests/test_ablation_units.py` (append)

- [ ] **Step 1: Write the failing tests** — append to `tests/test_ablation_units.py`:

```python
def _arm9_base():
    from types import SimpleNamespace
    return dict(channels=16, clip_dim=16, num_clip_token=77, cfg_indicator=0.0,
                noising_type="none", noising_scale=0.1,
                textVAE=SimpleNamespace(num_blocks=1, hidden_dim=32, num_attention_heads=2,
                                        dropout_prob=0.0, clip_loss_weight=0.0))


def test_factorized_defaults_off():
    from types import SimpleNamespace
    from libs.model.flowtok_t2i import FlowTok, DiTBlock
    m = FlowTok(SimpleNamespace(use_cross_attention=True, **_arm9_base()),
                num_latent_tokens=77, hidden_size=128, depth=2, num_heads=8)
    assert all(isinstance(b, DiTBlock) for b in m.blocks)


def test_factorized_off_state_dict_matches_arm7():
    from types import SimpleNamespace
    from libs.model.flowtok_t2i import FlowTok
    m_arm7 = FlowTok(SimpleNamespace(use_cross_attention=True, **_arm9_base()),
                     num_latent_tokens=77, hidden_size=128, depth=2, num_heads=8)
    m_off = FlowTok(SimpleNamespace(use_cross_attention=True, use_factorized_attn=False, **_arm9_base()),
                    num_latent_tokens=77, hidden_size=128, depth=2, num_heads=8)
    assert set(m_off.state_dict().keys()) == set(m_arm7.state_dict().keys())


def test_factorized_construction_and_forward():
    import torch
    from types import SimpleNamespace
    from libs.model.flowtok_t2i import FlowTok, FactorizedDiTBlock
    cfg = SimpleNamespace(use_cross_attention=True, use_factorized_attn=True, **_arm9_base())
    m = FlowTok(cfg, num_latent_tokens=77, hidden_size=128, depth=2, num_heads=8)
    assert all(isinstance(b, FactorizedDiTBlock) for b in m.blocks)
    assert len(m.blocks) == 2
    b = m.blocks[0]
    assert hasattr(b, "attn_sp") and hasattr(b, "attn_tp") and hasattr(b, "cross_attn")
    assert b.adaLN_modulation[-1].out_features == 12 * 128
    # v2v: T=2, L=77 -> seq 154
    x = torch.randn(2, 2 * 77, 16); ctx = torch.randn(2, 2 * 77, 16)
    ni = torch.zeros(2, dtype=torch.bool)
    out = m(x, t=torch.rand(2), null_indicator=ni, context=ctx)[0]
    assert out.shape == (2, 2 * 77, 16) and torch.isfinite(out).all()
    # i2i: T=1, seq 77
    xi = torch.randn(2, 77, 16); ci = torch.randn(2, 77, 16)
    oi = m(xi, t=torch.rand(2), null_indicator=ni, context=ci)[0]
    assert oi.shape == (2, 77, 16) and torch.isfinite(oi).all()


def test_factorized_reshape_roundtrip_and_grouping():
    import torch
    from libs.model.flowtok_t2i import FactorizedDiTBlock
    blk = FactorizedDiTBlock(hidden_size=8, num_heads=2, n_per_frame=4)
    B, T, L, D = 1, 3, 4, 8
    x = torch.randn(B, T * L, D)
    ident = lambda z: z
    # (a) round-trip: reshape -> identity attn -> inverse reshape == input
    assert torch.allclose(blk._frame_local(ident, x), x, atol=1e-6)
    assert torch.allclose(blk._axial_temporal(ident, x), x, atol=1e-6)
    # (b) grouping: ch0 = frame id, ch1 = token idx; mean-pool attn over the group dim
    g = torch.zeros(B, T * L, D)
    for tt in range(T):
        for ll in range(L):
            g[0, tt * L + ll, 0] = tt
            g[0, tt * L + ll, 1] = ll
    meanpool = lambda z: z.mean(dim=1, keepdim=True).expand_as(z)
    fl = blk._frame_local(meanpool, g)      # group = within-frame
    for tt in range(T):
        assert torch.allclose(fl[0, tt * L:(tt + 1) * L, 0], torch.full((L,), float(tt)), atol=1e-5)  # frame id kept
        assert torch.allclose(fl[0, tt * L:(tt + 1) * L, 1], torch.full((L,), (L - 1) / 2), atol=1e-5)  # token idx averaged
    tp = blk._axial_temporal(meanpool, g)   # group = across frames at fixed token idx
    for tt in range(T):
        for ll in range(L):
            assert abs(tp[0, tt * L + ll, 1].item() - ll) < 1e-5            # token idx kept
            assert abs(tp[0, tt * L + ll, 0].item() - (T - 1) / 2) < 1e-5   # frame id averaged
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd $FT && $PY -c "import tests.test_ablation_units as t; t.test_factorized_construction_and_forward()"`
Expected: FAIL — `ImportError: cannot import name 'FactorizedDiTBlock'`.

- [ ] **Step 3: Add `FactorizedDiTBlock`** — in `libs/model/flowtok_t2i.py`, immediately AFTER the `DiTBlock` class (after its `_forward` returns `x`, ~line 180) and BEFORE the next class (`FinalLayer`). `Attention`, `Mlp`, `CrossAttention`, `modulate` are already defined/imported above:

```python
class FactorizedDiTBlock(nn.Module):
    """Arm 9: factorized divided space-time attention with adaLN-Zero.
    Replaces DiTBlock's single full self-attention with frame-local spatial SA +
    axial temporal SA; keeps arm7's cross-attention (sat K/V) + MLP. 4 sub-layers,
    adaLN chunk(12). DiTBlock is left untouched."""
    def __init__(self, hidden_size, num_heads, n_per_frame, mlp_ratio=4.0):
        super().__init__()
        self.n_per_frame = n_per_frame
        self.norm_sp = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn_sp = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm_tp = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn_tp = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm_ca = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio),
                       act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 12 * hidden_size, bias=True))

    def forward(self, x, c, context=None):
        return torch.utils.checkpoint.checkpoint(self._forward, x, c, context, use_reentrant=False)

    def _frame_local(self, attn, x):
        B, S, D = x.shape; L = self.n_per_frame; T = S // L
        x = x.reshape(B * T, L, D)
        x = attn(x)
        return x.reshape(B, S, D)

    def _axial_temporal(self, attn, x):
        B, S, D = x.shape; L = self.n_per_frame; T = S // L
        x = x.reshape(B, T, L, D).transpose(1, 2).reshape(B * L, T, D)
        x = attn(x)
        return x.reshape(B, L, T, D).transpose(1, 2).reshape(B, S, D)

    def _forward(self, x, c, context=None):
        assert x.shape[1] % self.n_per_frame == 0, "seq_len must be divisible by n_per_frame"
        (sh_sp, sc_sp, g_sp,
         sh_tp, sc_tp, g_tp,
         sh_ca, sc_ca, g_ca,
         sh_mlp, sc_mlp, g_mlp) = self.adaLN_modulation(c).chunk(12, dim=1)
        x = x + g_sp.unsqueeze(1) * self._frame_local(self.attn_sp, modulate(self.norm_sp(x), sh_sp, sc_sp))
        x = x + g_tp.unsqueeze(1) * self._axial_temporal(self.attn_tp, modulate(self.norm_tp(x), sh_tp, sc_tp))
        x = x + g_ca.unsqueeze(1) * self.cross_attn(modulate(self.norm_ca(x), sh_ca, sc_ca), context)
        x = x + g_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), sh_mlp, sc_mlp))
        return x
```

- [ ] **Step 4: Change the block-list construction to a flag switch** — in `FlowTok.__init__`, find:

```python
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_cross_attn=self.use_cross_attention)
            for _ in range(depth)
        ])
```

Replace with (the `else` branch is byte-identical to the original → default behavior unchanged):

```python
        if getattr(config, "use_factorized_attn", False):
            self.blocks = nn.ModuleList([
                FactorizedDiTBlock(hidden_size, num_heads,
                                   n_per_frame=num_latent_tokens, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_cross_attn=self.use_cross_attention)
                for _ in range(depth)
            ])
```

(`num_latent_tokens`, `hidden_size`, `num_heads`, `mlp_ratio`, `depth` are all `__init__` params in scope here. `config` is the model-args object that already provides `use_cross_attention`.)

- [ ] **Step 5: Run the new tests to verify pass**

Run:
```bash
cd $FT && $PY -c "import tests.test_ablation_units as t; t.test_factorized_defaults_off(); t.test_factorized_off_state_dict_matches_arm7(); t.test_factorized_construction_and_forward(); t.test_factorized_reshape_roundtrip_and_grouping(); print('ARM9 UNIT OK')"
```
Expected: prints `ARM9 UNIT OK`.

- [ ] **Step 6: Run the FULL ablation suite (no regression on arms 1-8)**

Run: `cd $FT && $PY tests/test_ablation_units.py`
Expected: final line `N/N passed` (N = previous count + 4), exit 0, every line `PASS test_*`. If any prior test FAILS, backward-compat broke — investigate before committing.

- [ ] **Step 7: Commit**

```bash
cd $FT && git add libs/model/flowtok_t2i.py tests/test_ablation_units.py
git commit -m "feat(arm9): factorized frame-local + axial-temporal DiT block

FactorizedDiTBlock replaces DiTBlock's full self-attn with frame-local spatial SA
+ axial temporal SA (TimeSformer divided attn), keeps arm7 cross-attn + MLP, adaLN
chunk(12). Built when use_factorized_attn=True (default off => existing DiTBlock
path, bit-identical arms1-8). DiTBlock untouched. +4 unit tests incl. reshape
round-trip + grouping correctness.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: Configs (fact-B + fact-tiny)

**Files:**
- Create: `configs/Sat2Radar-v2v-cmp-fact-B-2021summer_gadi.py`
- Create: `configs/Sat2Radar-v2v-cmp-fact-tiny_gadi.py`

- [ ] **Step 1: Clone the B config**

Run: `cd $FT && cp configs/Sat2Radar-v2v-cmp-xattn-B-2021summer_gadi.py configs/Sat2Radar-v2v-cmp-fact-B-2021summer_gadi.py`

- [ ] **Step 2: Edit fact-B — add the flag.** In `configs/Sat2Radar-v2v-cmp-fact-B-2021summer_gadi.py`, find:
```python
    use_cross_attention=True,   # Arm 7: sat tokens as cross-attention KV.
    noising_type="none",        # cross_attention mode uses randn for radar noise.
```
Change to:
```python
    use_cross_attention=True,   # Arm 7: sat tokens as cross-attention KV.
    use_factorized_attn=True,   # Arm 9: factorized frame-local + axial-temporal self-attn.
    noising_type="none",        # cross_attention mode uses randn for radar noise.
```

- [ ] **Step 3: Edit fact-B — workdir + docstring.** Change the workdir string `sat2radar_flowtok_v2v_cmp_xattn_B` → `sat2radar_flowtok_v2v_cmp_fact_B`. Replace the module docstring first line `"""7-arm v2v comparison — Arm 7 (xattn): cross-attention conditioning.` with `"""Arm 9 (fact): arm7 cross-attention + factorized frame-local/axial-temporal self-attn.` (leave the rest of the file unchanged; `config.nnet.name="flowtok-b"` stays).

- [ ] **Step 4: Clone + edit the tiny config**

Run: `cd $FT && cp configs/Sat2Radar-v2v-cmp-xattn-tiny_gadi.py configs/Sat2Radar-v2v-cmp-fact-tiny_gadi.py`

Apply THREE edits to `configs/Sat2Radar-v2v-cmp-fact-tiny_gadi.py`:
(a) after `use_cross_attention=True,` add `    use_factorized_attn=True,   # Arm 9: factorized self-attn.`
(b) leave the dataset `filelist_path` (the string containing `sat2radar_flowtok_v2v_cmp_xattn_tiny/dataset_filelist.pkl`) UNCHANGED — reuse the arm-agnostic overfit clips.
(c) change ONLY the `config.workdir` string `sat2radar_flowtok_v2v_cmp_xattn_tiny` → `sat2radar_flowtok_v2v_cmp_fact_tiny` (NOT the filelist string — after editing, the file must still contain `..._xattn_tiny/dataset_filelist.pkl` for the dataset and `..._fact_tiny` for workdir/ckpt_root/sample_dir).

- [ ] **Step 5: Compile + load check**

Run:
```bash
cd $FT && $PY -m py_compile configs/Sat2Radar-v2v-cmp-fact-B-2021summer_gadi.py configs/Sat2Radar-v2v-cmp-fact-tiny_gadi.py && echo CONFIGS_OK
cd $FT && $PY -c "
import importlib.util as u
for f,wd in [('configs/Sat2Radar-v2v-cmp-fact-B-2021summer_gadi.py','fact_B'),('configs/Sat2Radar-v2v-cmp-fact-tiny_gadi.py','fact_tiny')]:
    s=u.spec_from_file_location('c',f); m=u.module_from_spec(s); s.loader.exec_module(m); c=m.get_config()
    assert c.nnet.model_args.use_factorized_attn is True, f
    assert c.nnet.model_args.use_cross_attention is True, f
    assert wd in c.workdir, (wd,c.workdir)
    print(f, 'flags OK workdir', c.workdir.split('/')[-1])
print('CONFIG_LOAD_OK')
"
cd $FT && [ "$(grep -c 'xattn_tiny/dataset_filelist.pkl' configs/Sat2Radar-v2v-cmp-fact-tiny_gadi.py)" = "1" ] && echo "TINY_DATASET_OK"
```
Expected: `CONFIGS_OK`, both `flags OK` lines, `CONFIG_LOAD_OK`, `TINY_DATASET_OK`.

- [ ] **Step 6: Param-count sanity (delta ≈ +49M vs arm 7)**

Run:
```bash
cd $FT && $PY -c "
from types import SimpleNamespace
from libs.model.flowtok_t2i import FlowTok
base=dict(channels=16,clip_dim=16,num_clip_token=77,cfg_indicator=0.0,noising_type='none',noising_scale=0.1,use_cross_attention=True,
          textVAE=SimpleNamespace(num_blocks=6,hidden_dim=256,num_attention_heads=4,dropout_prob=0.1,clip_loss_weight=0.0,align_quantized=False,use_pretrained=False,tokenizer_checkpoint='',freeze_encoder=False))
p=lambda m:sum(x.numel() for x in m.parameters())/1e6
m7=FlowTok(SimpleNamespace(**base),num_latent_tokens=77,hidden_size=768,depth=12,num_heads=12)
m9=FlowTok(SimpleNamespace(use_factorized_attn=True,**base),num_latent_tokens=77,hidden_size=768,depth=12,num_heads=12)
d=p(m9)-p(m7); print('arm7',round(p(m7),1),'M  arm9',round(p(m9),1),'M  delta',round(d,1),'M'); assert 44<d<55, d; print('PARAM_DELTA_OK')
"
```
Expected: delta ≈ 49 M and `PARAM_DELTA_OK`.

- [ ] **Step 7: Commit**

```bash
cd $FT && git add configs/Sat2Radar-v2v-cmp-fact-B-2021summer_gadi.py configs/Sat2Radar-v2v-cmp-fact-tiny_gadi.py
git commit -m "feat(arm9): fact-B + fact-tiny configs (clone of xattn + use_factorized_attn)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: PBS scripts (full / tiny gate / holdout)

**Files:**
- Create: `train_v2v_cmp_fact_full_gadi.sh`, `train_v2v_cmp_fact_tiny_gadi.sh`, `holdout_test_v2v_cmp_fact_gadi.sh`

- [ ] **Step 1: Full train script** — clone + edit + ADD auto-test submit:

```bash
cd $FT && cp train_v2v_cmp_xattn_full_gadi.sh train_v2v_cmp_fact_full_gadi.sh
```
Edit `train_v2v_cmp_fact_full_gadi.sh`:
(a) `#PBS -N v2v_cmp_xattn_full` → `#PBS -N v2v_cmp_fact_full`
(b) `CFG=$FT/configs/Sat2Radar-v2v-cmp-xattn-B-2021summer_gadi.py` → `...-fact-B-2021summer_gadi.py`
(c) `WD=...sat2radar_flowtok_v2v_cmp_xattn_B` → `...sat2radar_flowtok_v2v_cmp_fact_B`
(d) JOBLOG `..._v2v_cmp_xattn_full.log` → `..._v2v_cmp_fact_full.log`
(e) every echo string `v2v cmp xattn full` → `v2v cmp fact full`
(f) both `qsub train_v2v_cmp_xattn_full_gadi.sh` → `qsub train_v2v_cmp_fact_full_gadi.sh`
(g) in the TARGET-reached branch, after `rm -f "$WD/RESUBMIT_STALLED"`, insert the auto-test submit so the branch reads:
```bash
if [ "$NEW" -ge "$TARGET" ]; then
  echo "[$(date '+%F %T')] v2v cmp fact full: TARGET reached at $NEW. Done."
  rm -f "$WD/RESUBMIT_STALLED"
  echo "[$(date '+%F %T')] auto-submitting holdout test for fact"
  cd $FT && qsub holdout_test_v2v_cmp_fact_gadi.sh
elif [ "$RC" = "124" ] || [ "$NEW" -gt "$STEP" ]; then
```

- [ ] **Step 2: Tiny gate script** — clone + edit (gate logic unchanged) + fix the stale echo:

```bash
cd $FT && cp train_v2v_cmp_xattn_tiny_gadi.sh train_v2v_cmp_fact_tiny_gadi.sh
```
Edit `train_v2v_cmp_fact_tiny_gadi.sh`:
(a) `#PBS -N v2v_cmp_xattn_tiny` → `#PBS -N v2v_cmp_fact_tiny`
(b) `CFG=$FT/configs/Sat2Radar-v2v-cmp-xattn-tiny_gadi.py` → `...-fact-tiny_gadi.py`
(c) `TD=...sat2radar_flowtok_v2v_cmp_xattn_tiny` → `...sat2radar_flowtok_v2v_cmp_fact_tiny`
(d) JOBLOG `..._v2v_cmp_xattn_tiny.log` → `..._v2v_cmp_fact_tiny.log`
(e) every echo string `v2v cmp xattn tiny` → `v2v cmp fact tiny`
(f) `qsub train_v2v_cmp_xattn_full_gadi.sh` → `qsub train_v2v_cmp_fact_full_gadi.sh`
(g) the start echo (copied from an interleaved arm, currently mentions `(XL, token_concat_INTERLEAVED+x1, ... seq=2464)`) is inaccurate for arm9 — replace its parenthetical with `(B, factorized frame-local+axial-temporal, cross_attn, 16f, bs=8, radar seq=1232)`.

- [ ] **Step 3: Holdout test script** — clone + edit (gen-metrics already ON; do NOT add a skip flag):

```bash
cd $FT && cp holdout_test_v2v_cmp_xattn_gadi.sh holdout_test_v2v_cmp_fact_gadi.sh
```
Edit `holdout_test_v2v_cmp_fact_gadi.sh`:
(a) `#PBS -N htest_v2v_xattn` → `#PBS -N htest_v2v_fact`
(b) `CFG=$FT/configs/Sat2Radar-v2v-cmp-xattn-B-2021summer_gadi.py` → `...-fact-B-2021summer_gadi.py`
(c) `WD=...sat2radar_flowtok_v2v_cmp_xattn_B` → `...sat2radar_flowtok_v2v_cmp_fact_B`
(d) JOBLOG `..._htest_v2v_xattn.log` → `..._htest_v2v_fact.log`
(e) final echo `xattn v2v holdout test done` → `fact v2v holdout test done`

- [ ] **Step 4: Syntax-check all three**

Run: `cd $FT && for s in train_v2v_cmp_fact_full_gadi.sh train_v2v_cmp_fact_tiny_gadi.sh holdout_test_v2v_cmp_fact_gadi.sh; do bash -n "$s" && echo "OK $s"; done`
Expected: `OK` for all three.

- [ ] **Step 5: Verify no stale `xattn` tokens + correct cross-references**

Run:
```bash
cd $FT && grep -n "xattn" train_v2v_cmp_fact_full_gadi.sh train_v2v_cmp_fact_tiny_gadi.sh holdout_test_v2v_cmp_fact_gadi.sh; echo "grep_exit=$?"
cd $FT && grep -n "qsub" train_v2v_cmp_fact_full_gadi.sh train_v2v_cmp_fact_tiny_gadi.sh
```
Expected: first grep prints nothing and `grep_exit=1`. qsub refs: tiny → `qsub train_v2v_cmp_fact_full_gadi.sh`; full → `qsub holdout_test_v2v_cmp_fact_gadi.sh` (TARGET) and `qsub train_v2v_cmp_fact_full_gadi.sh` (resubmit). If any `xattn` remains, fix and re-run.

- [ ] **Step 6: Commit**

```bash
cd $FT && git add train_v2v_cmp_fact_full_gadi.sh train_v2v_cmp_fact_tiny_gadi.sh holdout_test_v2v_cmp_fact_gadi.sh
git commit -m "feat(arm9): fact PBS scripts (tiny gate -> full -> auto holdout test)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4: Launch the tiny gate

**Files:** none (execution + bookkeeping).

- [ ] **Step 1: Confirm branch**

Run: `cd $FT && git rev-parse --abbrev-ref HEAD`
Expected: `v2v-7arm-comparison`. If not, STOP.

- [ ] **Step 2: Submit the tiny gate**

Run: `cd $FT && qsub train_v2v_cmp_fact_tiny_gadi.sh`
Expected: a job id `NNNNNNNN.gadi-pbs`. (On pass → auto-qsubs `train_v2v_cmp_fact_full_gadi.sh` → at 60k auto-qsubs the holdout test.)

- [ ] **Step 3: Confirm queued + record**

Run: `qstat -f <jobid> 2>/dev/null | grep -E "Job_Name|job_state"`
Expected: `Job_Name = v2v_cmp_fact_tiny`, `job_state = Q` (or R). Record the job id and update memory `v2v-7arm-comparison.md` with the arm9 pipeline + job id.

---

## Self-Review

**Spec coverage:**
- §3.1 `FactorizedDiTBlock` → Task 1 Step 3 (exact class). §3.2 block-list switch → Task 1 Step 4. §3.3 `_forward` unchanged + init → verified by Task 1 Steps 5-6 (forward + state_dict + full suite). §3.4 i2i degradation → forward test (T=1). ✓
- §4 locked params (replace, axial, chunk12, B, Approach B) → encoded in the class (no full self-attn; 2 factorized sub-layers; chunk 12) + configs (`use_factorized_attn=True`, flowtok-b) + DiTBlock untouched. ✓
- §5 backward-compat → Task 1 Steps 1/5/6 (defaults-off, state_dict==arm7, full-suite). ✓
- §6 experiment setup → Tasks 2 & 3; auto-qsub chain → Task 3 Step 1(g) + tiny 2(f). ✓
- §7 testing (backward-compat, construction, forward i2i/v2v, reshape round-trip+grouping, param-count) → Task 1 tests + Task 2 Step 6. ✓

**Placeholder scan:** none — every code/edit step shows exact content or exact old→new strings; every command has expected output.

**Type/name consistency:** `FactorizedDiTBlock`; methods `_frame_local`/`_axial_temporal`/`_forward`; attrs `attn_sp`/`attn_tp`/`cross_attn`/`norm_sp`/`norm_tp`/`norm_ca`/`norm2`/`adaLN_modulation`/`n_per_frame`; flag `use_factorized_attn`; configs `Sat2Radar-v2v-cmp-fact-{B,tiny}`; workdir `sat2radar_flowtok_v2v_cmp_fact_B`; scripts `*_fact_*` — consistent across Tasks 1-4. ✓
