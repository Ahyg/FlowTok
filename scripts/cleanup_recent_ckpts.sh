#!/bin/bash
# Free space by deleting the HEAVY weight blobs of this session's recent
# experiments, while KEEPING every small record (metrics.json, align_*.json,
# log0.txt, *.md, sample/test panels & gifs).
#
# Scope is an explicit allow-list of 7 ssd_2 dirs created this session.
# It deliberately does NOT touch the large *older* ssd_1 runs
# (sat2radar_flowtok_run_v2v_*, *_bidir_*, *_uni_sat10ch_run1, ...) — those
# predate this session, were not created here, and are out of scope.
#
# NOTE: real disk pressure is on ssd_1 (96%); every recent experiment and the
# upcoming sweep live on ssd_2 (560G+ free), so this is hygiene, not a
# prerequisite. Anchors the sweep reuses (joint_tok_align/align_{A,B}.json and
# v2v_jointtok_{A,B}_run1/test8000/metrics.json) are explicitly preserved.
set -u
EXP="/mnt/ssd_2/yghu/Experiments"
log(){ echo "[cleanup] $*"; }

before=$(df -P /mnt/ssd_2 | tail -1 | awk '{print $4}')

# Token-count sweep — fully analyzed (docs/.../2026-05-14-token-sweep-results.md
# + eval_*/metrics + latent_utilization kept). Drop checkpoint state only.
for d in sat10ch_ae_tok77_run1 sat10ch_ae_tok128_run1 sat10ch_ae_tok256_run1; do
  p="${EXP}/${d}"; [ -d "$p" ] || continue
  log "prune ${d}: checkpoint-*/  + top-level *.bin"
  rm -rf "${p}"/checkpoint-* 2>/dev/null
  rm -f  "${p}"/pytorch_model*.bin "${p}"/optimizer.bin 2>/dev/null
done

# Joint-AE pilots — alignment already evaluated into joint_tok_align/align_{A,B}.json
# (kept). log0.txt + samples kept.
for d in joint_ae_sep_run1 joint_ae_joint_run1; do
  p="${EXP}/${d}"; [ -d "$p" ] || continue
  log "prune ${d}: {sat,radar}/checkpoint-*/"
  rm -rf "${p}"/sat/checkpoint-* "${p}"/radar/checkpoint-* 2>/dev/null
done

# v2v jointtok pilots — KEEP test8000/ (metrics.json is the §6 anchor) + samples.
# Only the trainable ckpts go.
for d in v2v_jointtok_A_run1 v2v_jointtok_B_run1; do
  p="${EXP}/${d}"; [ -d "$p" ] || continue
  log "prune ${d}: ckpts/  (test8000/metrics.json + samples kept)"
  rm -rf "${p}"/ckpts 2>/dev/null
done

after=$(df -P /mnt/ssd_2 | tail -1 | awk '{print $4}')
freed_gb=$(awk -v a="$before" -v b="$after" 'BEGIN{printf "%.1f",(b-a)/1048576}')
log "freed ~${freed_gb} GB on ssd_2 (avail ${before} -> ${after} KiB)"
log "kept (anchors): joint_tok_align/align_{A,B}.json, v2v_jointtok_{A,B}_run1/test8000/metrics.json"
echo "=== remaining sizes ==="
du -sh "${EXP}"/sat10ch_ae_tok*_run1 "${EXP}"/joint_ae_*_run1 \
       "${EXP}"/v2v_jointtok_*_run1 2>/dev/null
