# Benchmark scope, cuts and every attempt

Primary metric: test log-loss (natural log). "d" is the paired per-example difference
`ours - arm` averaged over repeats; **SE** is the between-repeat SD / sqrt(R) over disjoint
draws (it includes the training-sample variance); `se_within` in the JSON is the
within-test-set bootstrap SE (iid rows; Elec2: moving blocks of 48 rows). "Beats" means
|d| > 2 SE. Full numbers: `bench/results/<name>.json`; `python -m bench.summarize`.

## Verdict

- **Offset vs the reference as an ordinary feature: a tie.** On HIGGS, Bank and Elec2 the
  CV-chosen fractional offset (a6) and the anchored model (a7) are within 2 SE of the
  nearly free column (a4) in 14 of 16 (arm, dataset, n) cells; the exceptions are HIGGS
  n = 1000, a6 - a4 = -0.0019 (SE 0.0006) and a7 - a4 = -0.0014 (SE 0.0007), where the
  standardised-feature control a3 ties a6 (+0.0009, SE 0.0011).
- **The pre-registered win over the no-reference model holds on HIGGS and Bank**
  (a6 - a1 from -0.053 (0.004) at n = 250 to -0.028 (0.002) at n = 2000 on HIGGS;
  -0.088 (0.005) and -0.056 (0.004) on Bank) **and fails on Elec2** (+0.079 (0.047),
  +0.006 (0.010)). It shows that a reference helps, which is known.
- **kappa_cv < 1 beats kappa = 1 only where the reference is miscalibrated:** Elec2 (aged
  reference) -0.127 (0.035) at n = 250, -0.158 (0.037) at n = 1000; Bank with a shifted
  reference -0.0077 (0.0012) at n = 1000. Platt recalibration of the reference alone and the
  free column match it there; on HIGGS kappa_cv mostly picks 1.
- **The one-SE rule toward the reference (variant 1) is the only kept improvement:** at
  n <= 250 it beats the pristine free column (HIGGS n = 100: -0.016, SE 0.004) and the free
  column under the same C rule (n = 100: -0.008, SE 0.0044; n = 150: -0.0064, SE 0.0030).
  It never beats the reference used alone at n <= 250 (n = 100: +0.010, SE 0.004).
- **Leakage ablation:** a reference refit on the training rows and used in-sample is worse
  than the same reference cross-fitted by 0.06 to 0.53 log-loss in every cell (HIGGS
  n = 250: 1.144 vs 0.684; a6 - a9_insample = -0.513, SE 0.036).
- **Nonlinear residual:** LightGBM started at kappa x reference (a10) beats every linear arm
  on HIGGS for n >= 500 (a6 - a10 = +0.039, SE 0.003 at n = 1000).

## Cuts (benchmark plan -> what ran) and why

| item | plan | ran | reason |
|---|---|---|---|
| datasets | HIGGS, ACS Income, Elec2 (+ Bank null) | HIGGS, Elec2, Bank | ACS needs `folktables` and the census download (not in the shared env, several GB); budget |
| n_train | 250, 500, 1000, 2000, 5000 | HIGGS 250-2000; Elec2, Bank 250, 1000; variants 100-1000 | 5000 cannot give 10 disjoint repeats from the 38k HIGGS rows left after the reference; budget |
| repeats | 30 | HIGGS 10, Elec2 12 windows, Bank 8, variants 10 | budget; Elec2 cut from 20 and Bank from 10 while HIGGS ran under contention |
| C grid | 8 values | 11 values, 1e-4 .. 10 | smoke run hit the 1e-3 edge |
| anchored grid | kappa 6 x lambda 5 x C 8 | kappa 6 x lambda {inf, 10, 1, 0.1} + lambda 0, x C 11 | lambda = 0 makes kappa irrelevant, so one entry |
| SE | row bootstrap (ACS: cluster over PUMA) | between-repeat SE primary + within bootstrap | repeats are disjoint chunks / windows, so between-repeat SE is valid and includes the training draw |
| a8 (no source columns) | all datasets | HIGGS only | on Elec2 and Bank the reference uses every feature; removing them leaves no residual model |
| a9 (leakage) | all | main runs; not in variants | cost |
| a10 (LightGBM start) | all | main runs and two variants | cost |
| Elec2 origins | 15,000 + | 19,000 + | the Victoria columns and transfer are constant before row 17,424; a first run from 15,000 was stopped after one window because standardising a constant column made every linear arm extrapolate |
| tail metric | \|r\| >= 3 | reported, plus top-10 % \|r\| | HIGGS has few \|r\| >= 3 rows |

Kappa is chosen jointly with C from the same 5-fold grid (no nested CV for the reported test
numbers: the test rows of each repeat are never seen by the grid).

**Compute:** CPU only, 2 threads per process. Measured wall times (shared 8-core machine,
load 9-12 from other jobs; two to three of these processes ran at the same time):
HIGGS 63 min, Elec2 38 min, Bank 31 min, higgs_smalln_1se 28 min, its controls 17 min,
higgs_strongref 13 min, bank_shiftref 20 min; 3.5 h summed, about 2.3 h of wall clock with
the overlap. **The summed time exceeds the 2.5 h budget for a serial run**; an uncontended
smoke run took 80 s per HIGGS repeat against 380 s under load, so a serial rerun on an idle
machine is estimated at 1.5 to 2 h. Raw per-example test log-odds (11 MB) are kept outside
the repository in `ANCHORLOGIT_BENCH_RAW` (default `../_data/anchorlogit_bench/`).

## Main runs (log-loss; d = a6 - arm, SE)

| dataset, n | a1 none | a2 ref | a3 feat | a4 x100 | a5 k=1 | a6 kcv | d vs a3 | d vs a4 | d vs a5 |
|---|---|---|---|---|---|---|---|---|---|
| HIGGS 250 | .6839 | .6275 | .6494 | .6321 | .6300 | .6309 | -.0185 (.0034) | -.0012 (.0020) | +.0009 (.0006) |
| HIGGS 500 | .6722 | .6275 | .6357 | .6286 | .6287 | .6291 | -.0066 (.0029) | +.0005 (.0007) | +.0004 (.0004) |
| HIGGS 1000 | .6536 | .6275 | .6250 | .6278 | .6257 | .6259 | +.0009 (.0011) | -.0019 (.0006) | +.0002 (.0002) |
| HIGGS 2000 | .6461 | .6275 | .6182 | .6187 | .6189 | .6186 | +.0004 (.0003) | -.0002 (.0002) | -.0003 (.0006) |
| Bank 250 | .2920 | .1991 | .2629 | .2050 | .2051 | .2038 | -.0591 (.0033) | -.0012 (.0008) | -.0013 (.0009) |
| Bank 1000 | .2548 | .1991 | .2199 | .1984 | .2000 | .1986 | -.0213 (.0033) | +.0002 (.0005) | -.0014 (.0009) |
| Elec2 250 | 1.138 | .9089 | 1.193 | 1.247 | 1.345 | 1.218 | +.024 (.035) | -.029 (.035) | -.127 (.035) |
| Elec2 1000 | .6864 | .9089 | .6933 | .6940 | .8505 | .6926 | -.0007 (.0032) | -.0014 (.0031) | -.158 (.037) |

a3 (the reference as a standardised feature under the CV-chosen ridge) loses to the offset at
small n because the ridge shrinks the reference's coefficient; a4 (the same column x 100,
nearly unpenalised) does not have that problem and ties. Elec2 log-losses above ln 2 come from
a few drift weeks where every arm, including the reference, is confidently wrong (window 0
alone: plain ridge 1.43); the between-window SE reflects this.

## Variants (what would have to be true), all on held-out test rows

Premises for the offset to win: (a) the reference is stronger than any model on the features
alone, (b) the features hold noisy re-expressions of the reference, (c) n is small enough that
a free coefficient on the reference is noisy, (d) the reference is wrong in a subgroup the
features can detect.


1. **Smaller n + a stronger prior on kappa (`higgs_smalln_1se`, `_controls`).** Premise: the
   CV minimum over kappa is noisy at small n (kappa_cv lost to kappa = 1 at n = 100: +0.0044,
   SE 0.0023). Rule: among configurations within one fold-SE of the CV best, take the largest
   kappa, then the strongest ridge. Controls with the same C rule: kappa = 1 and the free
   column. Result (d = rule - arm):
   n = 100: vs a4 -0.0160 (0.0041), vs free_1se -0.0082 (0.0044), vs a6 -0.0118 (0.0031),
   vs ref alone +0.0104 (0.0040); n = 150: vs a4 -0.0090 (0.0034), vs free_1se -0.0064
   (0.0030), vs ref +0.0043 (0.0019); n = 250: vs a4 -0.0071 (0.0027), vs free_1se -0.0022
   (0.0018), vs ref +0.0027 (0.0011); n = 1000: vs a4 +0.0016 (0.0009), vs ref +0.0001
   (0.0002). The rule picks kappa = 1 in nearly every draw, so it equals the kappa = 1 control
   (d = 0.0000). **Kept** as a small-n setting (wins over the free column at n <= 150 on
   held-out rows); it is the classical offset with a conservative ridge, not a fractional one.
2. **Stronger reference that the features only re-express (`higgs_strongref`, reference on
   all 28 features).** Premises (a) + (b). Reference alone 0.5401; n = 100:
   a6 - a4 +0.0003 (0.0027), a6 - ref +0.0207 (0.0035); one-SE rule - a4 -0.0105 (0.0025),
   - ref +0.0099 (0.0041); n = 1000: every reference arm within 0.0003 of the reference.
   **Tie** with the free column for kappa_cv; nothing beats the reference alone.
3. **Reference wrong in a subgroup the features can detect (`bank_shiftref`, reference
   trained on clients under 40).** Premise (d). kappa_cv picks 0.75 in 10/10 draws at
   n = 1000 and beats kappa = 1 by -0.0077 (0.0012) and the reference by -0.0076 (0.0018),
   but ties the free column (-0.0004, 0.0003) and Platt on the reference (+0.0003, 0.0009);
   the one-SE rule (kappa near 1) loses to its free-column control by +0.0060 (0.0019).
   **Fraction confirmed useful against kappa = 1, tie against the controls.**

Dropped (not run): a finer kappa grid above 1 (kappa_cv never sat at an interior optimum
that the grid could miss on HIGGS / Bank, and above 1 is a calibration slope, which Platt
covers); ACS (cut above, never tested).

## Pre-registered checks

- a6 beats a1 at every n with the 90 % CI excluding 0 on >= 2 of 3 datasets: **met** (HIGGS,
  Bank; Elec2 fails).
- a6 or a7 vs a3: **tie at n >= 1000, win at n <= 500** (a3 is penalised); vs a4: tie.
- kappa_cv < 1 on ACS and Elec2: Elec2 **met** (median 0.25); ACS not run.
