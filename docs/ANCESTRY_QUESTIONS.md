# What this container is asking, and which file answers it

Two claims are being validated at the site. Each breaks into sub-questions, and each sub-question
has one experiment and one output file. This page is the map between them. It is also the
honest list of what a single site **cannot** answer, which matters as much: an underpowered null
read as a refutation is the most likely way this exercise goes wrong.

Read the table left to right: the question, the mode that answers it, the file to look at, and
what counts as a yes.

---

## 0. Before anything is interpreted: is the model working here?

Everything downstream is an analysis **of** the ancestry score. If the score is broken at this
site, the genetic arms return a confident null that looks exactly like "the signature does not
replicate". The things that break it are all silent — a different lead order, microvolts, a row
alignment recovered wrongly from the prepared arrays, the wrong five checkpoints — and none of
them raises an error.

| Question | Mode | File | A yes looks like |
|---|---|---|---|
| Does DeepECG-Pop still separate non-Europeans from Europeans here? | `score` | `score/model_check.csv` | clearly above chance, and in the band the reference tables record for the prior external run (the stage prints the comparison and its own verdict) |
| Does the AFR > EAS > SAS ordering hold? | `score` | `score/model_check.csv` | the three rows in the same order as the discovery reference, AFR > EAS > SAS |
| Did the whole run start from the right inputs? | `score` | `score/model_check_verdict.txt` | the one-line verdict says OK |
| Is the released 77-finding head's output order right here? | `fairness` | `fairness/auditability.json` → `output_mapping_verdict` | median per-finding AUROC near the discovery median |
| Are the genotypes on the build we think? | `survey`, then `geno` | `geno/variant_coverage.csv` | one build finds nearly every lead, the other almost none |

**If section 0 fails, stop.** Nothing below is interpretable and the fix is upstream.

---

## 1. The genetic signature

| Question | Mode | File | A yes looks like |
|---|---|---|---|
| Do the discovery loci replicate in Europeans? | `loci` | `loci/replication_summary.csv`, row `EUR` | high sign concordance, binomial *P* small, slope > 0 |
| Is the agreement in **effect size**, not just direction? | `loci` | same row: `slope`, `slope_lo/hi`, `pearson_r` | a slope whose interval excludes 0; see the note on winner's curse below |
| Could that concordance arise from the pipeline alone? | `loci` | `replication_summary.csv`, rows `sample = permuted phenotype` | ~50% concordance, *P* ≈ 0.5 — the real arm must stand well clear of it |
| Is it visible lead by lead? | `figures` | `fig_locus_replication.png` | points on the identity line, few at the floor |
| Does it hold outside Europeans? | `loci` | row `NON_EUR_META` | the fixed-effect meta of all non-European strata — see §5 |
| Is it just population structure / admixture? | `geno` + `loci` | `geno/geno_provenance.json` lists the covariates | the non-European and admixed strata are adjusted for the measured RFMix proportions as well as PCs (the admixture control) |
| Are there genome-wide signals at all here? | `scan` | `scan/<stratum>/scan_gwas_summary.json`, `fig_manhattan_*.png` | λ_GC near 1 with signal at the known leads |
| Is the test statistic inflated? | `scan` | `fig_qq.png`, `lambda_gc` | λ_GC ≈ 1.0–1.1 |

**On the slope.** The leads were *selected* at *P* < 5 × 10⁻⁸ in discovery, so their discovery
effects are upward-biased by that selection. A replication slope below 1 is expected even if
every effect is real. Read the slope's interval against 0, not against 1.

**On non-European strata.** The leads are European lead SNPs tagging causal variants through
European LD. Attenuation in an African-ancestry stratum is a statement about LD, not about
biology, and the container says so in `results_summary.md` as well as here.

---

## 2. Polygenic scores against the site's own ECGs

| Question | Mode | File | A yes looks like |
|---|---|---|---|
| Do the nine trait PGSs work at all here? (positive control) | `pgs` | `pgs/trait_pgs.csv`, `arm = own_measured_trait` | each PGS predicts its own measured trait |
| Does each trait PGS move the ancestry score? | `pgs` | `arm = ancestry_score` | positive β per SD |
| **Does that effect run through the measured trait?** | `pgs` | `arm = ancestry_score_adj_measured_trait`, `pct_attenuation` | a large attenuation means the score reads the conduction trait; a small one means the genotype indexes something the measurement misses |
| Does the decile fan reproduce Fig. 2e? | `pgs`, `figures` | `pgs/trait_pgs_deciles.csv`, `fig_pgs_deciles.png` | a monotone rise; panel b compares each span with ours |
| Does the frozen nine-trait composite reproduce? | `pgs` | `trait = COMPOSITE` | span comparable to discovery's +11.4 pp |
| Does a PGS **for the score itself** predict it within Europeans? | `pgs` | `pgs/ancestry_pgs.csv`, `arm = fixed_panel` | β ≈ +0.19 SD/SD, incremental *R*² of a few per cent |
| Does that survive the choice of *P* threshold and of clumping? | `pgs` | `arm = clumped` rows | a flat-ish ladder, not one spike |

### Every score here is applied, never refitted

**No weight is estimated at this site.** All of them are read from the frozen tables in `ref/` and
handed to `plink2 --score` unchanged, up to the sign of an allele flip where the site counts the
other allele. The only regression anywhere in this stage is *score → ancestry score, adjusted for
age, sex and PCs* — the score sits on the right-hand side as a fixed predictor. That regression
**is** the validation; it does not alter the score. The selftest asserts this directly rather than
claiming it: it reads back every weight file the stage wrote and checks each value against the
shipped table (`T3d-bis`, ~51,000 weights, zero permitted to differ).

Concretely:

| | variants | weights | selection done at the site |
|---|---|---|---|
| nine trait PGSs | all 615 published, minus any absent from the site's release (counted as `n_variants` vs `n_variants_published`) | published, unchanged | **none** |
| nine-trait composite | as above | frozen `composite_weights.tsv`, applied as a constant | **none** — it was fitted once, in discovery, and is applied here as a constant |
| ancestry-score PGS, **primary** (`arm = fixed_panel`) | every usable variant of the 21,055 | shipped meta betas, unchanged | **none** |
| ancestry-score PGS, sensitivity (`arm = clumped`) | C+T ladder | shipped meta betas, unchanged | clumping in the site's LD, *P* ladder |

**Why the headline is the fixed panel.** Clumping uses no phenotype, so it is not peeking — but it
does use the site's own LD, which means the surviving variant set differs from site to site and the
instrument is no longer the same object everywhere. The primary number is therefore the fixed
panel, which is identical at every site, and the clumped ladder is reported beside it so that the
discovery comparison is still available and any dependence on LD is visible rather than
baked in.

**Why the ancestry-score PGS is a stronger test here than in the paper.** In discovery every
participant is inside the meta-analysis, which forced leave-one-cohort-out weights. Your site is in
no arm of that meta, so the full three-cohort weights are shipped and used at full strength.

---

## 3. The fairness audit

| Question | Mode | File | A yes looks like |
|---|---|---|---|
| Does the 77-finding model fail unequally by **true** ancestry? | `fairness` | `fairness/gaps.csv`, `split = genetic` | non-zero FPR/FNR gaps with intervals excluding 0 |
| Can a hospital see the same thing **without genotyping**? | `fairness` | `split = inferred` | gaps of the same sign and size |
| Do the two agree well enough to be used that way? | `fairness` | `fairness/auditability.json`, `fig_fairness_auditability.png` | slope through the origin near 1, sign agreement well above half |
| Is the model's **ranking** also unequal, or only its errors at a shared cut? | `fairness` | `gaps.csv`, `metric = auroc` | read the pair: AUROC gaps null while threshold gaps are not |
| Does the result depend on the operating point? | `fairness` | `threshold_rule` = 0.5 and 90% specificity | both tell the same story |
| Is the gap just age or sex? | `fairness` | the design is 1:1 matched; `fairness/label_coverage.csv` | gaps persist inside matched pairs |
| Which findings can be audited here at all? | `fairness` | `fairness/label_coverage.csv` | ≥ 15 positives in both groups |
| Was the audit run on everyone, or a subsample? | `fairness` | `fairness/auditability.json` → `coverage` | two arms: `unmatched` is every scored person with an unambiguous ancestry call, `matched` is the 1:1 subset of it |

---

## 3b. Socioeconomic deprivation (only with `--ses`)

The discovery work asked this twice, with two different kinds of variable: an **area-level**
census deprivation index, and an **individual** household-income bracket. The two behaved
differently, and the reading was that what looked socioeconomic was a *neighbourhood* effect
rather than a property of the patient. The numbers behind that are in the manuscript, not in
this repository.

SDI is a neighbourhood measure, so this arm replicates the harder case, in a country where
census-tract deprivation tracks ancestry far more tightly than Townsend does in Britain. That is
the point of running it, and it is also the one way it can mislead.

| Question | Mode | File | A yes looks like |
|---|---|---|---|
| **How far is deprivation from being an ancestry proxy here?** | `ses` | `ses/ses_coupling.csv` | read this first. An AUROC near 0.5 means the arms below are clean; near 0.8 means adjusting for deprivation partly adjusts for ancestry |
| Can the ECG see deprivation at all? (arm A) | `ses` | `ses_quad.csv`, `arm = A_visibility` | the *within-EUR* row is the honest one; the pooled row can reach its value by reading ancestry. discovery found this near chance |
| Does ancestry survive matching on it? (arm B) | `ses` | `arm = B_matched` | **the arm that decides.** The cost of matching, against a sham carrying deprivation's own ancestry correlation |
| Is that cost just the ancestry correlation? | `ses` | `B_matched`, `coupling-matched sham` | if the real cost sits inside the sham distribution, nothing is left for socioeconomics |
| Is it just that matching shrank the sample? | `ses` | `B_matched`, `random same-size subset` | the band a match of that size costs with deprivation ignored |
| Does ancestry work in both halves? (arm C) | `ses` | `arm = C_halves` | expected null **whatever the truth** — see below |
| Does removing its direction cost anything? (arm D) | `ses` | `arm = D_erasure` or `D_adjust_score` | always read against its coupling-matched sham, never against zero |
| Did that erasure remove anything? | `ses` | `arm = D_potency` | the deprivation AUROC must fall to ~0.5 after the projection, or arm D's null is uninformative (a control the first version of this audit lacked) |
| Can the machinery move the number at all? | `ses` | `arm = D_control` | erasing age, which the score does track, should visibly move it |
| Does any of it depend on where the cut is? | `ses` | `ses_cut_grid.csv`, `fig_ses.png` panel d | a flat sweep |

**What each arm is worth, measured rather than asserted.** Each arm was simulated under a *true*
null at a cohort of roughly this size, and what came back decides which one you should report:

- **arm B** is the arm that bounds the socioeconomic contribution, and at this sample size the
  bound it achieves clears the 0.02 margin. **This is the arm to report.**
- **arm C** is unbiased but noisy — its sampling spread is wide enough that it exceeds the margin
  in a sizeable fraction of null runs, so a small difference there means nothing. Reported for
  comparability only.
- **arm D** at the score level returns a non-zero cost *even when the simulation contains no
  socioeconomic path at all* — pure over-adjustment — and its coupling-matched sham reproduces
  that cost. Read the pair, never the number.

**Every arm reports a bound, not a zero.** `achieved_bound` on each cost row is the smallest
margin that sample could have cleared. A failure to clear 0.02 usually means the stratum was
small, not that deprivation did something.

**And it is an ecological proxy.** An area index measures the neighbourhood, not the patient; its
measurement error pushes every arm toward the null, so a null here is weaker evidence than a null
from an individual measure. The honest pairing is: area-level, maximally ancestry-confounded,
still bounded *here* — and individual-level, where the discovery work found no imbalance to
begin with.

---

## 4. The cohort itself

| Question | Mode | File |
|---|---|---|
| How many people in each ancestry stratum, and which clear the minimum? | `prepare` | `cohort/strata.csv` |
| Is the ancestry score distributed as it was in October? | `score` | `score/score_by_stratum.csv`, `score/score_histogram.csv` |
| How many of the shipped variants exist in this release? | `geno` | `geno/variant_coverage.csv` |
| Who entered each analysis, after kinship and QC? | `geno` | `geno/analysis_sets.csv` |

---

## 5. What one site cannot answer, and why

These are listed so that a null here is not reported as a negative result.

**The non-European strata are small.** At the first external site, the people with both genotypes and a
linked ECG left the non-European strata very much smaller than the European one. A lead-level replication in a few hundred
people has essentially no power, and a genome-wide scan there is descriptive only. This is why:

- every stratum reports `expected_n_bonferroni` — the number of leads this *n* could have been
  expected to confirm — beside the number it actually confirmed;
- the four non-European strata are additionally combined into one `NON_EUR_META` row, which is the
  only non-European statement this design can support (~2,600 people);
- a stratum below 100 people is not analysed at all, and says so.

**Locus class.** Most of the leads are already-published ECG-trait loci, a few are cardiac-disease loci and one has no
cardiac annotation. Asking whether the published ones replicate better is a 58-vs-4 comparison and
is not powered anywhere, so no class-stratified statistic is reported. The figure colours the
points by class so the pattern can be seen; it is not tested.

**Heritability.** LDSC is not in this image. It needs reference LD scores matched to each
stratum's ancestry, and a stratified h² at this sample size would have an interval wide enough to
include most values of interest. The discovery h² and LDSC intercept are in
`ref/discovery_reference.json` for context only.

**Individual socioeconomic position.** Nothing in this container measures a patient's own income,
education or occupation. The deprivation arms use the neighbourhood as a proxy, which is the
weaker instrument and the one the discovery work showed carried an imbalance that the individual
measure did not. A site that can attach individual-level socioeconomic data should say so — it
would be the stronger test and this image cannot do it.

**Disease confounding of the fairness gap.** Whether an unequal error rate is explained by unequal
disease prevalence needs the site's full diagnostic history and is out of scope here. The matched
design controls age and sex only.

**Local ancestry.** The admixed stratum is adjusted for *global* RFMix proportions, not local
ancestry at each locus. In discovery the per-locus natural experiment was badly underpowered even in a far
larger sample, so it is not attempted at one hospital.

---

## 6. Where the privacy line is

Every question above is answered by an aggregate. Nothing per-person is exported, and this is
enforced at three points rather than trusted:

1. **At write time.** Every exported table goes through `privacy.write_table`, which refuses a
   frame carrying an identifier column, bounds its row count, and masks counts of 1–9.
2. **By minimum group size.** No stratum below 100 people is analysed; no metric is computed for a
   group below 20 per class; no finding with fewer than 15 positives in *both* ancestry groups
   enters the audit; no per-variant statistic is exported from a stratum below 100.
3. **At packaging.** `package` re-audits every file — including tab-separated ones and figures —
   against the site's own identifier list, and refuses to write the zip if anything trips. If the
   roster is absent the identifier search cannot run, and the package says so rather than
   claiming a clean audit.

The genome-wide summary statistics never leave: `scan` exports a thinned point cloud (positions
and −log₁₀ *P*) and the QQ curve, and keeps the full statistics in `work/`.

The `survey` report is the one file sent back without passing through `package`, so it is masked
and audited by `survey` itself before it is written.
