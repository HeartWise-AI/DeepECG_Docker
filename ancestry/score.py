"""Mode `score` -- run the two heads over the site's ECGs and build the GWAS phenotype.

TWO MODELS, ONE PASS EACH, AND THEY ARE NOT THE SAME MODEL:

  ancestry   DeepECG-Pop: the five fine-tuned ancestry folds of this paper on the shared SSL
             backbone, averaged.  Output is P(non-European) per recording.  This is the thing the
             GWAS is a GWAS OF, and the thing whose 0.5 call splits the fairness audit's
             "inferred" arm.  Identical weights, identical code and identical units to the image
             MGB-MGH ran in October, so the score distribution here should reproduce that one;
             the package reports both so a reader can check.

  findings   DeepECG-SSL: the publicly released 77-finding diagnostic head on the same backbone.
             Output is 77 sigmoid probabilities per recording.  This is the model being AUDITED.
             It is a different head with a different training set, and nothing in this image
             fine-tunes, calibrates or thresholds it -- it is run exactly as released, at its
             released default operating point of 0.5.

UNITS.  Both heads were trained on millivolts, and `prepare` writes millivolts, so neither is
rescaled here.  infer_lib's unit guard runs on the ancestry pass and will stop a run whose
waveforms are microvolts or raw ADC counts, which is the one input error that produces plausible
numbers rather than an error.

THE PHENOTYPE.  P(non-European) -> logit -> rank-inverse-normal WITHIN STRATUM.  Within stratum
because the strata are analysed separately and a score compressed against 1.0 in an
African-ancestry stratum must not be put on the same scale as a spread-out European one by
accident; after RINT a beta is in SD of the score per allele copy everywhere, so the strata are
comparable.

    python -m ancestry score --work /out/work --results /out/results [--head both]
                      [--device auto] [--batch-size 64] [--shard 1/8] [--merge 'work/score_*.csv']

Writes work/scores.csv and work/findings.parquet (BOTH CARRY IDS, both stay at the site) and
results/score/*.csv (distributions by stratum, masked).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from ancestry import privacy as PV                                       # noqa: E402
from ancestry import lib as L                                            # noqa: E402
from ancestry import weights as WT                                       # noqa: E402

BINS = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]


def load_roster(work):
    r = pd.read_csv(Path(work) / 'roster.csv', dtype={'participant_id': str, 'geno_id': str})
    return r


def prepared_dir(work, override=None):
    if override:
        return Path(override)
    f = Path(work) / 'prepared_dir.txt'
    if not f.exists():
        raise SystemExit('run `prepare` first (no work/prepared_dir.txt)')
    return Path(f.read_text().strip())


def shard_slice(n, spec):
    if not spec:
        return 0, n
    i, k = (int(x) for x in spec.split('/'))
    edges = np.linspace(0, n, k + 1).astype(int)
    return int(edges[i - 1]), int(edges[i])


# ------------------------------------------------------------------------------- the two heads
def _embed_one(model, X_nct, device, batch_size):
    """The pooled representation one fold's model computes, per ECG.

    Captured through infer_lib's OWN `return_embedding` path -- a forward pre-hook on
    `model.proj` -- rather than by editing infer_lib.py, which is a byte-identical copy of the
    deployed image and has to stay that way for the ancestry scores to be reproducible to the
    bit. This function lives here because it is new work, not vendored work.
    """
    import torch
    N = X_nct.shape[0]
    out = None
    with torch.no_grad():
        for s in range(0, N, batch_size):
            e = min(N, s + batch_size)
            xb = torch.from_numpy(np.ascontiguousarray(X_nct[s:e], dtype=np.float32)).to(device)
            _logits, emb = model(xb, return_embedding=True)
            if emb is None:
                raise SystemExit('the model did not expose an embedding: this checkpoint does '
                                 'not have the `proj` layer the hook attaches to, so '
                                 '--save-embeddings cannot work here')
            z = emb.detach().float().cpu().numpy()
            if z.ndim == 3:            # (B, T, D) -- pool over time, as the head itself does
                z = z.mean(axis=1)
            z = z.reshape(e - s, -1)
            if out is None:
                out = np.empty((N, z.shape[1]), dtype=np.float32)
            out[s:e] = z
    return out


def embeddings(models, X, dev, batch_size):
    """(N, n_folds, D). One block per fold, NOT averaged across folds.

    The five folds are separately fine-tuned models, so their feature spaces are only
    approximately aligned and a mean over them is not a representation of anything. Every arm
    that uses these runs per fold and aggregates the STATISTIC, as the discovery analysis did.
    """
    Xn = np.asarray(X)
    Xn = Xn if (Xn.shape[1] == 12 and Xn.shape[2] != 12) else np.swapaxes(Xn, -1, -2)
    blocks = []
    for k, m in enumerate(models, 1):
        print(f'[embed] fold {k}/{len(models)} on {Xn.shape[0]:,} ECGs', flush=True)
        blocks.append(_embed_one(m, Xn, dev, batch_size))
    return np.stack(blocks, axis=1)


def score_ancestry(X, args):
    from ancestry import infer as IL
    dev = IL.pick_device(args.device)
    models, ckpts = IL.load_fold_models(WT.resolve('ancestry_folds', args.models_root),
                                        WT.resolve('ssl_backbone', args.ssl_ckpt),
                                        dev, n_folds=5)
    p, folds = IL.ensemble_predict(models, X, dev, batch_size=args.batch_size)
    E = None
    if getattr(args, 'save_embeddings', False):
        E = embeddings(models, X, dev, args.batch_size)
    return p, folds, ckpts, E


def score_findings(X, args):
    """The released 77-finding head, run as released.

    Loaded through the same fairseq-signals checkpoint loader the ancestry heads use, with the SSL
    backbone passed as `model_path` -- the released checkpoint stores the head and expects the
    backbone to be supplied, which is why a bare torch.load of it does not give a usable model.
    """
    import torch
    from ancestry import infer as IL
    dev = IL.pick_device(args.device)
    head = WT.resolve('findings_head', args.wcr_ckpt)
    print(f'[load] findings head: {head}', flush=True)
    m = IL.WCREcgTransformer(str(head), str(WT.resolve('ssl_backbone', args.ssl_ckpt))
                             ).to(dev).eval()
    Xn = np.asarray(X)
    Xn = Xn if (Xn.shape[1] == 12 and Xn.shape[2] != 12) else np.swapaxes(Xn, -1, -2)
    n = len(Xn)
    out = None
    with torch.no_grad():
        for s in range(0, n, args.batch_size):
            e = min(n, s + args.batch_size)
            xb = torch.from_numpy(np.ascontiguousarray(Xn[s:e], dtype=np.float32)).to(dev)
            z = m(xb)
            q = torch.sigmoid(z).float().cpu().numpy().reshape(e - s, -1)
            if out is None:
                out = np.empty((n, q.shape[1]), dtype=np.float32)
                print(f'[infer] findings head emits {q.shape[1]} outputs', flush=True)
            out[s:e] = q
            if n >= 2000 and (e // args.batch_size) % 50 == 0:
                print(f'    {e:,}/{n:,} ECGs', flush=True)
    return out


# ------------------------------------------------------------------------------- the gate
# THE SINGLE MOST IMPORTANT TABLE IN THE PACKAGE, and the one to read first.
#
# Everything downstream -- the discovery leads, the polygenic scores, the "inferred ancestry" arm of the
# fairness audit -- is an analysis OF this score.  If the score is not working at this site the
# genetic arms return a confident null and that null looks exactly like "the signature does not
# replicate", which is the wrong conclusion drawn from a broken input.  The things that break it
# are all silent: a lead order that differs from ours, microvolts, a row alignment recovered
# wrongly from the prepared arrays, the wrong five checkpoints.  Each of them lands the AUROC at
# chance; none of them raises.
#
# So before anything is interpreted, the same five checkpoints are scored against the site's own
# RFMix calls, and compared with two numbers fixed before this container was built and read from
# the frozen reference tables: what the earlier external run at another site got, and what the
# three discovery cohorts got per subpopulation.
def model_check(D, res, ref):
    """AUROC of P(non-European) against RFMix truth, overall and per subpopulation."""
    R = json.loads((Path(ref) / 'discovery_reference.json').read_text())
    prior = next(iter(R.get('external_prior', {}).values()), {})
    sub = R.get('subpopulation_auroc_discovery', {})

    E = D[D.stratum == 'EUR']
    rows = []

    def arm(name, pos, ref_auroc, ref_note):
        d = pd.concat([E, pos])
        y = (d.stratum != 'EUR').to_numpy().astype(float)
        if len(pos) < PV.MIN_PER_CLASS or len(E) < PV.MIN_PER_CLASS:
            rows.append(dict(comparison=name, n=len(d), n_pos=len(pos), auroc=np.nan,
                             lo=np.nan, hi=np.nan, reference=ref_auroc, note=PV.SUPPRESSED))
            return
        pr = d.p_nonEUR.to_numpy(float)
        a = L.auroc(y, pr)
        lo, hi = L.boot_ci(lambda i: L.auroc(y[i], pr[i]), len(d), 1000, 2000)
        rows.append(dict(comparison=name, n=len(d), n_pos=len(pos), auroc=round(a, 4),
                         lo=round(lo, 4), hi=round(hi, 4), reference=ref_auroc, note=ref_note))

    nonE = D[D.stratum.isin([s for s in L.STRATA if s != 'EUR' and s != 'ADMIXED'])]
    arm('non-European vs European', nonE,
        (sub.get('all non-EUR') or {}).get('auroc'), 'discovery reference, three cohorts')
    for a3 in ('AFR', 'EAS', 'SAS'):
        arm(f'{a3} vs European', D[D.stratum == a3],
            (sub.get(a3) or {}).get('auroc'), 'discovery reference, three cohorts')
    arm('admixed vs European', D[D.stratum == 'ADMIXED'], None,
        'no discovery reference: the discovery phenotype is European-only')

    # the earlier run recorded for this site in the reference tables, which used every
    # genotyped person with an ECG
    allp = D[D.stratum.isin(L.STRATA)]
    y = (allp.stratum != 'EUR').to_numpy().astype(float)
    if len(np.unique(y)) == 2:
        pr = allp.p_nonEUR.to_numpy(float)
        a = L.auroc(y, pr)
        lo, hi = L.boot_ci(lambda i: L.auroc(y[i], pr[i]), len(allp), 1000, 2000)
        ece = L.ece(y, pr)
        rows.append(dict(comparison='everyone (admixed counted non-European)', n=len(allp),
                         n_pos=int(y.sum()), auroc=round(a, 4), lo=round(lo, 4), hi=round(hi, 4),
                         reference=prior.get('auroc'),
                         note=f'this site, earlier run (ECE then {prior.get("ece")}; '
                              f'now {ece:.4f})'))
    T = pd.DataFrame(rows)
    PV.write_table(T, res / 'score' / 'model_check.csv', where='model_check.csv')

    head = T[T.comparison.str.startswith('non-European')].auroc
    val = float(head.iloc[0]) if len(head) and np.isfinite(head.iloc[0]) else np.nan
    print('\n' + T.to_string(index=False))
    # The WEAK band is the REFERENCE range itself, read from the frozen tables at run time --
    # this repository carries no measured AUROC of ours, so the band cannot be read off the
    # source. Only the chance floor is written here, and a floor at chance is not a result.
    refv = (sub.get('all non-EUR') or {}).get('auroc') or prior.get('auroc')
    weak = 0.95 * float(refv) if refv else None
    if not np.isfinite(val):
        verdict = 'UNKNOWN -- too few non-Europeans to check the model at all'
    elif val < 0.65:
        verdict = ('FAIL -- the ancestry model is at or near chance here. Do NOT interpret the '
                   'genetic or inferred-ancestry results: check the lead order, the units and '
                   'the prepared-array alignment first (survey, then prepare_provenance.json).')
    elif weak is not None and val < weak:
        verdict = ('WEAK -- below both the discovery range and the earlier run recorded for '
                   'this site. The downstream nulls may be the input, not the biology.')
    elif weak is None:
        verdict = ('OK -- the score separates the groups. No reference range is present in the '
                   'frozen tables, so there is nothing to compare it with here.')
    else:
        verdict = 'OK -- the score reproduces; the genetic and fairness arms are interpretable.'
    print(f'\n  GATE: {verdict}')
    return T, verdict


# ------------------------------------------------------------------------------- phenotype
def build_phenotype(S, roster):
    """logit(P(non-EUR)) -> RINT within stratum.  Returns the roster with the phenotype columns."""
    D = roster.merge(S, on='participant_id', how='inner')
    D['score_logit'] = L.logit(D.p_nonEUR.to_numpy())
    D['pheno_rint'] = np.nan
    for s, g in D.groupby('stratum'):
        if len(g) < PV.MIN_STRATUM:
            continue
        D.loc[g.index, 'pheno_rint'] = L.rint(g.score_logit.to_numpy())
    return D


def distributions(D, res):
    """What leaves: per-stratum summary and a fixed-bin histogram.  No thresholds, no ids."""
    rows, hist = [], []
    for s in L.STRATA:
        g = D[D.stratum == s]
        if len(g) < PV.MIN_STRATUM:
            rows.append(dict(stratum=s, n=len(g), note=PV.SUPPRESSED))
            continue
        p = g.p_nonEUR.to_numpy()
        rows.append(dict(stratum=s, n=len(g), mean_p=round(float(p.mean()), 4),
                         median_p=round(float(np.median(p)), 4),
                         q25=round(float(np.percentile(p, 25)), 4),
                         q75=round(float(np.percentile(p, 75)), 4),
                         pct_called_nonEUR=round(100 * float((p >= 0.5).mean()), 2),
                         mean_rf_nonEUR=round(float(g.rf_nonEUR.mean()), 4), note=''))
        c, _ = np.histogram(p, bins=BINS)
        for lo, hi, k in zip(BINS[:-1], BINS[1:], c):
            hist.append(dict(stratum=s, bin_lo=lo, bin_hi=hi, n=int(k)))
    PV.write_table(pd.DataFrame(rows), res / 'score' / 'score_by_stratum.csv',
                   where='score_by_stratum.csv')
    PV.write_table(pd.DataFrame(hist), res / 'score' / 'score_histogram.csv',
                   where='score_histogram.csv')
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(prog='python -m ancestry score', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', required=True,
                    help="the site's working directory: intermediates, per-person tables and "
                         'arrays. It stays at the site and is never packaged')
    ap.add_argument('--results', required=True,
                    help='the aggregate-only output directory; `package` zips exactly this')
    ap.add_argument('--prepared', default=None,
                    help='the prepared/ directory (default: the one `prepare` recorded in work/)')
    ap.add_argument('--head', default='both', choices=['ancestry', 'findings', 'both'],
                    help='which model head to run: the five ancestry folds, the 77-finding '
                         'head, or both (default). One pass each, over the same arrays')
    # Left unset, each is resolved by ancestry.weights: $MODELS_ROOT / $SSL_CKPT / $WCR_CKPT
    # as the image sets them, else the pinned revisions in the model manifests.
    ap.add_argument('--models-root', default=None,
                    help='the five ancestry fold checkpoints (default: $MODELS_ROOT, else the '
                         'pinned revision in ancestry/model_manifest.json)')
    ap.add_argument('--ssl-ckpt', default=None,
                    help='the shared SSL backbone (default: $SSL_CKPT, else the pinned snapshot)')
    ap.add_argument('--wcr-ckpt', default=None,
                    help='the 77-finding head (default: $WCR_CKPT, else the pinned snapshot)')
    ap.add_argument('--device', default='auto',
                    help="cpu, cuda, cuda:1 ...; 'auto' takes a GPU when there is one")
    ap.add_argument('--ref', default=str(L.ref_dir()),
                    help='the frozen discovery tables (default: $REFERENCE_DIR, else the '
                         'reference/ folder of the pinned weights snapshot)')
    ap.add_argument('--batch-size', type=int, default=64,
                    help='ECGs per forward pass; lower it if the GPU runs out of memory')
    ap.add_argument('--shard', default=None, metavar='I/N',
                    help='score only slice I of N (for CPU fan-out); merge afterwards')
    ap.add_argument('--merge', nargs='+', default=None,
                    help='merge shard outputs written earlier and build the phenotype')
    ap.add_argument('--save-embeddings', action='store_true',
                    help="also write each fold's penultimate representation to "
                         'work/embeddings.npy (N x 5 x D, ~75 MB per 5,000 people). Needed only '
                         'by `ses` arms A and D; off by default because it costs a second '
                         'forward pass and nothing else in the image reads it')
    a = ap.parse_args()

    work = Path(a.work); res = Path(a.results)
    L.require('score', [(work / 'roster.csv',
                         'the roster: who is in which ancestry stratum, and which ECG row',
                         'prepare')])
    (res / 'score').mkdir(parents=True, exist_ok=True)
    roster = load_roster(work)

    # ------------------------------------------------------------------ merge mode
    if a.merge:
        files = sorted(sum([glob.glob(p) for p in a.merge], []))
        if not files:
            raise SystemExit(f'--merge matched nothing: {a.merge}')
        S = pd.concat([pd.read_csv(f, dtype={'participant_id': str}) for f in files],
                      ignore_index=True).drop_duplicates('participant_id')
        print(f'  merged {len(files)} shards -> {len(S):,} people')
        # if the shards saved embeddings, concatenate those in the same order
        eids = sorted(glob.glob(str(work / 'embeddings_*of*_ids.csv')))
        if eids:
            Es, Is = [], []
            for f in eids:
                npy = Path(f.replace('_ids.csv', '.npy'))
                if npy.exists():
                    Es.append(np.load(npy))
                    Is.append(pd.read_csv(f, dtype={'participant_id': str}))
            if Es:
                np.save(work / 'embeddings.npy', np.concatenate(Es, axis=0))
                pd.concat(Is, ignore_index=True).to_csv(work / 'embeddings_ids.csv', index=False)
                print(f'  merged {len(Es)} embedding shards -> '
                      f'{sum(len(e) for e in Es):,} people')
    else:
        pdir = prepared_dir(work, a.prepared) / 'all_genotyped'
        X = np.load(pdir / 'X.npy', mmap_mode='r')
        if len(X) != len(roster):
            raise SystemExit(f'X.npy has {len(X):,} rows, roster has {len(roster):,}: '
                             're-run prepare')
        lo, hi = shard_slice(len(X), a.shard)
        sl = roster.iloc[lo:hi].reset_index(drop=True)
        L.banner(f'score: rows {lo:,}-{hi:,} of {len(X):,}  (head={a.head})')
        Xs = np.asarray(X[lo:hi])

        S = pd.DataFrame({'participant_id': sl.participant_id.to_numpy()})
        prov = {}
        if a.head in ('ancestry', 'both'):
            p, folds, ckpts, E = score_ancestry(Xs, a)
            S['p_nonEUR'] = p
            for k in range(folds.shape[1]):
                S[f'p_fold{k + 1}'] = folds[:, k]
            prov['ancestry_checkpoints'] = ckpts
            if E is not None:
                tag = f'_{a.shard.replace("/", "of")}' if a.shard else ''
                np.save(work / f'embeddings{tag}.npy', E)
                pd.DataFrame({'participant_id': sl.participant_id.to_numpy()}).to_csv(
                    work / f'embeddings{tag}_ids.csv', index=False)
                prov['embeddings'] = list(E.shape)
                print(f'  wrote {work / f"embeddings{tag}.npy"}  {E.shape} '
                      f'({E.nbytes / 1e6:.0f} MB)   CARRIES IDS BY ROW, STAYS HERE')
        if a.head in ('findings', 'both'):
            W = score_findings(Xs, a)
            F = pd.DataFrame(W, columns=[f'out{i}' for i in range(W.shape[1])])
            F.insert(0, 'participant_id', sl.participant_id.to_numpy())
            name = f'findings{"_" + a.shard.replace("/", "of") if a.shard else ""}.parquet'
            F.to_parquet(work / name, index=False)
            print(f'  wrote {work / name}  (CARRIES IDS, stays here)')
            prov['findings_checkpoint'] = str(WT.resolve('findings_head', a.wcr_ckpt))
            prov['findings_n_outputs'] = int(W.shape[1])
        if a.shard:
            f = work / f'scores_{a.shard.replace("/", "of")}.csv'
            S.to_csv(f, index=False)
            print(f'  wrote {f}  (shard; merge with --merge)')
            return
        (work / 'score_provenance.json').write_text(json.dumps(prov, indent=2))

    if 'p_nonEUR' not in S:
        print('  findings-only pass: no phenotype to build')
        return

    D = build_phenotype(S, roster)
    D.to_csv(work / 'scores.csv', index=False)
    print(f'\n  wrote {work / "scores.csv"}  ({len(D):,} people)   CARRIES IDS, STAYS HERE')
    out = distributions(D, res)
    print('\n' + out.to_string(index=False))
    MC, verdict = model_check(D, res, a.ref)
    (res / 'score' / 'model_check_verdict.txt').write_text(verdict + '\n')
    n_pheno = int(D.pheno_rint.notna().sum())
    print(f'\n  phenotype built for {n_pheno:,} people in '
          f'{D.loc[D.pheno_rint.notna(), "stratum"].nunique()} strata')


if __name__ == '__main__':
    main()
