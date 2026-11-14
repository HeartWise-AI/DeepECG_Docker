"""Mode `survey` -- read-only: what is on this machine, and will the run work?

Run this before anything else and send back the output.  It opens no model, writes nothing outside
its own report, and prints no identifier -- only shapes, column names, counts and whether a path
exists.  Its job is to answer, in one page, the four questions that otherwise cost a week of email:

  1. Are the ECGs where we think, and are they millivolts?  (A microvolt array scores like noise
     and raises nothing; the unit guard is the only thing between that and a plausible-looking
     result.)
  2. Do the genotypes exist, for which chromosomes, in which format, and on which BUILD?  The
     build is inferred by looking up the discovery leads in both coordinate systems and seeing
     which one matches -- a GRCh38 position in a GRCh37 release matches almost nothing, and that
     is the single commonest way one of these runs produces a confident null.
  3. How many people fall in each ancestry stratum, and which of them clear the minimum?
  4. Is there ground truth for the fairness audit, and how many of the 77 findings does it cover?

    python -m ancestry survey --rfmix RF.txt [...] --geno '/path/chr@' [--prepared DIR] [--labels F.csv]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from ancestry import lib as L                                            # noqa: E402

OK, NO, HM = '  [ok] ', '  [--] ', '  [??] '


def main():
    ap = argparse.ArgumentParser(prog='python -m ancestry survey', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rfmix', nargs='*', default=[],
                    help='the RFMix table(s) you intend to pass to `prepare`')
    ap.add_argument('--ecg-meta', nargs='*', default=[],
                    help='the ECG inventory you intend to pass to `prepare`')
    ap.add_argument('--geno', default=None,
                    help='the genotypes: plink2 pgen/bed prefix, bgen, or VCF/BCF, with @ for '
                         'the chromosome number')
    ap.add_argument('--prepared', default=None,
                    help='the prepared/ directory (default: the one `prepare` recorded in work/)')
    ap.add_argument('--labels', default=None,
                    help='adjudicated 0/1 labels for the 77 findings, one row per person '
                         '(yes/no, Y/N, true/false are all read)')
    ap.add_argument('--statements', default=None,
                    help='free-text ECG statements instead of labels; mapped to the 77 '
                         'findings by the shipped rules')
    ap.add_argument('--ses', default=None, help='area-level deprivation, to check it is usable')
    ap.add_argument('--ses-col', default=None,
                    help='the deprivation column of --ses (an index or a household income)')
    ap.add_argument('--ref', default=str(L.ref_dir()),
                    help='the frozen discovery tables (default: $REFERENCE_DIR, else the '
                         'reference/ folder of the pinned weights snapshot)')
    ap.add_argument('--out', default=None,
                    help='where to write (default: beside --results)')
    ap.add_argument('--work', default=None,
                    help='where the run will write, so its free space is checked')
    ap.add_argument('--sample-col', default='Sample', help='sample-id column of the RFMix table')
    # The same four column names `prepare` takes, so the check here is the check that will run.
    ap.add_argument('--ecg-id-col', default='ecg_id', help='recording-id column of --ecg-meta')
    ap.add_argument('--gender-col', default='gender', help='sex column of --ecg-meta')
    ap.add_argument('--age-col', default='age', help='age-at-recording column')
    ap.add_argument('--path-col', default='full_ecg_path', help='waveform-path column')
    ap.add_argument('--ancestries', nargs='+', default=['AFR', 'EAS', 'SAS', 'EUR'],
                    help='the RFMix columns to read')
    a = ap.parse_args()
    ref = Path(a.ref)
    rep = {'version': L.VERSION}

    L.banner('survey (read-only)')
    print('tools')
    for n, p in (('plink2', shutil.which('plink2') or '/opt/bin/plink2'),
                 ('plink1.9', shutil.which('plink') or '/opt/bin/plink')):
        print((OK if Path(p).exists() else NO) + f'{n}: {p}')
        rep[n] = Path(p).exists()
    try:
        import torch
        print(OK + f'torch {torch.__version__}, cuda '
              f'{"yes: " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no"}')
        rep['cuda'] = bool(torch.cuda.is_available())
    except Exception as e:
        print(NO + f'torch not importable: {e}')

    from ancestry import submit as SL
    rep['slurm'] = {k: v for k, v in SL.report(SL.detect()).items()
                    if k in ('sbatch', 'inside_job', 'in_container', 'can_submit',
                             'gpu_partitions')}

    print('\nmodels')
    from ancestry import weights as W
    for n, artefact in (('SSL backbone', 'ssl_backbone'),
                        ('77-finding head', 'findings_head'),
                        ('ancestry folds', 'ancestry_folds')):
        try:
            q = Path(W.resolve(artefact, download=False))
        except W.WeightsUnavailable as e:
            print(NO + f'{n}: {e}')
            rep.setdefault('models', {})[n] = False
            continue
        sz = (sum(f.stat().st_size for f in q.rglob('*') if f.is_file()) if q.is_dir()
              else q.stat().st_size) if q.exists() else 0
        print((OK if sz else NO) + f'{n}: {q} ({sz / 1e6:,.0f} MB)')
        rep.setdefault('models', {})[n] = bool(sz)

    print('\ndisk')
    # the scan is the only step with a large footprint: per-chromosome summary statistics for
    # every stratum, written to --work and never exported.  ~2 GB per stratum is typical.
    for label, where in (('--work (or the current directory)', a.work or '.'),):
        try:
            du = shutil.disk_usage(where)
            free = du.free / 1e9
            print((OK if free >= 50 else HM) + f'{label}: {free:,.0f} GB free'
                  + ('' if free >= 50 else '  (a genome-wide scan wants ~50 GB; the lead,'
                                           ' PGS and fairness arms want ~5)'))
            rep['free_gb'] = round(free, 1)
        except OSError as e:
            print(HM + f'{label}: could not be measured ({e})')

    print('\nreference tables')
    for f in sorted(ref.glob('*.tsv')):
        print(OK + f'{f.name}: {sum(1 for _ in open(f)) - 1:,} rows')

    print('\nECGs')
    if not a.prepared and not a.ecg_meta:
        print(HM + 'not checked: pass --prepared <dir> and/or --ecg-meta <file ...>')
    if a.prepared:
        X = Path(a.prepared) / 'all_genotyped' / 'X.npy'
        if X.exists():
            A = np.load(X, mmap_mode='r')
            s = np.asarray(A[:min(len(A), 128)], dtype=np.float32)
            peak = float(np.median(np.abs(s).max(axis=(1, 2))))
            print(OK + f'{X}: {A.shape}')
            print((OK if 0.2 <= peak <= 20 else NO) +
                  f'median per-ECG peak |amplitude| {peak:.3g} '
                  f'({"millivolts, as the models expect" if 0.2 <= peak <= 20 else "NOT millivolts -- convert before scoring"})')
            rep['n_ecg'] = int(len(A)); rep['peak_mv'] = peak
        else:
            print(NO + f'no {X}')
    for f in a.ecg_meta:
        try:
            with open(f, 'rb') as fh:
                is_pq = fh.read(4) == b'PAR1'
            if is_pq:
                import pyarrow.parquet as pq
                cols = list(pq.read_schema(f).names)
            else:
                cols = list(L.read_table(f, what='--ecg-meta', nrows=5).columns)
            kind = 'parquet' if is_pq else 'delimited text'
            print(OK + f'{Path(f).name}: {len(cols)} columns ({kind})')
            for want, flag in ((a.ecg_id_col, '--ecg-id-col'), (a.gender_col, '--gender-col'),
                               (a.age_col, '--age-col'), (a.path_col, '--path-col')):
                if want and want not in cols and want.lower() not in [c.lower() for c in cols]:
                    print(NO + f'  {want!r} is not a column here -- pass {flag} <name>')
        except SystemExit as e:
            print(NO + f'{f}: {e}')
        except Exception as e:                                           # noqa: BLE001
            print(NO + f'{f}: {type(e).__name__}: {e}')

    print('\nancestry')
    if not a.rfmix:
        print(HM + 'not checked: pass --rfmix <file ...>')
    if a.rfmix:
        frames = []
        for f in a.rfmix:
            d = L.read_table(f, what='--rfmix', nrows=5)
            low = {c.lower(): c for c in d.columns}
            sc = a.sample_col if a.sample_col in d.columns else low.get(a.sample_col.lower())
            anc = [c if c in d.columns else low.get(c.lower()) for c in a.ancestries]
            missing = [w for w, got in zip(a.ancestries, anc) if got is None]
            if sc is None or missing:
                # Printing "[ok] id column None, ancestry columns [None, None, None, None]" and
                # then five strata of zero is how a column-naming problem gets mistaken for a
                # cohort with no non-Europeans in it.
                print(NO + f'{Path(f).name}: cannot find the columns this file is read by.')
                print(HM + f'  columns present: {list(d.columns)[:12]}'
                          + (' ...' if len(d.columns) > 12 else ''))
                if sc is None:
                    print(HM + f'  no id column matching {a.sample_col!r}'
                               '  -- pass --sample-col <name>')
                if missing:
                    print(HM + f'  no ancestry column(s) for {missing}'
                               '  -- pass --ancestries <name> <name> ...')
                print(HM + '  the strata below will all read zero until this is fixed; that is '
                           'the column names, not the cohort.')
            else:
                print(OK + f'{Path(f).name}: id column {sc!r}, ancestry columns {anc}')
            full = L.read_table(f, what='--rfmix')
            full.columns = [c for c in full.columns]
            P = pd.DataFrame({x: pd.to_numeric(full[y], errors='coerce')
                              for x, y in zip(a.ancestries, anc) if y})
            for c in L.ANCESTRIES:
                if c not in P:
                    P[c] = 0.0
            frames.append(pd.Series(L.assign_stratum(P)).value_counts())
        tot = pd.concat(frames, axis=1).fillna(0).sum(axis=1).astype(int)
        for s in L.STRATA:
            v = int(tot.get(s, 0))
            print((OK if v >= 100 else HM) + f'{s}: {v:,}'
                  + ('' if v >= 100 else '  (below the 100-person minimum; not analysed)'))
        rep['strata'] = {s: int(tot.get(s, 0)) for s in L.STRATA}

    print('\ngenotypes')
    if not a.geno:
        print(HM + "not checked: pass --geno '<prefix>chr@' (or a VCF pattern)")
    if a.geno:
        found = []
        for c in range(1, 23):
            try:
                L.geno_args(a.geno, c); found.append(c)
            except SystemExit:
                pass
        from ancestry import convert as CV
        if CV.looks_like_vcf(a.geno):
            print(HM + 'these are VCF/BCF. plink can read them directly, but it re-reads them '
                       'on every call -- about a hundred calls over a full run.')
            print(HM + '  run `convert` once first (or just use `all`, which inserts it).')
            rep['vcf_input'] = True
        print((OK if found else NO) + f'chromosomes resolvable: {len(found)} '
              + (f'({found[0]}..{found[-1]})' if found else ''))
        rep['chroms'] = found
        if found:
            # a private temporary directory, not a fixed path in /tmp: a stale file from an
            # earlier run at the same path would otherwise be read as this run's success
            with tempfile.TemporaryDirectory() as td:
                td = Path(td)
                try:
                    args = L.geno_args(a.geno, found[0])
                    L.sh([L.plink2()] + args + ['--write-snplist', '--out', str(td / 'all')],
                         check=False, quiet=True)
                    snps = td / 'all.snplist'
                    nv = sum(1 for _ in open(snps)) if snps.exists() else 0
                    print((OK if nv else HM) + f'plink2 reads chr{found[0]}: '
                          + (f'{nv:,} variants' if nv else 'FAILED, see the message above'))
                    rep['n_variants_chr_probe'] = nv
                except SystemExit as e:
                    print(NO + f'plink2 could not read it: {e}')

                # WHICH BUILD.  This is the question the survey exists to answer, and it is
                # answered by looking, not by inference: the discovery leads are extracted BY
                # POSITION in both coordinate systems and the one that finds variants is the
                # build.  A GRCh38 position in a GRCh37 release hits almost nothing, so the two
                # counts are never close, and a run started on the wrong build produces a clean,
                # confident, entirely wrong null days later.
                from ancestry import geno_ops as G
                hits = {}
                for b in ('grch38', 'grch37'):
                    Lx = pd.read_csv(ref / f'leads_{b}.tsv', sep='\t', dtype={'CHR': str})
                    sub = Lx[Lx.CHR.astype(str).str.replace('chr', '', regex=False)
                             == str(found[0])]
                    if not len(sub):
                        continue
                    rng_file = G.write_extract(sub, td / f'{b}.range')
                    try:
                        L.sh([L.plink2()] + L.geno_args(a.geno, found[0])
                             + G.extract_args(rng_file)
                             + ['--write-snplist', '--out', str(td / b)],
                             check=False, quiet=True)
                        f2 = td / f'{b}.snplist'
                        hits[b] = sum(1 for _ in open(f2)) if f2.exists() else 0
                    except SystemExit:
                        hits[b] = 0
                    print(f'{OK if hits.get(b) else HM}{b}: {hits.get(b, 0)} of {len(sub)} '
                          f'chr{found[0]} leads found')
                rep['build_probe'] = hits
                if hits and max(hits.values()) > 0:
                    best = max(hits, key=lambda k: hits[k])
                    other = min(hits, key=lambda k: hits[k])
                    clear = hits[best] >= 2 * max(1, hits[other])
                    rep['build'] = best if clear else None
                    print((OK if clear else HM) +
                          (f'build looks like {best.upper()} -- pass --build {best}'
                           if clear else
                           'the two builds score alike on this chromosome; run `geno`, which '
                           'tries every lead genome-wide, before trusting either'))
                else:
                    print(NO + 'neither build found any of this chromosome\'s leads. The '
                               'genotypes may be on a different build again, or the chromosome '
                               'may be spelled differently (chr1 vs 1) -- `geno` prints both.')

    print('\nground truth for the fairness audit')
    F = pd.read_csv(ref / 'findings77.tsv', sep='\t')
    if a.labels and Path(a.labels).exists():
        D = L.read_table(a.labels, what='--labels', nrows=2000)
        have = [f for f in F.finding if f in D.columns]
        print((OK if have else NO) + f'{a.labels}: {len(have)} of 77 findings by name')
        if not have:
            print(HM + f'  its columns: {list(D.columns)[:10]} ...')
            print(HM + '  rename them to the `finding` column of ref/findings77.tsv, or pass '
                       '--statements instead')
        rep['label_findings'] = len(have)
    elif a.statements and Path(a.statements).exists():
        print(OK + f'{a.statements}: free text; {(F.muse_regex.fillna("") != "").sum()} of 77 '
              'findings have a text pattern')
    else:
        print(NO + 'no --labels and no --statements: the fairness audit cannot run')

    # ----------------------------------------------------------------- deprivation (optional)
    if a.ses:
        if not Path(a.ses).exists():
            print(NO + f'--ses {a.ses} does not exist')
        else:
            try:
                from ancestry import ses as RS
                V, vcol = RS.load_ses(a.ses, a.ses_col)
                v = V.sdi.to_numpy(float)
                pct = (np.nanmin(v) >= 0) and (np.nanmax(v) <= 100) and (np.nanmax(v) > 1.5)
                print(OK + f'{a.ses}: column {vcol!r}, {len(V):,} people, '
                           f'range {np.nanmin(v):g}-{np.nanmax(v):g}')
                cut, label, _n = RS.pick_cut(v, 'national-median')
                print(('   ' if pct else HM)
                      + f'  cut would be {label} = {cut:g}, putting '
                        f'{100 * np.mean(v >= cut):.0f}% of these people on the deprived side'
                      + ('' if pct else '  -- the values do not look like a 1-100 percentile '
                                        'rank, so there is no national anchor and the cohort '
                                        'median would be used; say so in any caption'))
                rep['ses'] = dict(column=vcol, n=int(len(V)), looks_like_percentile=bool(pct),
                                  pct_deprived_at_cut=round(100 * float(np.mean(v >= cut)), 1))
                # the join is the thing that actually fails in practice
                if a.prepared or a.ecg_meta:
                    ids = None
                    for src in (Path(a.prepared or '') / 'all_genotyped' / 'ids.csv',):
                        if src.exists():
                            ids = pd.read_csv(src, dtype=str).iloc[:, 0].astype(str)
                    if ids is not None:
                        hit = V.participant_id.isin(set(ids)).mean()
                        hit2 = V.participant_id.map(L.canon_id).isin(
                            set(ids.map(L.canon_id))).mean()
                        best = max(hit, hit2)
                        print((OK if best > 0.5 else NO)
                              + f'  {100 * best:.0f}% of these ids match the prepared cohort'
                              + ('' if hit >= hit2 else ' (after the canonical id fallback)'))
                        rep['ses_id_match_pct'] = round(100 * float(best), 1)
            except SystemExit as e:
                print(NO + f'--ses {a.ses}: {e}')
            except Exception as e:                                       # noqa: BLE001
                print(NO + f'--ses {a.ses}: {type(e).__name__}: {e}')

    if a.out:
        # This report is the one file that leaves the site WITHOUT passing through `package`,
        # so it is masked and audited here instead.  It holds only counts and yes/no answers,
        # but a stratum of four people is still a count of four people.
        from ancestry import privacy as PV
        Path(a.out).write_text(json.dumps(PV.mask(rep), indent=2, default=str))
        bad = PV.audit_files([Path(a.out)])
        if bad:
            Path(a.out).unlink()
            raise SystemExit('the survey report did not pass its own disclosure audit and was '
                             f'deleted: {bad[:5]}')
        print(f'\n  wrote {a.out}  (masked and audited; safe to send back)')


if __name__ == '__main__':
    main()
