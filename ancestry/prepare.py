"""Mode `prepare` -- the roster: who is in which ancestry stratum, with which ECG, and where.

This image asks three questions of MGB-MGH that the October ancestry run did not: a GWAS, a
polygenic-score validation, and a fairness audit against the hospital's own reads.  All three
need something the October run deliberately does not produce -- a link from a row of X.npy to a
person, and from that person to a genotype.  That link is built HERE and stays here.

WHAT THIS MODE DOES, in order:

  1. Runs the deployed container's own `prepare` (prepare_inputs.py, byte-identical to the image
     MGB already ran), so all_genotyped/ is the same directory as before and the ancestry numbers
     in this package can be compared with October's without a caveat.  `--reuse-prepared` skips
     this and points at the October output instead, which is faster and removes even the
     possibility of a difference.

  2. Rebuilds the linkage with the SAME arguments and uses all_genotyped/source_rows.npy to recover
     exactly which persons survived waveform loading.  A legacy prepared directory without this
     identity-preserving index is accepted only when its covariate alignment is provably unique;
     an ambiguous reuse stops instead of guessing.

  3. Assigns each person to an ancestry STRATUM at the 0.70 RFMix gate: EUR, AFR, EAS, SAS, or
     ADMIXED when no single ancestry reaches it.  EUR is the discovery setup.  AFR, EAS and SAS
     are new, and the whole point of them is that the European-discovered leads have never been
     asked to work there.

  4. Writes work/roster.csv (ids, stratum, fractions, age, sex, batch, X row) -- WHICH STAYS AT
     THE SITE -- and results/cohort/*.csv, which are masked counts and nothing else.

    python -m ancestry prepare --rfmix RF1.txt [RF2.txt ...] --ecg-meta meta1.parquet [...] \
        --work /out/work --results /out/results [--output /out/prepared] \
        [--reuse-prepared /path/to/october/prepared] \
        [--linkage conv.csv --barcode-col Barcode --patient-id-col ... --date-col ...]

Every flag of the deployed `prepare` is accepted and forwarded unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

from ancestry import prepare_external as PI                              # noqa: E402
from ancestry import privacy as PV                                       # noqa: E402
from ancestry import lib as L                                            # noqa: E402

# Arguments that belong to ancestry.prepare_external and are forwarded to it verbatim.
FORWARD = ['--rfmix', '--ecg-meta', '--linkage', '--threshold', '--caliper', '--ancestries',
           '--eur-label', '--sample-col', '--barcode-col', '--pseudo-col', '--ecg-id-col',
           '--patient-id-col', '--patient-id-prefix', '--date-col', '--pairing', '--gender-col',
           '--age-col', '--path-col', '--ecg-prefix', '--rfmix-sep', '--seed', '--min-cell']
FORWARD_FLAGS = ['--skip-repeats', '--no-plots', '--write-unmatched']


def resolve_columns(a):
    """Let the RFMix header name its own columns rather than making the caller spell them out.

    A live site writes 'Sample' where the container's default says 'sample', and the ancestry
    columns can be cased either way.  Resolving against the actual header is the difference
    between a run that starts and a run that stops on a capital letter.
    """
    probe = None if a.rfmix_sep in (None, 'auto') else a.rfmix_sep
    head = list(PI._read(a.rfmix[0], what='--rfmix', sep=probe, nrows=0).columns)
    low = {c.lower(): c for c in head}
    if a.sample_col not in head and a.sample_col.lower() in low:
        print(f'  rfmix id column: {low[a.sample_col.lower()]!r} (asked for {a.sample_col!r})')
        a.sample_col = low[a.sample_col.lower()]
    anc = [low.get(c.lower(), c) if c not in head else c for c in a.ancestries]
    if anc != a.ancestries:
        print(f'  rfmix ancestry columns: {anc} (asked for {a.ancestries})')
        if a.eur_label in a.ancestries:
            a.eur_label = anc[a.ancestries.index(a.eur_label)]
        a.ancestries = anc
    missing = [c for c in [a.sample_col] + a.ancestries if c not in head]
    if missing:
        raise SystemExit(f'{a.rfmix[0]}: no column matches {missing}\n  header: {head}\n'
                         '  pass --sample-col / --ancestries as this file spells them')
    return a


def run_deployed_prepare(a, extra):
    """The deployed container's own prepare, with the arguments it was given in October."""
    argv = ['ancestry.prepare_external', '--output', str(a.output)]
    for f in FORWARD:
        v = getattr(a, f[2:].replace('-', '_'), None)
        if v is None or v is False:
            continue
        argv += [f] + ([str(x) for x in v] if isinstance(v, list) else [str(v)])
    for f in FORWARD_FLAGS:
        if getattr(a, f[2:].replace('-', '_'), False):
            argv.append(f)
    argv += extra
    print('  running the deployed prepare:\n    ' + ' '.join(argv[1:]))
    sys.argv = argv
    PI.main()


def recover_rows(a, prepared: Path):
    """Which person is each row of X.npy?  Exact retained indices, or a hard stop.

    Current preparations persist source_rows.npy alongside X.npy.  Older prepared directories can
    be aligned by covariates only when that mapping is provably unique; duplicate covariate keys
    with dropped rows are rejected and must be regenerated with the current image.
    """
    gdir = prepared / 'all_genotyped'
    C = pd.read_csv(gdir / 'covariates.csv')
    nX = len(np.load(gdir / 'X.npy', mmap_mode='r'))
    if len(C) != nX:
        raise SystemExit(f'{gdir}: covariates.csv has {len(C):,} rows, X.npy has {nX:,}. '
                         'This prepared directory is inconsistent.')
    print(f'  prepared: {nX:,} ECGs in all_genotyped/')

    batch_ids = list(range(1, len(a.rfmix) + 1))
    round_ids = list(range(1, len(a.ecg_meta) + 1))
    R = PI.build_rfmix_table(a.rfmix, batch_ids, a.sample_col, a.ancestries, a.threshold,
                             a.rfmix_sep)
    M = PI.load_ecg_metadata(a.ecg_meta, round_ids, a.ecg_id_col, a.gender_col, a.age_col,
                             a.path_col, a.ecg_prefix, patient_id_col=a.patient_id_col,
                             patient_id_prefix=a.patient_id_prefix, date_col=a.date_col)
    LK = PI.link(R, M, a.linkage, a.barcode_col, a.pseudo_col)
    LK['gender'] = PI.normalise_sex(LK['gender'])
    P = PI.person_table(LK, round_ids, require_demographics=False).reset_index(drop=True)
    print(f'  linkage rebuilt: {len(P):,} persons with a linked ECG')
    if len(P) < nX:
        raise SystemExit(f'the rebuilt linkage has fewer persons ({len(P):,}) than all_genotyped '
                         f'has ECGs ({nX:,}): these are not the arguments `prepare` was run with.')

    anc = [c for c in a.ancestries if c in C.columns and c in P.columns]

    def key(row, sexc, agec):
        return (str(row['batch']), str(row['round']), str(row[sexc]),
                f"{float(row[agec]):.4f}" if pd.notna(row[agec]) else 'na',
                tuple(f'{float(row[c]):.6f}' for c in anc))

    pk = [key(r, 'gender', 'age') for _, r in P.iterrows()]
    ck = [key(r, 'sex', 'age') for _, r in C.iterrows()]
    source_rows = gdir / 'source_rows.npy'
    if source_rows.exists():
        keep = np.load(source_rows, allow_pickle=False)
        if keep.ndim != 1 or not np.issubdtype(keep.dtype, np.integer):
            raise SystemExit(f'{source_rows}: expected a one-dimensional integer array')
        keep = keep.astype(np.int64, copy=False)
        if len(keep) != nX:
            raise SystemExit(f'{source_rows}: has {len(keep):,} rows, expected {nX:,}')
        if len(keep) and (keep[0] < 0 or keep[-1] >= len(P) or np.any(np.diff(keep) <= 0)):
            raise SystemExit(f'{source_rows}: indices must be unique, increasing, and within the '
                             f'{len(P):,}-row rebuilt linkage')
        selected = [pk[int(i)] for i in keep]
        if selected != ck:
            bad = next(i for i, (left, right) in enumerate(zip(selected, ck)) if left != right)
            raise SystemExit(f'{source_rows}: retained identity row {bad:,} does not match '
                             'covariates.csv. Nothing was written.')
        print(f'  recovered {len(keep):,}/{nX:,} rows from exact retained source indices '
              f'({len(P) - len(keep):,} waveform loads failed)')
        return P.iloc[keep].reset_index(drop=True), C, anc

    # Compatibility with prepared directories created before source_rows.npy existed.  A key is
    # ambiguous precisely when multiple linked people share it and only some of them survived.
    pc, cc = Counter(pk), Counter(ck)
    ambiguous = [k for k, n in pc.items() if n > 1 and 0 < cc[k] < n]
    if ambiguous:
        raise SystemExit(
            f'{gdir}: this legacy prepared directory has no source_rows.npy and '
            f'{len(ambiguous):,} duplicated covariate key(s) lost only some waveform rows, so '
            'participant identity cannot be recovered safely. Re-run `prepare` with the current '
            'image instead of using --reuse-prepared.')

    keep, j = [], 0
    for i, k in enumerate(ck):
        while j < len(pk) and pk[j] != k:
            j += 1
        if j >= len(pk):
            raise SystemExit(f'could not align covariates row {i:,} to the rebuilt linkage. '
                             'Nothing was written.')
        keep.append(j); j += 1
    print(f'  aligned {len(keep):,}/{nX:,} rows '
          f'({len(P) - len(keep):,} persons dropped by prepare: waveform would not load)')
    return P.iloc[keep].reset_index(drop=True), C, anc


def roster_ids(linked_rows, geno_id_col=None):
    """Return participant ids and genotype ids without conflating their namespaces."""
    if 'project_pseudo_id' not in linked_rows or 'Sample' not in linked_rows:
        raise SystemExit('the rebuilt linkage must contain project_pseudo_id and Sample')
    pid = linked_rows['project_pseudo_id'].astype(str).to_numpy()
    gid_col = geno_id_col or 'Sample'
    if gid_col not in linked_rows.columns:
        raise SystemExit(f'--geno-id-col {gid_col!r} is not a column of the linked table '
                         f'(have: {list(linked_rows.columns)})')
    return pid, linked_rows[gid_col].astype(str).to_numpy()


def main():
    ap = argparse.ArgumentParser(prog='python -m ancestry prepare', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rfmix', nargs='+', required=True,
                    help='RFMix weighted-sum matrix/matrices: one row per genotyped sample, one '
                         'column per continental ancestry (csv/tsv/txt/.gz)')
    ap.add_argument('--ecg-meta', nargs='+', required=True,
                    help='the ECG inventory: one row per recording, with an id, a path, age, sex '
                         'and (if you have them) a patient id and a date. One file per round')
    ap.add_argument('--work', required=True, help='stays at the site; carries ids')
    ap.add_argument('--results', required=True, help='aggregate only; this is what you send back')
    ap.add_argument('--output', default=None, help='where to write the prepared arrays')
    ap.add_argument('--reuse-prepared', default=None,
                    help="an existing prepared directory (October's); skips re-preparing")
    ap.add_argument('--geno-id-col', default=None,
                    help='column of the linkage table holding the GENOTYPE id, when it is not '
                         'the RFMix sample id (default: the RFMix sample id is the genotype id)')
    ap.add_argument('--linkage', default=None,
                    help='optional barcode -> participant conversion table, when the genotype ids '
                         'and the ECG ids are different namespaces')
    ap.add_argument('--threshold', type=float, default=L.GATE,
                    help='the ANCESTRY GATE: a participant is called a continental ancestry when '
                         f'that ancestry is at least this fraction of their genome (default '
                         f'{L.GATE:g}, the discovery value -- change it and the strata, not just '
                         'their size, change)')
    ap.add_argument('--caliper', type=float, default=2.0,
                    help='age caliper in YEARS for the 1:1 sex-and-age matched subset: a pair is '
                         'kept only if the two differ by less than this (default 2)')
    ap.add_argument('--ancestries', nargs='+', default=['AFR', 'EAS', 'SAS', 'EUR'],
                    help='the RFMix columns to read, in the order they are reported')
    ap.add_argument('--eur-label', default='EUR',
                    help='which of --ancestries is the European reference class (default EUR)')
    ap.add_argument('--sample-col', default='Sample',
                    help='the sample-id column of the RFMix table')
    ap.add_argument('--barcode-col', default='Barcode',
                    help='the genotype-side id column of --linkage')
    ap.add_argument('--pseudo-col', default='project_pseudo_id',
                    help='the participant-side id column of --linkage')
    ap.add_argument('--ecg-id-col', default='ecg_id',
                    help='the recording-id column of --ecg-meta')
    ap.add_argument('--patient-id-col', default=None,
                    help='the PERSON column of --ecg-meta, when one person has several '
                         'recordings. Without it every row is treated as a different person, '
                         'and the repeat analyses cannot run')
    ap.add_argument('--patient-id-prefix', default='',
                    help='prefix to strip from the patient id (e.g. a site code)')
    ap.add_argument('--date-col', default=None,
                    help='the recording-date column; needed for the closest-in-time pairing and '
                         'for the interval the stability analysis reports')
    ap.add_argument('--pairing', default='both', choices=['first_last', 'closest', 'both'],
                    help="how two recordings of one person are paired: first_last (the widest "
                         "interval, the harshest test), closest (the two consecutive recordings "
                         "nearest in time, needs --date-col), or both (default) -- the interval "
                         "drives the number, so one pairing alone would misstate stability")
    ap.add_argument('--gender-col', default='gender',
                    help='the sex column of --ecg-meta (M/F, Male/Female, 1/0 all read)')
    ap.add_argument('--age-col', default='age',
                    help='the age-at-recording column of --ecg-meta, in years')
    ap.add_argument('--path-col', default='full_ecg_path',
                    help='the waveform-path column of --ecg-meta')
    ap.add_argument('--ecg-prefix', default='ECG_',
                    help='prefix to strip from the ECG id when it is built from the path')
    ap.add_argument('--rfmix-sep', default='auto',
                    help="delimiter of the RFMix table; 'auto' reads it from the file (default)")
    ap.add_argument('--seed', type=int, default=0,
                    help='random seed for the matching draw, so the matched set is reproducible')
    ap.add_argument('--min-cell', type=int, default=PV.MIN_CELL,
                    help=f'suppress any returned cell smaller than this (default '
                         f'{PV.MIN_CELL}); raising it suppresses more, lowering it is a '
                         'disclosure decision for your own committee')
    ap.add_argument('--skip-repeats', action='store_true',
                    help='do not build the repeat-ECG arrays (skips the stability analysis)')
    ap.add_argument('--no-plots', action='store_true',
                    help='write the tables but draw no plates')
    ap.add_argument('--write-unmatched', action='store_true',
                    help='also write the arrays for everyone who did not enter the matched '
                         'subset, so an analysis can use the whole genotyped cohort')
    a, extra = ap.parse_known_args()

    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    res = Path(a.results); (res / 'cohort').mkdir(parents=True, exist_ok=True)
    a = resolve_columns(a)

    L.banner('prepare: roster and ancestry strata')
    if a.reuse_prepared:
        prepared = Path(a.reuse_prepared)
        print(f'  reusing {prepared} (not re-running the deployed prepare)')
        if not (prepared / 'all_genotyped' / 'X.npy').exists():
            raise SystemExit(f'no all_genotyped/X.npy under {prepared}')
    else:
        a.output = a.output or str(work / 'prepared')
        run_deployed_prepare(a, extra)
        prepared = Path(a.output)

    T, C, anc = recover_rows(a, prepared)

    # ---------------------------------------------------------------- the roster (stays here)
    props = pd.DataFrame({c: pd.to_numeric(C[c], errors='coerce').to_numpy() for c in anc})
    for c in L.ANCESTRIES:                       # a cohort may not carry all four
        if c not in props:
            props[c] = 0.0
    stratum = L.assign_stratum(props, ancestries=L.ANCESTRIES, gate=a.threshold)
    # PI.link preserves the RFMix genotype-side identifier under the canonical Sample column.
    # Participant ids are a different namespace whenever a linkage table is used.
    pid, gid = roster_ids(T, a.geno_id_col)

    roster = pd.DataFrame({
        'row': np.arange(len(C)),                # the row of X.npy, which is what score reads
        'participant_id': pid,
        'geno_id': gid,
        'stratum': stratum,
        'age': pd.to_numeric(C['age'], errors='coerce').to_numpy(),
        'sex': C['sex'].to_numpy(),
        'batch': T['batch'].to_numpy(),
        'round': T['round'].to_numpy(),
    })
    for c in L.ANCESTRIES:
        roster[f'rf_{c}'] = props[c].to_numpy()
    roster['rf_nonEUR'] = 1.0 - roster.rf_EUR
    if 'in_matched' in C:
        roster['in_matched'] = C['in_matched'].to_numpy()
    roster.to_csv(work / 'roster.csv', index=False)
    print(f'\n  wrote {work / "roster.csv"}  ({len(roster):,} people)   THIS FILE CARRIES IDS '
          'AND STAYS HERE')
    (work / 'prepared_dir.txt').write_text(str(prepared.resolve()) + '\n')

    # ---------------------------------------------------------------- what leaves (counts only)
    n = roster.groupby('stratum').size().reindex(L.STRATA).fillna(0).astype(int)
    S = pd.DataFrame({'stratum': n.index, 'n': n.to_numpy()})
    S['analysed'] = [('yes' if v >= PV.MIN_STRATUM else
                      f'no (below the {PV.MIN_STRATUM}-person minimum)') for v in S.n]
    S['mean_age'] = [round(float(roster.loc[roster.stratum == s, 'age'].mean()), 1)
                     if v >= PV.MIN_STRATUM else np.nan for s, v in zip(S.stratum, S.n)]
    S['pct_female'] = [round(100 * float((roster.loc[roster.stratum == s, 'sex']
                                          .astype(str).str.upper().isin(['F', '0', 'FEMALE'])
                                          ).mean()), 1) if v >= PV.MIN_STRATUM else np.nan
                       for s, v in zip(S.stratum, S.n)]
    S['mean_rf_dominant'] = [round(float(roster.loc[roster.stratum == s, f'rf_{s}'].mean()), 3)
                             if (s in L.ANCESTRIES and v >= PV.MIN_STRATUM) else np.nan
                             for s, v in zip(S.stratum, S.n)]
    PV.write_table(S, res / 'cohort' / 'strata.csv', where='strata.csv')
    print('\n' + S.to_string(index=False))

    B = (roster.groupby(['batch', 'stratum']).size().unstack(fill_value=0)
         .reindex(columns=L.STRATA, fill_value=0).reset_index())
    PV.write_table(B, res / 'cohort' / 'strata_by_batch.csv', where='strata_by_batch.csv',
                   count_cols=L.STRATA)

    prov = dict(version=L.VERSION, n_prepared=int(len(roster)), gate=a.threshold,
                ancestries=list(a.ancestries), rfmix=[str(x) for x in a.rfmix],
                ecg_meta=[str(x) for x in a.ecg_meta], linkage=a.linkage,
                prepared_dir=str(prepared.resolve()), reused=bool(a.reuse_prepared),
                strata={k: int(v) for k, v in n.items()},
                strata_analysable=[s for s, v in n.items() if v >= PV.MIN_STRATUM])
    (res / 'cohort' / 'prepare_provenance.json').write_text(
        json.dumps(PV.mask(prov, k=a.min_cell), indent=2))
    print(f'\n  strata that clear the {PV.MIN_STRATUM}-person minimum: '
          f'{", ".join(prov["strata_analysable"]) or "none"}')
    print(f'  wrote {res / "cohort"}/  (counts only)')


if __name__ == '__main__':
    main()
