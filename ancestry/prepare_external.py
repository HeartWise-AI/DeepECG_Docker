"""Mode `prepare` -- build model-ready inputs from RFMix matrices + ECG metadata.

End-to-end preprocessing (adapted from the LifeLines/UGLI pipeline notebook):
  RFMix weighted-sum matrices  ->  per-sample continental ancestry proportions
  -> dominant-ancestry label at --threshold (else ADMIXED)
  -> (optional) conversion/linkage table  Sample(barcode) -> project_pseudo_id
  -> link to ECG metadata (one parquet per round; ecg_id -> project_pseudo_id, or an explicit
     patient-id column with --patient-id-col when ECG id and patient id are different columns)
  -> one ECG per person (earliest round) for the main analyses, plus, when a person has more
     than one ECG, a repeat pair per person for the intrapatient-stability analysis
  + participant-flow / coverage / overlap / match-balance tables (small cells masked).

Written under <output>/:
  all_genotyped/  INPUT OF `validate`. Everyone with genetics + a linked ECG (EUR, non-EUR AND
               admixed), one ECG each, plus covariates.csv (ancestry proportions, sex, age,
               batch, round, in_matched; no ids). `validate` derives every label and subgroup
               from it and runs analyses 1-7 in one inference pass.
  first_only/  the 1:1 age/sex-matched subset (validate analysis 5 is computed from the
               in_matched flag; this folder is for the optional stand-alone modes). If nobody
               can be matched (e.g. unrecognised sex codes) it is skipped with a warning and
               analysis 5 is reported as suppressed; nothing else is affected.
  all_rounds/  every ECG of the matched persons (optional `admixed` mode).
  all_genotyped_repeats/  INPUT OF `validate` ANALYSIS 8 (intrapatient stability). Persons with
               >= 2 ECGs, two recordings each (first/last and, with dates, the closest-in-time
               consecutive pair), plus covariates.csv carrying an arbitrary within-file person
               index -- never the site's id -- the gap in days and the person's ECG count.
  unmatched/   only with --write-unmatched: gated EUR + non-EUR as X.npy/y.npy for the
               optional stand-alone `metrics` / `ensemble` modes.

Column names, the ECG-id prefix, ancestries and matching caliper are all configurable;
defaults match the reference notebook. ECG waveforms are read from the path column of
the metadata (np.load -> (2500,12) or (12,2500)).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd

from ancestry.privacy import mask_frame


# --------------------------------------------------------------------------- labels
def classify(df, ancestries, threshold):
    props = df[ancestries].to_numpy(dtype=float)
    dom = np.argmax(props, axis=1)
    val = props[np.arange(len(df)), dom]
    return np.where(val >= threshold, np.asarray(ancestries)[dom], "ADMIXED").astype(object)


def _read(path, what='', dtype=None, sep=None, **kw):
    """Read a site-supplied delimited table, sniffing the delimiter and naming the real problem.

    An empty file and a failed id join produce the same downstream symptom -- zero rows -- so
    the two are separated here, where the file is still in hand."""
    p = Path(path)
    label = what or str(p)
    if not p.exists():
        raise SystemExit(f'{label}: {p} does not exist')
    if p.stat().st_size == 0:
        raise SystemExit(f'{label}: {p} is an empty file (0 bytes)')
    use = sep or _sniff(p)
    try:
        D = pd.read_csv(p, sep=use, dtype=dtype, **kw)
    except Exception as e:                                               # noqa: BLE001
        raise SystemExit(f'{label}: {p} could not be parsed as a '
                         f'{_SEPNAME.get(use, use)}-separated table '
                         f'({type(e).__name__}: {e})')
    if D.shape[1] == 1 and use != ',' and kw.get('nrows') != 0:
        raise SystemExit(f'{label}: {p} parsed to a single column named '
                         f'{list(D.columns)[0]!r}; its delimiter was read as '
                         f'{_SEPNAME.get(use, use)}. Check the file.')
    return D


def _col(df, wanted, candidates, what, kind, flag):
    """Resolve a column name case-insensitively, or fail naming what was actually there."""
    lower = {str(c).strip().lower(): c for c in df.columns}
    if wanted and wanted in df.columns:
        return wanted
    if wanted and str(wanted).strip().lower() in lower:
        return lower[str(wanted).strip().lower()]
    for c in candidates:
        if str(c).lower() in lower:
            return lower[str(c).lower()]
    cols = list(df.columns)
    raise SystemExit(f'{what}: no {kind} found'
                     + (f' (asked for {wanted!r})' if wanted else '')
                     + f'.\n  columns present: {cols[:15]}'
                     + (' ...' if len(cols) > 15 else '')
                     + f'\n  pass {flag} <name>')


_SEPNAME = {'\t': 'tab', ',': 'comma', ';': 'semicolon', '|': 'pipe'}


def _read_meta(path, cols):
    """The ECG inventory, in whatever format the site exports it.

    Parquet is what the discovery cohorts used and what the documentation asks for, but an ECG
    inventory pulled out of a hospital reporting system arrives as CSV or TSV far more often
    than as parquet, and refusing it would make the site convert a file for no reason. Format
    is decided by content, not by extension: a `.csv` that is really parquet, and a `.parquet`
    that is really text, both read correctly here."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f'--ecg-meta: {p} does not exist')
    if p.stat().st_size == 0:
        raise SystemExit(f'--ecg-meta: {p} is an empty file (0 bytes)')
    with open(p, 'rb') as fh:
        magic = fh.read(4)
    is_parquet = magic == b'PAR1'
    try:
        if is_parquet:
            m = pd.read_parquet(p, columns=cols).copy()
        else:
            m = _read(p, what='--ecg-meta').copy()
            missing = [c for c in cols if c not in m.columns]
            if missing:
                lower = {str(c).strip().lower(): c for c in m.columns}
                ren = {lower[c.lower()]: c for c in missing if c.lower() in lower}
                if ren:
                    m = m.rename(columns=ren)
                    print(f'[meta] {p.name}: matched {list(ren.values())} case-insensitively')
                    missing = [c for c in cols if c not in m.columns]
            if missing:
                raise KeyError(missing)
            m = m[cols].copy()
    except Exception as e:                      # name the missing column, not a pyarrow trace
        try:
            have = list(pd.read_parquet(p).columns) if is_parquet else \
                list(_read(p, what='--ecg-meta', nrows=0).columns)
        except Exception:                                                # noqa: BLE001
            have = []
        missing = [c for c in cols if c not in have]
        fmt = 'parquet' if is_parquet else 'delimited text'
        raise SystemExit(
            f'--ecg-meta: {p} (read as {fmt}) is missing column(s) {missing or cols}.'
            f'\n  columns present: {have[:12]}{" ..." if len(have) > 12 else ""}'
            f'\n  name them with --ecg-id-col / --gender-col / --age-col / --path-col'
            f'\n  (and --patient-id-col / --date-col if you passed those)'
            f'\n  underlying error: {type(e).__name__}: {e}') from e
    return m


def _sniff(path):
    """The delimiter the file actually uses. RFMix weighted-sum tables are usually tab-separated,
    but a site that opened one in Excel and saved it will hand us commas or semicolons, and
    reading that with sep='\t' yields ONE column whose name is the whole header -- which then
    fails as "missing columns" and sends them looking for the wrong problem."""
    import gzip
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt', errors='replace') as fh:
        head = fh.readline()
    best, n = '\t', 1
    for s_ in ('\t', ',', ';', '|'):
        if len(head.split(s_)) > n:
            best, n = s_, len(head.split(s_))
    return best


def build_rfmix_table(paths, batch_ids, sample_col, ancestries, threshold, sep):
    parts = []
    for p, bid in zip(paths, batch_ids):
        use = _sniff(p) if sep in (None, 'auto') else sep
        d = pd.read_csv(p, sep=use)
        missing = [c for c in [sample_col, *ancestries] if c not in d.columns]
        if missing:
            raise ValueError(f"{p}: missing columns {missing} (have {list(d.columns)[:8]}...)")
        d = d.copy()
        for c in ancestries:
            num = pd.to_numeric(d[c], errors="coerce")
            if num.isna().any():
                bad = d.loc[num.isna(), c].astype(str).head(3).tolist()
                raise ValueError(f"{p}: column {c} has {int(num.isna().sum())} non-numeric values, "
                                 f"e.g. {bad} (comma decimals? convert to '0.97')")
            d[c] = num
        d["Sample"] = d[sample_col].astype(str).str.strip()
        d["batch"] = bid
        d["ancestry"] = classify(d, ancestries, threshold)
        parts.append(d[["Sample", "batch", "ancestry", *ancestries]])
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------- ECG metadata
def load_ecg_metadata(paths, round_ids, ecg_id_col, gender_col, age_col, path_col, prefix,
                      patient_id_col=None, patient_id_prefix="", date_col=None):
    """One row per ECG. The person key is `project_pseudo_id`.

    By default it is the ECG id with `--ecg-prefix` stripped (the Lifelines convention, where the
    ECG id embeds the participant id). Sites where the patient id is its OWN column pass
    `--patient-id-col`; that column then becomes the person key and the ECG id is kept separately
    as `ecg_uid`, so several ECGs of one patient group correctly instead of looking like several
    patients. `--date-col` (optional) adds the acquisition timestamp, which orders repeats and
    gives the interval between them."""
    parts = []
    for p, r in zip(paths, round_ids):
        cols = [ecg_id_col, gender_col, age_col, path_col]
        for extra in (patient_id_col, date_col):
            if extra and extra not in cols:
                cols.append(extra)
        m = _read_meta(p, cols)
        m["ecg_uid"] = pd.Series(m[ecg_id_col], dtype="string").astype(str).str.strip()
        if patient_id_col:
            key = (pd.Series(m[patient_id_col], dtype="string")
                   .str.replace(rf"^{patient_id_prefix}", "", regex=True) if patient_id_prefix
                   else pd.Series(m[patient_id_col], dtype="string"))
        else:
            key = pd.Series(m[ecg_id_col], dtype="string").str.replace(rf"^{prefix}", "", regex=True)
        m["project_pseudo_id"] = key.astype(str).str.strip()
        if m["project_pseudo_id"].isin(["", "nan", "None", "<NA>"]).any():
            bad = int(m["project_pseudo_id"].isin(["", "nan", "None", "<NA>"]).sum())
            raise ValueError(f"{p}: {bad:,} rows have an empty patient id in "
                             f"{patient_id_col or ecg_id_col}")
        m["when"] = (pd.to_datetime(m[date_col], errors="coerce") if date_col
                     else pd.Series(pd.NaT, index=m.index))
        if date_col and m["when"].isna().all():
            print(f"[WARNING] round {r}: no value in --date-col {date_col} could be parsed as a "
                  "date; repeats will be ordered by round and ECG id instead", flush=True)
        m["round"] = r
        m = m.rename(columns={gender_col: "gender", age_col: "age", path_col: "full_ecg_path"})
        if len(m):
            print(f"[round {r}] {len(m):,} ECG rows | {m['ecg_uid'].iloc[0]} -> person "
                  f"{m['project_pseudo_id'].iloc[0]}"
                  + (f" | {int(m['when'].notna().sum()):,} dated" if date_col else ""), flush=True)
        parts.append(m[["project_pseudo_id", "ecg_uid", "round", "gender", "age",
                        "full_ecg_path", "when"]])
    t = pd.concat(parts, ignore_index=True)
    per = t.groupby("project_pseudo_id").size()
    print(f"[metadata] {len(t):,} ECG rows from {len(per):,} persons "
          f"({int((per > 1).sum()):,} with >= 2 ECGs, max {int(per.max()) if len(per) else 0})",
          flush=True)
    return t


def link(rfmix_table, ecg_meta, linkage, barcode_col, pseudo_col):
    if linkage is None:                       # Sample already == project_pseudo_id
        t = rfmix_table.copy(); t["project_pseudo_id"] = t["Sample"]
        out = t.merge(ecg_meta, on="project_pseudo_id", how="inner")
        print(f"[link] no conversion table: Sample==project_pseudo_id | ECG rows={len(out):,}")
        _warn_overlap(rfmix_table, t, out)
        return out
    lk = _read(linkage, what='--linkage', dtype=str)
    barcode_col = _col(lk, barcode_col, ('barcode', 'sample', 'genotype_id', 'sample_id'),
                       '--linkage', 'genotype/barcode column', '--barcode-col')
    pseudo_col = _col(lk, pseudo_col, ('project_pseudo_id', 'participant_id', 'pseudo_id',
                                       'empi', 'mrn', 'patient_id'),
                      '--linkage', 'participant id column', '--pseudo-col')
    lk[barcode_col] = lk[barcode_col].astype(str).str.strip()
    lk[pseudo_col] = lk[pseudo_col].astype(str).str.strip()
    m = rfmix_table.merge(lk[[barcode_col, pseudo_col]].rename(
        columns={barcode_col: "Sample", pseudo_col: "project_pseudo_id"}),
        on="Sample", how="inner")
    out = m.merge(ecg_meta, on="project_pseudo_id", how="inner")
    print(f"[link] rfmix={len(rfmix_table):,} | barcode-linked={len(m):,} | ECG rows={len(out):,}")
    _warn_overlap(rfmix_table, m, out)
    return out


def _warn_overlap(rfmix_table, barcode_linked, linked):
    """String joins fail silently; shout when the overlap looks like an id-format mismatch."""
    n = rfmix_table["Sample"].nunique()
    for what, k in [("RFMix samples found in the linkage table", barcode_linked["Sample"].nunique()),
                    ("RFMix samples with >=1 linked ECG", linked["Sample"].nunique())]:
        if n and k / n < 0.10:
            print(f"[WARNING] only {k:,}/{n:,} ({k / n:.1%}) {what}. This usually means an id-format "
                  "mismatch (leading zeros, int vs text, prefix), not a small cohort.", flush=True)


# --------------------------------------------------------------------------- tables
def coverage_table(linked, rfmix_table, round_ids):
    idx = sorted(rfmix_table["batch"].unique())
    out = pd.DataFrame(index=idx)
    out["Genotyped"] = rfmix_table.groupby("batch")["Sample"].nunique()
    for r in round_ids:
        out[f"Round {r}"] = linked[linked["round"] == r].groupby("batch")["project_pseudo_id"].nunique()
    out["Any ECG"] = linked.groupby("batch")["project_pseudo_id"].nunique()
    out["Total ECGs"] = linked.groupby("batch").size()
    out = out.fillna(0).astype(int); out.loc["Total"] = out.sum(); out.index.name = "batch"
    return out


def ancestry_overlap(linked, rfmix_table, group_order, round_ids):
    out = pd.DataFrame(index=list(group_order))
    out["Genotyped"] = rfmix_table.groupby("ancestry")["Sample"].nunique()
    for r in round_ids:
        out[f"Round {r}"] = linked[linked["round"] == r].groupby("ancestry")["project_pseudo_id"].nunique()
    out["Any ECG"] = linked.groupby("ancestry")["project_pseudo_id"].nunique()
    out["Total ECGs"] = linked.groupby("ancestry").size()
    out = out.reindex(list(group_order)).fillna(0).astype(int)
    out.loc["Total"] = out.sum(); out.index.name = "Ancestry"
    return out


# --------------------------------------------------------------------------- matching
_SEX = {"MALE": "MALE", "M": "MALE", "MAN": "MALE",
        "FEMALE": "FEMALE", "F": "FEMALE", "WOMAN": "FEMALE"}


def normalise_sex(col):
    """MALE/FEMALE (also M/F, Man/Woman, any case). Anything else -> 'UNKNOWN'. 0/1 is
    deliberately NOT mapped: its coding differs between cohorts -- recode it upstream."""
    s = col.astype(str).str.strip().str.upper()
    out = s.map(_SEX).fillna("UNKNOWN")
    bad = s[out == "UNKNOWN"].value_counts()
    if len(bad):
        print(f"[sex] {int(bad.sum()):,} ECG rows with unrecognised sex (still scored; "
              f"excluded from the matched set and the sex fairness split only): {bad.head(8).to_dict()}")
    return out


def _ordered(t, round_priority):
    """ECGs of each person, earliest first: by round, then acquisition date when there is one,
    then the ECG id so the order is deterministic without dates."""
    rank = {r: i for i, r in enumerate(round_priority)}
    t = t.copy()
    t["__rk"] = t["round"].map(rank).fillna(99)
    t["__when"] = pd.to_datetime(t["when"], errors="coerce") if "when" in t else pd.NaT
    t["__uid"] = t["ecg_uid"].astype(str) if "ecg_uid" in t else ""
    return t.sort_values(["project_pseudo_id", "__rk", "__when", "__uid"],
                         na_position="last").reset_index(drop=True)


def person_table(linked, round_priority, require_demographics=True):
    t = linked.copy()
    t["age"] = pd.to_numeric(t["age"], errors="coerce")
    n0 = len(t)
    if require_demographics:                  # matching needs sex + age; unmatched does not
        t = t[t["gender"].isin(["MALE", "FEMALE"]) & t["age"].notna()]
    t = (_ordered(t, round_priority)
           .drop_duplicates("project_pseudo_id", keep="first")
           .drop(columns=["__rk", "__when", "__uid"]).reset_index(drop=True))
    print(f"[persons] {n0:,} ECG rows -> {len(t):,} unique usable persons"
          + ("" if require_demographics else " (age/sex not required)"))
    return t


def match_1to1(cases, controls, caliper, seed):
    rng = np.random.default_rng(seed)
    pool = {s: {"idx": g.index.to_numpy(), "age": g["age"].to_numpy(float),
                "used": np.zeros(len(g), bool)} for s, g in controls.groupby("gender")}
    pairs = []
    for ci in rng.permutation(cases.index.to_numpy()):
        s, a = cases.at[ci, "gender"], float(cases.at[ci, "age"])
        p = pool.get(s)
        if p is None or p["used"].all():
            continue
        d = np.abs(p["age"] - a); d[p["used"]] = np.inf
        j = int(np.argmin(d))
        if caliper is not None and d[j] > caliper:
            continue
        p["used"][j] = True
        pairs.append((int(ci), int(p["idx"][j]), float(d[j])))
    return pairs


def load_ecg(val):
    """One waveform as (2500, 12) float32 millivolts.

    Accepts what the main pipeline accepts, so a site already running DeepECG_Docker can point
    this at the files it already has: `.npy`, or `.base64` through the repository's own
    ECGFileHandler -- that format is a flat float32 buffer whose sample-major order is set by
    ECGFileHandler.save_ecg_signal, so there is no shape to guess. An in-memory array is taken
    as given.

    Shape handling follows utils/files_handler's rules rather than a second set of its own: a
    trailing singleton is squeezed, a lead-major array is transposed, and a longer recording is
    decimated by an integer step (a 10-second 500 Hz ECG -> 2500 samples), while anything
    shorter than 2500 samples is refused rather than padded. XML is NOT read here: upstream
    parses CLSA and MHI XML in XMLProcessor, but its cohort scaling happens further along, and
    an unverified scale would silently change every score instead of failing. Convert with the
    main pipeline first -- `.base64` is exactly what it writes.

    Units are not corrected, only checked later by infer.check_units: a silent rescale is the
    one failure this pipeline must not have.
    """
    if isinstance(val, (str, os.PathLike)):
        name = str(val)
        if name.lower().endswith(".xml"):
            raise ValueError(
                "XML is not read directly: run the main pipeline's preprocessing first and pass "
                "the .base64 files it writes (see utils/files_handler.XMLProcessor)")
        if name.endswith(".base64"):
            # The format is defined by utils/files_handler.ECGFileHandler.save_ecg_signal: a
            # base64 of raw float32, sample-major, 12 leads. Decoded identically here rather
            # than imported, because that module pulls in utils/credentials, which uses
            # `dict[str, str]` at module level and so needs Python >= 3.10, while this package
            # is 3.8-clean and runs in the analysis environment too. A test asserts the two
            # agree byte-for-byte wherever the import is possible.
            import base64
            arr = np.frombuffer(base64.b64decode(pathlib.Path(name).read_text()),
                                dtype=np.float32).reshape(-1, 12)
        else:
            arr = np.load(name)
    else:
        arr = val
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    if arr.ndim != 2:
        raise ValueError(f"unexpected ECG shape {arr.shape}")
    if arr.shape[0] == 12 and arr.shape[1] != 12:
        arr = arr.T
    if arr.shape[1] != 12:
        raise ValueError(f"unexpected ECG shape {arr.shape}: 12 leads are required")
    if arr.shape[0] != 2500:
        n = arr.shape[0]
        if n < 2500:
            raise ValueError(f"signal too short: {n} samples, 2500 are required")
        if n % 2500:
            # Only an exact multiple can be decimated to 2500 by taking every k-th sample. For
            # anything else, keeping the first 2500 would silently shorten the WINDOW instead of
            # lowering the sampling rate -- a 3000-sample recording would become 2500 samples of
            # a different duration and still score. Upstream's step arithmetic leaves such an
            # array at the wrong length, where the model raises; refusing here says why.
            raise ValueError(
                f"{n} samples is not a multiple of 2500, so it cannot be decimated to 10 s at "
                f"250 Hz by an integer step (2500, 5000 and 7500 are 250, 500 and 750 Hz). "
                f"Resample it first rather than letting it be truncated")
        arr = arr[::n // 2500, :]
    return np.ascontiguousarray(arr, dtype=np.float32)


def _stack_ecgs(paths, ids, out=None):
    """Load waveforms into (N,2500,12) float32. With `out` (a path), write straight into an
    on-disk .npy memmap so cohorts larger than RAM work."""
    n = len(paths)
    X = (np.lib.format.open_memmap(out, mode="w+", dtype=np.float32, shape=(n, 2500, 12))
         if out else np.empty((n, 2500, 12), dtype=np.float32))
    ok = np.ones(n, bool); n_bad = 0
    for i, p in enumerate(paths):
        try:
            X[i] = load_ecg(p)
        except Exception as e:
            ok[i] = False; n_bad += 1
            if n_bad <= 20:
                print(f"  skip ECG {i} ({ids[i]}): {e}")
            elif n_bad == 21:
                print("  ... further skipped ECGs not printed (count in the summary)")
        if n >= 5000 and (i + 1) % 5000 == 0:
            print(f"  loaded {i + 1:,}/{n:,} ECGs", flush=True)
    if n_bad:
        print(f"[ecg] {n_bad:,}/{n:,} waveforms could not be loaded and were skipped")
    return X, ok


def _save_ecgs(paths, ids, out_path):
    """Stream waveforms to `out_path` (.npy) without holding them in RAM; drops failed rows."""
    tmp = str(out_path) + ".tmp.npy"
    X, ok = _stack_ecgs(paths, ids, out=tmp)
    if ok.all():
        X.flush(); del X; os.replace(tmp, out_path)
    else:
        Y = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32,
                                      shape=(int(ok.sum()), 2500, 12))
        keep = np.flatnonzero(ok)
        for s in range(0, len(keep), 2048):
            Y[s:s + 2048] = X[keep[s:s + 2048]]
        Y.flush(); del X, Y; os.remove(tmp)
    return ok


# ------------------------------------------------------------------- repeats (analysis 8)
def repeat_pairs(linked, round_priority, pairing="both"):
    """Two recordings per person for the intrapatient-stability analysis.

    first_last  the first and the last ECG of the person: the widest interval, the harshest
                reading of "the same person recorded years apart".
    closest     the two CONSECUTIVE recordings closest in time (needs dates). Reported beside
                first_last because the interval drives the number: in our own data the same
                cohort gives 0.605 first-to-last (median 11.4 y) and 0.84 at a short interval,
                so one pairing alone would misstate the model's stability.

    Returns a long frame (two rows per person per pairing) with the gap in days and the
    person's total ECG count. No site identifier survives: `person` is a within-file index.
    """
    t = _ordered(linked, round_priority)
    n_ecgs = t.groupby("project_pseudo_id")["full_ecg_path"].transform("size")
    t = t[n_ecgs >= 2].copy()
    if not len(t):
        return pd.DataFrame(), pd.Series(dtype=int)
    t["n_ecgs"] = t.groupby("project_pseudo_id")["full_ecg_path"].transform("size")
    dated = t.groupby("project_pseudo_id")["__when"].transform(lambda c: c.notna().all())
    out = []

    if pairing in ("first_last", "both"):
        fl = pd.concat([t.groupby("project_pseudo_id", sort=False).head(1).assign(rep=0),
                        t.groupby("project_pseudo_id", sort=False).tail(1).assign(rep=1)])
        fl["pairing"] = "first_last"
        out.append(fl)

    if pairing in ("closest", "both"):
        d = t[dated].copy()
        if len(d):
            d["__next"] = d.groupby("project_pseudo_id")["__when"].shift(-1)
            d["__gap"] = (d["__next"] - d["__when"]).dt.total_seconds()
            cand = d[d["__gap"].notna()]
            if len(cand):
                pick = cand.loc[cand.groupby("project_pseudo_id")["__gap"].idxmin()]
                nxt = (d.set_index(["project_pseudo_id", "__when"])
                        .loc[list(zip(pick["project_pseudo_id"], pick["__next"]))]
                        .reset_index())
                cl = pd.concat([pick.assign(rep=0), nxt.assign(rep=1)])
                cl["pairing"] = "closest"
                out.append(cl.drop(columns=[c for c in ("__next", "__gap") if c in cl]))
        if pairing == "closest" and not out:
            print("[repeats] --pairing closest needs parsed dates; none available -> no repeats")
            return pd.DataFrame(), pd.Series(dtype=int)

    R = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if not len(R):
        return pd.DataFrame(), pd.Series(dtype=int)
    R = R.sort_values(["pairing", "project_pseudo_id", "rep"]).reset_index(drop=True)
    # gap in days per (pairing, person); NaN when that person has no usable dates
    g = R.pivot_table(index=["pairing", "project_pseudo_id"], columns="rep", values="__when",
                      aggfunc="first")
    gap = ((g[1] - g[0]).dt.days if {0, 1} <= set(g.columns) else pd.Series(index=g.index, dtype=float))
    R = R.merge(gap.rename("gap_days"), left_on=["pairing", "project_pseudo_id"],
                right_index=True, how="left")
    keep = R.groupby(["pairing", "project_pseudo_id"])["rep"].transform("nunique") == 2
    R = R[keep].reset_index(drop=True)
    # a person with exactly two ECGs has the same pair twice: score it once
    if R["pairing"].nunique() > 1:
        sig = (R.groupby(["pairing", "project_pseudo_id"])["full_ecg_path"]
                .apply(lambda c: tuple(sorted(c))).rename("sig").reset_index())
        wide = sig.pivot(index="project_pseudo_id", columns="pairing", values="sig")
        if {"first_last", "closest"} <= set(wide.columns):
            dup = wide.index[wide["first_last"] == wide["closest"]]
            n_dup = len(dup)
            R = R[~((R["pairing"] == "closest") & R["project_pseudo_id"].isin(dup))].reset_index(drop=True)
            if n_dup:
                print(f"[repeats] {n_dup:,} persons have exactly one possible pair "
                      "(closest == first/last): scored once")
    R["person"] = pd.factorize(R["project_pseudo_id"])[0]          # arbitrary within-file index
    per_person = t.drop_duplicates("project_pseudo_id").set_index("project_pseudo_id")["n_ecgs"]
    for pr, g2 in R.groupby("pairing"):
        gd = g2.drop_duplicates(["project_pseudo_id"])["gap_days"]
        med = f"{gd.median():.0f}" if gd.notna().any() else "no dates"
        print(f"[repeats] {pr}: {g2['project_pseudo_id'].nunique():,} persons, "
              f"median gap {med} days", flush=True)
    return R, per_person


def repeat_coverage(linked, group_order, min_cell):
    """How many ECGs each person has, by ancestry group -- counts only, small cells masked."""
    n = linked.groupby(["ancestry", "project_pseudo_id"]).size().rename("k").reset_index()
    buckets = pd.cut(n["k"], [0, 1, 2, 4, 9, np.inf], labels=["1", "2", "3-4", "5-9", "10+"])
    tab = pd.crosstab(n["ancestry"], buckets)
    tab = tab.reindex(index=[g for g in group_order if g in tab.index],
                      columns=["1", "2", "3-4", "5-9", "10+"], fill_value=0)
    tab["Persons"] = tab.sum(axis=1)
    tab["With >=2 ECGs"] = tab[["2", "3-4", "5-9", "10+"]].sum(axis=1)
    tab.loc["Total"] = tab.sum()
    tab.index.name = "Ancestry"
    return mask_frame(tab, k=min_cell)


def build_repeats(R, out_dir, ancestries, eur_label, threshold, min_persons=20):
    """Write all_genotyped_repeats/: the waveforms of the selected pairs + aggregate covariates.

    covariates.csv carries NO site identifier: `person` is a within-file index, and the only
    per-row information is the pairing, which recording of the pair it is, the gap in days, the
    person's ECG count and the usual ancestry/sex/age/batch/round columns.
    """
    n_persons = R["project_pseudo_id"].nunique()
    if n_persons < min_persons:
        print(f"[repeats] only {n_persons} persons with >= 2 ECGs (< {min_persons}): "
              "skipping all_genotyped_repeats/ (analysis 8 will be reported as unavailable)")
        return None
    os.makedirs(out_dir, exist_ok=True)
    ok = _save_ecgs(R["full_ecg_path"].to_numpy(), R["project_pseudo_id"].to_numpy(),
                    f"{out_dir}/X.npy")
    R = R[ok].reset_index(drop=True)
    # a pair is usable only if BOTH its recordings loaded
    full = R.groupby(["pairing", "person"])["rep"].transform("nunique") == 2
    if not full.all():
        drop = np.flatnonzero(~full.to_numpy())
        print(f"[repeats] {len(drop)} rows dropped: the other recording of the pair failed to load")
        X = np.load(f"{out_dir}/X.npy", mmap_mode="r")
        keep = np.flatnonzero(full.to_numpy())
        np.save(f"{out_dir}/X.npy", np.asarray(X[keep]))
        R = R[full].reset_index(drop=True)
    cov = R[list(ancestries)].astype(float).copy()
    cov["person"] = pd.factorize(R["project_pseudo_id"])[0]
    cov["pairing"] = R["pairing"].to_numpy()
    cov["rep"] = R["rep"].to_numpy()
    cov["gap_days"] = pd.to_numeric(R["gap_days"], errors="coerce").to_numpy()
    cov["n_ecgs"] = pd.to_numeric(R["n_ecgs"], errors="coerce").to_numpy()
    cov["sex"] = R["gender"].to_numpy()
    cov["age"] = pd.to_numeric(R["age"], errors="coerce").to_numpy()
    cov["batch"] = R["batch"].to_numpy()
    cov["round"] = R["round"].to_numpy()
    cov.to_csv(f"{out_dir}/covariates.csv", index=False)
    with open(f"{out_dir}/info.json", "w") as f:
        json.dump({"ancestries": list(ancestries), "eur_label": eur_label, "threshold": threshold,
                   "n_rows": int(len(R)), "n_persons": int(cov["person"].nunique()),
                   "pairings": sorted(R["pairing"].unique().tolist()),
                   "has_dates": bool(pd.to_numeric(R["gap_days"], errors="coerce").notna().any())},
                  f, indent=2)
    print(f"[save:all_genotyped_repeats] X=({len(R)}, 2500, 12) | "
          f"{cov['person'].nunique():,} persons | pairings {sorted(R['pairing'].unique())} "
          f"-> {out_dir}")
    return cov


def build_matched(persons, out_dir, noneur, eur_label, caliper, seed):
    cases = persons[persons["ancestry"].isin(noneur)].copy()
    controls = persons[persons["ancestry"] == eur_label].copy()
    print(f"[pool] non-EUR cases={len(cases)} | EUR controls={len(controls)} | "
          f"by group: {cases['ancestry'].value_counts().to_dict()}")
    pairs = match_1to1(cases, controls, caliper, seed) if len(cases) and len(controls) else []
    if not pairs:      # not fatal: the matched set is only the confounding check (analysis 5)
        print(f"[WARNING] no age/sex-matched pairs (matchable non-EUR={len(cases)}, EUR={len(controls)}). "
              "Analysis 5 will be reported as suppressed; everything else is unaffected. Check the "
              "[sex] line above (unrecognised sex codes) or loosen --caliper.", flush=True)
        return pd.DataFrame(columns=["batch", "ancestry", "project_pseudo_id", "round", "gender",
                                     "age", "y", "pair_id", "role"])
    d = np.array([x for *_, x in pairs])
    print(f"[match] {len(pairs)} pairs | |dage| mean={d.mean():.2f} max={d.max():.2f} yr")
    cols = ["batch", "ancestry", "project_pseudo_id", "round", "gender", "age", "full_ecg_path"]
    rows = []
    for pid, (ci, kj, _) in enumerate(pairs):
        rows.append({**cases.loc[ci, cols].to_dict(), "y": 1, "pair_id": pid, "role": "case"})
        rows.append({**controls.loc[kj, cols].to_dict(), "y": 0, "pair_id": pid, "role": "control"})
    matched = pd.DataFrame(rows)
    X, ok = _stack_ecgs(matched["full_ecg_path"].to_numpy(), matched["project_pseudo_id"].to_numpy())
    bad = set(matched.loc[~ok, "pair_id"]); keep = ok & ~matched["pair_id"].isin(bad).to_numpy()
    X, matched = X[keep], matched[keep].reset_index(drop=True)
    y = matched["y"].to_numpy(np.int64)
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/X.npy", X); np.save(f"{out_dir}/y.npy", y)
    matched.drop(columns=["full_ecg_path"]).to_csv(f"{out_dir}/matched_manifest.csv", index=False)
    print(f"[save] X={X.shape} y={y.shape} (EUR/0={(y==0).sum()}, non-EUR/1={(y==1).sum()}) -> {out_dir}")
    return matched


def build_unmatched(persons, out_dir, noneur, eur_label):
    """All eligible EUR (y=0) + non-EUR (y=1) persons, one ECG each, no matching."""
    t = persons[persons["ancestry"].isin([*noneur, eur_label])].reset_index(drop=True)
    t["y"] = (t["ancestry"] != eur_label).astype(np.int64)
    os.makedirs(out_dir, exist_ok=True)
    ok = _save_ecgs(t["full_ecg_path"].to_numpy(), t["project_pseudo_id"].to_numpy(),
                    f"{out_dir}/X.npy")
    t = t[ok].reset_index(drop=True)
    y = t["y"].to_numpy(np.int64)
    np.save(f"{out_dir}/y.npy", y)
    np.save(f"{out_dir}/groups.npy", t["project_pseudo_id"].to_numpy().astype(str))
    cols = ["batch", "ancestry", "project_pseudo_id", "round", "gender", "age", "y"]
    t[cols].to_csv(f"{out_dir}/manifest.csv", index=False)
    print(f"[save:unmatched] X=({len(t)}, 2500, 12) (EUR/0={(y==0).sum()}, non-EUR/1={(y==1).sum()}, "
          f"prevalence={y.mean():.3f}) -> {out_dir}")
    return t


def build_all_genotyped(persons_all, matched, out_dir, ancestries, eur_label, threshold):
    """Everyone with a linked ECG, any ancestry (incl. ADMIXED), one ECG each, plus the
    covariates `validate` needs to derive every label/subgroup itself. No ids written."""
    t = persons_all.reset_index(drop=True).copy()
    os.makedirs(out_dir, exist_ok=True)
    ok = _save_ecgs(t["full_ecg_path"].to_numpy(), t["project_pseudo_id"].to_numpy(),
                    f"{out_dir}/X.npy")
    t = t[ok].reset_index(drop=True)
    in_m = set(matched["project_pseudo_id"]) if matched is not None else set()
    cov = t[list(ancestries)].astype(float).copy()
    cov["sex"] = t["gender"].to_numpy()
    cov["age"] = pd.to_numeric(t["age"], errors="coerce").to_numpy()
    cov["batch"] = t["batch"].to_numpy()
    cov["round"] = t["round"].to_numpy()
    cov["in_matched"] = t["project_pseudo_id"].isin(in_m).astype(int).to_numpy()
    cov.to_csv(f"{out_dir}/covariates.csv", index=False)
    try:                                       # warn early; the scoring modes refuse outright
        from ancestry import infer as infer_lib
        infer_lib.check_units(np.load(f"{out_dir}/X.npy", mmap_mode="r"))
    except ValueError as e:
        print(f"[WARNING] {e}", flush=True)
    with open(f"{out_dir}/info.json", "w") as f:
        json.dump({"ancestries": list(ancestries), "eur_label": eur_label,
                   "threshold": threshold, "n": int(len(t))}, f, indent=2)
    print(f"[save:all_genotyped] X=({len(t)}, 2500, 12) | "
          f"{t['ancestry'].value_counts().to_dict()} | in_matched={int(cov['in_matched'].sum())} "
          f"-> {out_dir}")


def build_matched_all_rounds(matched, linked, out_dir):
    labels = matched[["project_pseudo_id", "y", "pair_id"]].drop_duplicates("project_pseudo_id")
    exp = linked.merge(labels, on="project_pseudo_id", how="inner").reset_index(drop=True)
    X, ok = _stack_ecgs(exp["full_ecg_path"].to_numpy(), exp["project_pseudo_id"].to_numpy())
    X, exp = X[ok], exp[ok].reset_index(drop=True)
    y = exp["y"].to_numpy(np.int64)
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/X.npy", X); np.save(f"{out_dir}/y.npy", y)
    np.save(f"{out_dir}/groups.npy", exp["project_pseudo_id"].to_numpy().astype(str))
    print(f"[save:all-rounds] X={X.shape} from {exp['project_pseudo_id'].nunique()} persons -> {out_dir}")
    return exp


def build_admixed(persons, linked, out_dir, eur_label, all_rounds):
    if all_rounds:
        adm = linked[linked["ancestry"] == "ADMIXED"].copy()
    else:
        adm = persons[persons["ancestry"] == "ADMIXED"].reset_index(drop=True)
    adm = adm[adm[eur_label].notna()].reset_index(drop=True)
    if len(adm) == 0:
        print("[admixed] none found -> skipping admixed set")
        return
    X, ok = _stack_ecgs(adm["full_ecg_path"].to_numpy(), adm["project_pseudo_id"].to_numpy())
    X, adm = X[ok], adm[ok].reset_index(drop=True)
    eur = adm[eur_label].to_numpy(np.float32)
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/X_admixed.npy", X)
    np.save(f"{out_dir}/rfmix_eur.npy", eur)              # the `admixed` mode reads this
    np.save(f"{out_dir}/groups_admixed.npy", adm["project_pseudo_id"].to_numpy().astype(str))
    print(f"[save] X_admixed={X.shape} rfmix_eur range [{eur.min():.3f},{eur.max():.3f}] "
          f"from {adm['project_pseudo_id'].nunique()} persons -> {out_dir}")


# --------------------------------------------------------------------------- figures
def match_validation_plot(matched, out_png):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    case = matched[matched["y"] == 1]; ctrl = matched[matched["y"] == 0]
    piv = matched.pivot(index="pair_id", columns="role", values="age")
    dage = (piv["case"] - piv["control"]).to_numpy(float)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    lo, hi = matched["age"].min(), matched["age"].max(); bins = np.linspace(lo, hi, 25)
    ax[0].hist(ctrl["age"], bins=bins, alpha=.55, color="#3b7dd8", label=f"EUR (n={len(ctrl)})")
    ax[0].hist(case["age"], bins=bins, alpha=.55, color="#d1495b", label=f"non-EUR (n={len(case)})")
    ax[0].set(xlabel="Age (years)", ylabel="Count", title="Age distribution"); ax[0].legend(frameon=False)
    m = max(1, int(np.ceil(np.abs(dage).max() or 1)))
    ax[1].hist(dage, bins=np.arange(-m - .5, m + 1.5, 1), color="#6c757d")
    ax[1].axvline(0, color="#d1495b", lw=1.2)
    ax[1].set(xlabel="case age - control age", ylabel="Pairs",
              title=f"Paired age diff (mean|d|={np.abs(dage).mean():.2f})")
    fig.suptitle("Age/sex matching validation", fontweight="bold")
    fig.tight_layout(); fig.savefig(out_png, dpi=200, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rfmix", nargs="+", required=True, help="RFMix weighted-sum matrices")
    ap.add_argument("--ecg-meta", nargs="+", required=True, help="ECG metadata parquet(s), one per round")
    ap.add_argument("--linkage", default=None, help="conversion table CSV (barcode->pseudo); optional")
    ap.add_argument("--output", required=True,
                    help="where to write the prepared arrays (X.npy, y.npy, covariates); this directory stays at the site")
    ap.add_argument("--threshold", type=float, default=0.70,
                    help="the ancestry gate: a participant is called a continental ancestry at this RFMix fraction or more (default 0.70, the discovery value)")
    ap.add_argument("--caliper", type=float, default=2.0, help="max age gap (yr); <0 disables")
    ap.add_argument("--ancestries", nargs="+", default=["AFR", "EAS", "SAS", "EUR"],
                    help="the RFMix columns to read")
    ap.add_argument("--eur-label", default="EUR", help="which column is the European reference class")
    ap.add_argument("--sample-col", default="Sample", help="sample-id column of the RFMix table")
    ap.add_argument("--barcode-col", default="Barcode", help="genotype-side id column of --linkage")
    ap.add_argument("--pseudo-col", default="project_pseudo_id",
                    help="participant-side id column of --linkage")
    ap.add_argument("--ecg-id-col", default="ecg_id", help="recording-id column of --ecg-meta")
    ap.add_argument("--patient-id-col", default=None,
                    help="column of the ECG metadata holding the PATIENT id, when it is not the "
                         "ECG id with --ecg-prefix removed. Use it whenever one patient can have "
                         "several ECGs: it is what groups a person's recordings for analysis 8.")
    ap.add_argument("--patient-id-prefix", default="",
                    help="regex prefix stripped from --patient-id-col (default: nothing)")
    ap.add_argument("--date-col", default=None,
                    help="acquisition date/time column; orders repeats and gives the interval "
                         "between them (analysis 8 interval strata). Optional.")
    ap.add_argument("--pairing", default="both", choices=["first_last", "closest", "both"],
                    help="which two recordings of a person analysis 8 compares (default: both)")
    ap.add_argument("--skip-repeats", action="store_true",
                    help="do not write all_genotyped_repeats/ (skips analysis 8)")
    ap.add_argument("--gender-col", default="gender", help="sex column of --ecg-meta")
    ap.add_argument("--age-col", default="age", help="age-at-recording column, in years")
    ap.add_argument("--path-col", default="full_ecg_path", help="waveform-path column")
    ap.add_argument("--ecg-prefix", default="ECG_", help="prefix to strip from the ECG id")
    ap.add_argument("--rfmix-sep", default="\t",
                    help="delimiter of the RFMix table (default tab); `prepare` passes auto")
    ap.add_argument("--seed", type=int, default=0, help="seed for the matching draw")
    ap.add_argument("--no-plots", action="store_true", help="write the tables but draw no plates")
    ap.add_argument("--min-cell", type=int, default=10,
                    help="counts of 1..min_cell-1 people are written as '<min_cell' in every table")
    ap.add_argument("--write-unmatched", action="store_true",
                    help="also write unmatched/ (gated EUR+non-EUR as X.npy/y.npy) for the optional "
                         "stand-alone `metrics`/`ensemble` modes; `validate` does not need it")
    ap.add_argument("--skip-all-genotyped", action="store_true",
                    help="do not write all_genotyped/ (the input of `validate`)")
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    anc = args.ancestries
    noneur = [a for a in anc if a != args.eur_label]
    group_order = [*noneur, args.eur_label, "ADMIXED"]
    batch_ids = list(range(1, len(args.rfmix) + 1))
    round_ids = list(range(1, len(args.ecg_meta) + 1))
    caliper = None if args.caliper is not None and args.caliper < 0 else args.caliper

    rfmix_table = build_rfmix_table(args.rfmix, batch_ids, args.sample_col, anc,
                                    args.threshold, args.rfmix_sep)
    counts = pd.DataFrame({b: {g: int((classify(rfmix_table[rfmix_table.batch == b], anc, args.threshold) == g).sum())
                               for g in group_order} for b in batch_ids}).T
    mask_frame(counts, k=args.min_cell).to_csv(out / "ancestry_counts.csv")
    print("[ancestry counts]\n" + counts.to_string())

    ecg_meta = load_ecg_metadata(args.ecg_meta, round_ids, args.ecg_id_col, args.gender_col,
                                 args.age_col, args.path_col, args.ecg_prefix,
                                 patient_id_col=args.patient_id_col,
                                 patient_id_prefix=args.patient_id_prefix, date_col=args.date_col)
    # link() carries the ancestry-proportion columns through (rfmix_table has them),
    # so `linked` already exposes the per-sample EUR fraction used by the admixed set.
    linked = link(rfmix_table, ecg_meta, args.linkage, args.barcode_col, args.pseudo_col)
    mask_frame(coverage_table(linked, rfmix_table, round_ids), k=args.min_cell).to_csv(
        out / "ecg_coverage.csv")
    mask_frame(ancestry_overlap(linked, rfmix_table, group_order, round_ids), k=args.min_cell).to_csv(
        out / "ancestry_ecg_overlap.csv")

    linked["gender"] = normalise_sex(linked["gender"])                         # once, logged
    persons_all = person_table(linked, round_ids, require_demographics=False)   # everyone
    persons = person_table(linked, round_priority=round_ids)                    # matchable
    first_dir, all_dir, full_dir = out / "first_only", out / "all_rounds", out / "unmatched"

    unmatched = None
    if args.write_unmatched:
        print("\n-- unmatched (gated EUR + non-EUR, one ECG/person; for stand-alone metrics/ensemble) --")
        unmatched = build_unmatched(persons_all, str(full_dir), noneur, args.eur_label)
        build_admixed(persons_all, linked, str(full_dir), args.eur_label, all_rounds=False)
    else:   # still report the full gated N in the flow/balance tables
        unmatched = persons_all[persons_all["ancestry"].isin([*noneur, args.eur_label])].copy()
        unmatched["y"] = (unmatched["ancestry"] != args.eur_label).astype(int)

    print("\n-- variant 1: first-only (1:1 age/sex matched, one ECG/person) --")
    matched = build_matched(persons, str(first_dir), noneur, args.eur_label, caliper, args.seed)
    build_admixed(persons_all, linked, str(first_dir), args.eur_label, all_rounds=False)

    if not args.skip_all_genotyped:
        print("\n-- all_genotyped (everyone incl. admixed, one ECG/person; input of `validate`) --")
        build_all_genotyped(persons_all, matched, str(out / "all_genotyped"), anc,
                            args.eur_label, args.threshold)

    repeats_dir, repeat_cov = out / "all_genotyped_repeats", None
    mask_frame(repeat_coverage(linked, group_order, args.min_cell), k=args.min_cell).to_csv(
        out / "repeat_coverage.csv")
    if not args.skip_repeats:
        print("\n-- all_genotyped_repeats (persons with >= 2 ECGs; input of validate analysis 8) --")
        R, _per = repeat_pairs(linked, round_ids, args.pairing)
        if len(R):
            repeat_cov = build_repeats(R, str(repeats_dir), anc, args.eur_label, args.threshold,
                                       min_persons=max(args.min_cell * 2, 20))
        else:
            print("[repeats] no person has two usable ECGs -> analysis 8 unavailable")

    print("\n-- variant 2: all-rounds (matched persons, every ECG) --")
    if len(matched):
        build_matched_all_rounds(matched, linked, str(all_dir))
    build_admixed(persons_all, linked, str(all_dir), args.eur_label, all_rounds=True)

    # participant flow: where every eligible person goes (aggregate counts only)
    def _n(df, mask=None):
        d = df if mask is None else df[mask]
        return int(d["project_pseudo_id"].nunique()) if "project_pseudo_id" in d else int(len(d))
    is_case = lambda d: d["ancestry"].isin(noneur)
    is_ctrl = lambda d: d["ancestry"] == args.eur_label
    rows = [
        ("Genotyped (RFMix)", int((rfmix_table["ancestry"].isin(noneur)).sum()),
         int((rfmix_table["ancestry"] == args.eur_label).sum()),
         int((rfmix_table["ancestry"] == "ADMIXED").sum())),
        ("... with >=1 linked ECG", _n(persons_all, is_case(persons_all)),
         _n(persons_all, is_ctrl(persons_all)), _n(persons_all, persons_all["ancestry"] == "ADMIXED")),
        ("... and valid sex + age (matchable)", _n(persons, is_case(persons)),
         _n(persons, is_ctrl(persons)), _n(persons, persons["ancestry"] == "ADMIXED")),
    ]
    if unmatched is not None:
        rows.append(("gated (validate analysis 1, PRIMARY)", int((unmatched.y == 1).sum()),
                     int((unmatched.y == 0).sum()), None))
    rows.append(("1:1 matched (validate analysis 5)", int((matched.y == 1).sum()),
                 int((matched.y == 0).sum()), None))
    flow = pd.DataFrame(rows, columns=["step", "non-EUR", "EUR", "ADMIXED"]).set_index("step").astype("Int64")
    mask_frame(flow, k=args.min_cell).to_csv(out / "participant_flow.csv"); print("\n[participant flow]\n" + flow.to_string())

    # match-balance table
    def _col(d):
        if not len(d):
            return [0, "n/a", "n/a"]
        a = pd.to_numeric(d.age, errors="coerce")
        return [len(d), f"{a.mean():.1f} ({a.std(ddof=1):.1f})" if a.notna().sum() > 1 else "n/a",
                f"{(d.gender == 'FEMALE').mean() * 100:.0f}%"]
    case, ctrl = matched[matched.y == 1], matched[matched.y == 0]
    bal = pd.DataFrame({"matched non-EUR (y=1)": _col(case), "matched EUR (y=0)": _col(ctrl)},
                       index=["N", "Age mean(SD)", "Female %"])
    if unmatched is not None:
        bal["gated non-EUR (all)"] = _col(unmatched[unmatched.y == 1])
        bal["gated EUR (all)"] = _col(unmatched[unmatched.y == 0])
    for c in bal.columns:                     # a column of < min_cell people is masked entirely
        if 0 < int(bal.at["N", c]) < args.min_cell:
            bal[c] = [f"<{args.min_cell}", "suppressed", "suppressed"]
    bal.to_csv(out / "match_summary.csv"); print("[match balance]\n" + bal.to_string())

    if not args.no_plots and len(matched):
        match_validation_plot(matched, str(out / "match_validation.png"))
        print(f"[plot] wrote {out}/match_validation.png")

    print(f"\n[done] inputs ready:\n"
          + (f"  {full_dir}/   -> X.npy y.npy groups.npy X_admixed.npy rfmix_eur.npy  (gated, optional)\n"
             if args.write_unmatched else "")
          + f"  {out / 'all_genotyped'}/ -> X.npy covariates.csv info.json  (input of validate)\n"
          f"  {first_dir}/  -> X.npy y.npy  X_admixed.npy rfmix_eur.npy  (1:1 age/sex matched)\n"
          f"  {all_dir}/    -> X.npy y.npy groups.npy  X_admixed.npy rfmix_eur.npy  (all rounds)\n"
          + (f"  {repeats_dir}/ -> X.npy covariates.csv info.json  (analysis 8, intrapatient stability)\n"
             if repeat_cov is not None else "")
          + f"  participant_flow.csv, match_summary.csv, repeat_coverage.csv, match_validation.png "
          f"+ QC tables in {out}\n"
          f"Next: `validate --input {out / 'all_genotyped'}` (analyses 1-7"
          + (", and 8 from all_genotyped_repeats/)." if repeat_cov is not None else ")."))


if __name__ == "__main__":
    main()
