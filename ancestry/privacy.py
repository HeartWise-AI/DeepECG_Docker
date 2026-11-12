"""Small-cell disclosure control shared by every file that may leave the partner's site.

Two rules, applied everywhere:
  * COUNTS  -- any count of people between 1 and MIN_CELL-1 is written as "<MIN_CELL"
               (0 and counts >= MIN_CELL are written as-is).
  * METRICS -- no metric, curve point or bin mean is computed on fewer than `--min-n` people
               per class; such groups/bins are reported as suppressed (see run_validate.py).
"""
from __future__ import annotations

MIN_CELL = 10

# keys whose integer values are counts of people
COUNT_KEYS = {"n", "n_pos", "n_neg", "n_excluded", "n_total", "admixed_n", "tp", "fp", "tn", "fn",
              "n_persons", "n_rows"}
# dicts whose VALUES are all counts of people
COUNT_DICTS = {"n_by_dominant_ancestry", "vote_count_dist", "ensemble_vote_count_dist", "confusion"}
# count grids written as nested lists (analysis 8): masked cell-by-cell where they are built
COUNT_GRIDS = {"score_grid", "delta_histogram"}


def cell(v, k=MIN_CELL):
    """Mask a single count."""
    try:
        iv = int(v)
    except (TypeError, ValueError):
        return v
    if iv != v:
        return v
    return f"<{k}" if 0 < iv < k else iv


def mask(obj, k=MIN_CELL, _in_count_dict=False):
    """Recursively mask counts in a JSON-like object (keys in COUNT_KEYS / COUNT_DICTS)."""
    if isinstance(obj, dict):
        out = {}
        for key, v in obj.items():
            if _in_count_dict or key in COUNT_KEYS:
                out[key] = cell(v, k) if not isinstance(v, (dict, list)) else mask(v, k)
            else:
                out[key] = mask(v, k, _in_count_dict=key in COUNT_DICTS)
        return out
    if isinstance(obj, list):
        return [mask(v, k) for v in obj]
    return obj


def mask_frame(df, cols=None, k=MIN_CELL):
    """Mask integer count cells of a DataFrame (all numeric columns unless `cols` given)."""
    import pandas as pd
    d = df.copy()
    for c in (cols if cols is not None else d.columns):
        if c in d and (pd.api.types.is_numeric_dtype(d[c]) or d[c].dtype == object):
            d[c] = d[c].map(lambda v: cell(v, k) if not (isinstance(v, float) and v != v) else v)
    return d


def small_count_cells(obj, k=MIN_CELL):
    """Return paths of any unmasked count in 1..k-1 (used by selftest T5)."""
    bad = []

    def walk(o, path, in_cd=False):
        if isinstance(o, dict):
            for key, v in o.items():
                p = f"{path}.{key}"
                if (in_cd or key in COUNT_KEYS) and isinstance(v, int) and 0 < v < k:
                    bad.append(p)
                walk(v, p, key in COUNT_DICTS)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
    walk(obj, "")
    return bad


COUNT_TABLES = {"ancestry_counts.csv", "ecg_coverage.csv", "ancestry_ecg_overlap.csv",
                "participant_flow.csv", "match_summary.csv", "repeat_coverage.csv"}
MAX_CURVE_POINTS = 400        # our thinned curves are <= 200; anything longer looks per-person

# Tables that are legitimately longer than MAX_CURVE_POINTS because a row is a VARIANT, not a
# person.  Each is allowed its own bound and its own column whitelist; a file not on this list is
# still held to the per-person-sized rule.  The whitelist is the real control -- a long file is
# only safe because of what its columns are, and listing the name alone would not be enough.
LONG_TABLES = {
    "scan_manhattan.tsv": (600_000, {"CHROM", "CHR", "POS", "BP", "NEGLOG10P", "P", "ID"}),
    "scan_qq.tsv": (100_000, {"EXPECTED", "OBSERVED"}),
    "leads_per_stratum.csv": (2_000, None),
    "variant_coverage.csv": (2_000, None),
    "trait_pgs_deciles.csv": (2_000, None),
    "ancestry_pgs_deciles.csv": (2_000, None),
    "gaps.csv": (5_000, None),
    "label_coverage.csv": (2_000, None),
    "score_histogram.csv": (2_000, None),
}
# suffixes whose bytes are searched for a participant identifier.  A figure is included: a .png
# carries its text as bytes in the file, so a stray id in a title or an axis label is findable.
ID_SEARCH_SUFFIXES = {".json", ".csv", ".tsv", ".txt", ".md", ".pdf", ".svg", ".png", ".html"}
TABLE_SUFFIXES = {".csv": ",", ".tsv": "\t"}


def audit_files(paths, k=MIN_CELL, ids=()):
    """Disclosure audit of files about to leave the site. Returns a list of problems (empty =
    clean): unmasked counts of 1..k-1, over-long arrays (possible per-person values), score
    thresholds, per-person-sized tables, or any of the given participant ids."""
    import json
    from pathlib import Path

    import pandas as pd
    problems = []
    for p in map(Path, paths):
        if p.suffix == ".json":
            obj = json.loads(p.read_text())
            problems += [f"{p.name}{c}: unmasked small count" for c in small_count_cells(obj, k)]

            def walk(o, path):
                if isinstance(o, dict):
                    for key, v in o.items():
                        if key == "reference_ours":          # our own aggregate reference curves
                            continue
                        if "threshold" in key.lower() and isinstance(v, list):
                            problems.append(f"{p.name}{path}.{key}: score thresholds")
                        walk(v, f"{path}.{key}")
                elif isinstance(o, list):
                    if len(o) > MAX_CURVE_POINTS:
                        problems.append(f"{p.name}{path}: list of {len(o)} values")
                    for i, v in enumerate(o):
                        walk(v, f"{path}[{i}]")
            walk(obj, "")
        elif p.suffix in TABLE_SUFFIXES:
            d = pd.read_csv(p, sep=TABLE_SUFFIXES[p.suffix], dtype=str,
                            index_col=0 if p.name in COUNT_TABLES else None)
            limit, allowed = LONG_TABLES.get(p.name, (MAX_CURVE_POINTS, None))
            if len(d) > limit:
                problems.append(f"{p.name}: {len(d)} rows (limit {limit:,}; per-person sized?)")
            if allowed is not None:
                extra = [c for c in d.columns if c.upper() not in allowed]
                if extra:
                    problems.append(f"{p.name}: unexpected column(s) {extra} in a long table")
            cols = list(d.columns) if p.name in COUNT_TABLES else \
                [c for c in d.columns if c in ("n", "n_pos", "n_neg", "n_persons")]
            for c in cols:
                for v in d[c].dropna():
                    if v.isdigit() and 0 < int(v) < k:
                        problems.append(f"{p.name}:{c}={v}: unmasked small count")
        # an identifier column in ANY exported table, whatever its name passed at write time
        if p.suffix in TABLE_SUFFIXES:
            try:
                hdr = pd.read_csv(p, sep=TABLE_SUFFIXES[p.suffix], nrows=0)
                bad = [c for c in hdr.columns if str(c).strip().lower() in FORBIDDEN_COLUMNS]
                if bad:
                    problems.append(f"{p.name}: identifier column(s) {bad}")
            except Exception:
                pass
        if ids and p.suffix in ID_SEARCH_SUFFIXES:
            blob = p.read_bytes()
            if any(i.encode() in blob for i in ids):
                problems.append(f"{p.name}: contains a participant id")
    return problems


# =============================================================================================
# Added for the MGB-MGH genetic + fairness build.  The rules above were written for one model's
# metrics; these cover the three new kinds of output -- variant statistics, per-stratum tables
# and per-finding tables -- and they are enforced where the tables are written, not only audited
# at packaging time.
# =============================================================================================

MIN_STRATUM = 100        # no ancestry stratum is analysed at all below this many people
MIN_PER_CLASS = 20       # no metric is computed for a group with fewer than this per class
MIN_POS_FINDING = 15     # a clinical finding needs this many positives IN BOTH GROUPS
MAX_MANHATTAN_ROWS = 600_000   # the thinned point cloud; a full summary statistic is ~10^7
MAX_TABLE_ROWS = 100_000       # any other exported table

# A variant statistic is an aggregate over the whole stratum, so it is exportable -- but only from
# a stratum large enough that an allele count cannot be read back to a person.  An allele
# frequency in 12 people is 24 chromosomes and is a genotype, not a statistic.
MIN_N_FOR_VARIANT_STATS = MIN_STRATUM

# Columns that must never appear in anything leaving the site, whatever the file.  `IID`/`FID` are
# plink's own spelling, `participant_id` ours, `MRN`/`EMPI`/`PatientID` the hospital's.
FORBIDDEN_COLUMNS = {
    'iid', 'fid', 'participant_id', 'patient_id', 'patientid', 'subject_id', 'sample_id',
    'entity_id', 'ecg_id', 'mrn', 'empi', 'barcode', 'eid', 'npy_path', 'ecg_path', 'path',
    'acquisitiondate', 'acquisition_date', 'date', 'dob', 'birth_date',
}


def check_columns(df, where=''):
    """Raise if a frame about to be written carries an identifier column. Called at write time."""
    bad = [c for c in df.columns if str(c).strip().lower() in FORBIDDEN_COLUMNS]
    if bad:
        raise ValueError(f'{where}: refusing to write identifier column(s) {bad}')
    return df


def write_table(df, path, where=None, max_rows=MAX_TABLE_ROWS, mask_counts=True, **kw):
    """The only way an aggregate table is written in this image.

    Checks the column names, bounds the row count, masks small counts, and writes. A table that
    trips any of the three raises instead of being written, so a privacy mistake stops the run at
    the point it is made rather than being discovered by the audit at packaging time.
    """
    where = where or str(path)
    check_columns(df, where)
    if len(df) > max_rows:
        raise ValueError(f'{where}: {len(df):,} rows exceeds the {max_rows:,} allowed for an '
                         f'aggregate table -- this looks per-person')
    d = mask_frame(df, cols=[c for c in df.columns
                             if str(c).lower() in COUNT_KEYS or str(c).lower().startswith('n_')
                             or str(c).lower() == 'n']) if mask_counts else df
    d.to_csv(path, index=False, **kw)
    return path


def stratum_ok(n, what=''):
    """True when a stratum is large enough to report anything at all."""
    return int(n) >= MIN_STRATUM


SUPPRESSED = 'suppressed (below the minimum group size)'
