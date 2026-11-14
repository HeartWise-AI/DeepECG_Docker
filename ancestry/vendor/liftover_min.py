"""Lift GRCh38 positions to GRCh37 (or back) from a UCSC chain file, without liftOver installed.

The UGLI imputation manifests are GRCh37 (HRC r1.1), our reference positions are GRCh38, so the
The discovery leads and the score variants have to exist in both builds before anything can be matched.  A
chain file is a plain text alignment, so a point lift is a short walk over its blocks -- no binary,
no network, nothing to install inside the container.

Chain format: a header line
    chain score tName tSize tStrand tStart tEnd qName qSize qStrand qStart qEnd id
followed by aligned blocks `size dt dq` (and a final bare `size`).  All coordinates are 0-based
half-open; ours are 1-based, hence the -1/+1.  A point is lifted only if it falls INSIDE an aligned
block -- a position in a gap has no counterpart and is reported unmapped rather than nudged to the
nearest base.  Where several chains cover a point (segmental duplication, an alt locus) the
highest-scoring one wins and the disagreement is counted, which is the one case where a lift is
worth distrusting.

    python liftover_min.py hg38ToHg19.over.chain.gz in.tsv out.tsv --chr-col CHR --pos-col POS
"""
from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path


def load_chains(path, chroms=None):
    """{tName: [(score, tStart, tEnd, qName, qSize, qStrand, blocks)]} for the chroms we need."""
    op = gzip.open if str(path).endswith('.gz') else open
    chains, cur = {}, None
    with op(path, 'rt') as fh:
        for line in fh:
            if line.startswith('chain'):
                p = line.split()
                score, t_name, t_start = int(p[1]), p[2], int(p[5])
                q_name, q_size, q_strand, q_start = p[7], int(p[8]), p[9], int(p[10])
                t_short = t_name[3:] if t_name.startswith('chr') else t_name
                if chroms is not None and t_short not in chroms:
                    cur = None
                    continue
                cur = [score, t_start, int(p[6]), q_name, q_size, q_strand, q_start, []]
                chains.setdefault(t_short, []).append(cur)
            elif cur is not None and line.strip():
                p = line.split()
                cur[7].append((int(p[0]), int(p[1]) if len(p) > 1 else 0,
                               int(p[2]) if len(p) > 2 else 0))
            elif not line.strip():
                cur = None
    for v in chains.values():
        v.sort(key=lambda c: -c[0])                     # best-scoring chain first
    return chains


def lift(chains, chrom, pos):
    """(qChrom, qPos1, n_alternatives, qStrand); pos is 1-based."""
    p0 = pos - 1
    hits = []
    for score, t_start, t_end, q_name, q_size, q_strand, q_start, blocks in chains.get(chrom, []):
        if not (t_start <= p0 < t_end):
            continue
        t, q = t_start, q_start
        for size, dt, dq in blocks:
            if t <= p0 < t + size:
                r = q + (p0 - t)
                if q_strand == '-':
                    r = q_size - 1 - r
                hits.append((score, q_name[3:] if q_name.startswith('chr') else q_name,
                             r + 1, q_strand))
                break
            t += size + dt
            q += size + dq
    if not hits:
        return None, None, 0, None
    hits.sort(key=lambda h: -h[0])
    distinct = {(h[1], h[2], h[3]) for h in hits}
    return hits[0][1], hits[0][2], len(distinct) - 1, hits[0][3]


def reverse_complement(allele):
    """Reverse-complement a DNA allele; refuse symbolic alleles that cannot be oriented."""
    allele = str(allele).upper()
    if not allele or set(allele) - set('ACGTN'):
        raise SystemExit(f'cannot reverse-complement non-DNA allele {allele!r}')
    return allele.translate(str.maketrans('ACGTN', 'TGCAN'))[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('chain')
    ap.add_argument('infile')
    ap.add_argument('outfile')
    ap.add_argument('--chr-col', default='CHR')
    ap.add_argument('--pos-col', default='POS')
    ap.add_argument('--allele-cols', nargs='*', default=[],
                    help='allele columns to reverse-complement on negative-strand mappings')
    ap.add_argument('--sep', default='\t')
    a = ap.parse_args()

    rows = Path(a.infile).read_text().rstrip('\n').split('\n')
    head = rows[0].split(a.sep)
    ci, pi = head.index(a.chr_col), head.index(a.pos_col)
    try:
        allele_indices = [head.index(column) for column in a.allele_cols]
    except ValueError as e:
        raise SystemExit(f'allele column not found: {e}') from e
    want = {r.split(a.sep)[ci].lower().replace('chr', '') for r in rows[1:]}
    chains = load_chains(a.chain, want)
    print(f'chains loaded for {len(chains)} chromosomes')

    out = [a.sep.join(head + ['CHR37', 'POS37'])]
    n_ok = n_amb = n_cross = 0
    for r in rows[1:]:
        f = r.split(a.sep)
        c = f[ci].lower().replace('chr', '')
        qc, qp, alt, strand = lift(chains, c, int(f[pi]))
        if qc is None:
            out.append(a.sep.join(f + ['NA', 'NA']))
            continue
        n_ok += 1
        n_amb += alt > 0
        n_cross += qc != c
        if strand == '-':
            for i in allele_indices:
                f[i] = reverse_complement(f[i])
        out.append(a.sep.join(f + [qc, str(qp)]))
    Path(a.outfile).write_text('\n'.join(out) + '\n')
    n = len(rows) - 1
    print(f'lifted {n_ok}/{n}; {n - n_ok} fell in a gap and are NA')
    print(f'  {n_amb} mapped by more than one chain (best score kept)')
    print(f'  {n_cross} changed chromosome'
          + ('  <-- inspect these before use' if n_cross else ''))
    if n_ok < n:
        print('  an unmapped position cannot be tested in the other build; it is not a failure of\n'
              '  the data, it is a region the two assemblies do not align')


if __name__ == '__main__':
    main()
