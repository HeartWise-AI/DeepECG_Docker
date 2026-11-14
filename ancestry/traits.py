"""Mode `traits` -- measure the ECGs, so the trait polygenic scores have something to be right about.

The nine trait PGSs are only meaningful at a site if each one predicts its own measured trait
there.  MGB-MGH's cart measurements may be available in the EHR, in which case pass them straight
to `pgs --measurements` and skip this mode.  Where they are not, the traits are measured from the
same waveforms the models are scored on: one NeuroKit2 delineation per recording, fiducials taken
on lead II and read out in all twelve leads, every per-beat quantity aggregated by median.

This is a DIFFERENT instrument from a cart's measurement module and the package says so wherever
the numbers appear: in our own hands NeuroKit2 and a GE/Philips panel correlate 0.95 on R
amplitude in lead I and 0.59 on PR.  The positive control is therefore judged on sign and on
clearing t = 3, not on reproducing a discovery effect size.

    python -m ancestry traits --work /out/work [--jobs 16] [--shard 1/8] [--merge 'work/traits_*.csv']

The measurement work is done by vendor/measure_neurokit.py, byte-identical to the script that ran
at Lifelines.  Writes work/measurements.csv, which CARRIES IDS and stays at the site.
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
from ancestry import lib as L                                            # noqa: E402


def main():
    ap = argparse.ArgumentParser(prog='python -m ancestry traits', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', required=True,
                    help="the site's working directory: intermediates, per-person tables and "
                         'arrays. It stays at the site and is never packaged')
    ap.add_argument('--prepared', default=None,
                    help='the prepared/ directory (default: the one `prepare` recorded in work/)')
    ap.add_argument('--jobs', type=int, default=8,
                    help='worker processes for the CPU measurement pass')
    ap.add_argument('--fs', type=int, default=250,
                    help='sampling rate of the prepared arrays, in Hz (default 250)')
    ap.add_argument('--units', default='auto', choices=['auto', 'mV', 'uV'],
                    help="amplitude units of the arrays; 'auto' infers them and refuses a "
                         'cohort that looks like neither')
    ap.add_argument('--shard', default=None, metavar='I/N',
                    help='run shard I of N, to split the pass across machines (e.g. 3/16)')
    ap.add_argument('--merge', nargs='+', default=None,
                    help='merge the per-shard outputs into work/measurements.csv')
    ap.add_argument('--limit', type=int, default=0,
                    help='stop after this many ECGs; for a timing check, not for a result')
    a, extra = ap.parse_known_args()

    work = Path(a.work)
    R = pd.read_csv(work / 'roster.csv', dtype={'participant_id': str})
    ids = work / 'ids.csv'
    pd.DataFrame({'participant_id': R.participant_id}).to_csv(ids, index=False)
    pdir = Path(a.prepared) if a.prepared else \
        Path((work / 'prepared_dir.txt').read_text().strip())
    out = (work / f'traits_{a.shard.replace("/", "of")}.csv') if a.shard else \
        (work / 'measurements.csv')

    L.banner('traits: NeuroKit2 measurements' + (f'  (shard {a.shard})' if a.shard else ''))
    argv = ['measure_neurokit.py', '--input', str(pdir / 'all_genotyped'), '--ids', str(ids),
            '--output', str(out), '--jobs', str(a.jobs), '--fs', str(a.fs),
            '--units', a.units] + extra
    if a.limit:
        argv += ['--limit', str(a.limit)]
    if a.shard:
        argv += ['--shard', a.shard]
    if a.merge:
        argv = ['measure_neurokit.py', '--input', str(pdir / 'all_genotyped'), '--ids', str(ids),
                '--merge'] + list(a.merge) + ['--output', str(work / 'measurements.csv')]
    sys.argv = argv
    runpy.run_path(str(HERE / 'vendor' / 'measure_neurokit.py'), run_name='__main__')
    print(f'\n  wrote {out}   CARRIES IDS, STAYS HERE')


if __name__ == '__main__':
    main()
