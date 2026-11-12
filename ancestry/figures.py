"""Mode `figures` -- draw every plate from the aggregate tables, and compute nothing.

A figure that computes is a figure that can disagree with its own table.  Every number drawn here
is read from results/*/*.csv; if a panel cannot be drawn the plate says why on the axes rather
than being silently omitted, because a missing panel in a package that travels by email is
indistinguishable from a panel that was never asked for.

THE PLATES

  fig_locus_replication     the Fig. 2f analogue.  Our discovery effect on x, the site's on y, both as
                            |beta| on log axes, one point per lead, coloured by locus class, open
                            where the site's P >= 0.05.  The identity line is drawn, and so is the
                            inverse-variance slope through the origin -- the headline number,
                            because one p-value per lead at a quarter of the discovery sample is
                            not the question.
  fig_manhattan_<stratum>   the full scan, one per stratum, with the discovery lead positions
                            ticked along the top so a reader can see at a glance whether the
                            site's signal sits where ours does.  The genome-wide line is drawn
                            whatever the stratum, and the subtitle says when a stratum is too
                            small for the line to mean anything.
  fig_qq                    the QQ plots with lambda_GC, all strata on one plate.  This is the
                            panel that says whether the phenotype behaves like a polygenic trait
                            at this site or like a batch artefact.
  fig_pgs_deciles           the Fig. 2e analogue: mean predicted P(non-European) by oriented
                            decile of each trait PGS inside the European stratum, T amplitude in
                            black, our own discovery curves behind in grey.
  fig_positive_control      each trait score against its OWN measured trait.  Read this one
                            first: it is the gate the panel above depends on.
  fig_ancestry_pgs          the ancestry-score PGS -- the p-value ladder and the decile curve.
  fig_fairness_fpr          the fairness analogue: false-positive gap per 100 patients, one row per
                            finding, measured on true genetic ancestry (blue) and on ancestry
                            predicted from the ECG (gold).  If the two agree, a hospital can audit
                            a vendor model for this without genotypes.
  fig_fairness_auditability one against the other, with the identity line and the fitted slope.
  fig_ses                   the deprivation arm, drawn only when `ses` ran: the ancestry/
                            deprivation coupling first, then every arm against its
                            coupling-matched sham with the equivalence margin shaded, the
                            visibility ceiling, and the threshold sweep.

    python -m ancestry figures --results /out/results [--formats png pdf]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                          # noqa: E402

HERE = Path(__file__).resolve().parent
from ancestry import style as PS                                         # noqa: E402
from ancestry import lib as L                                            # noqa: E402

CLASS_COLOUR = {'published ECG-trait locus': PS.BLUE,
                'cardiac disease locus, no ECG trait': '#B8860B',
                'no cardiac annotation': '#C0392B'}
GENETIC, INFERRED = PS.BLUE, '#DD8452'
SIG = 5e-8


# Every plate is labelled with the site that ran it. `figures` is given the same --run-name
# as `package`, and this is what the labels read: hard-coding one site's name would mislabel
# every other site's plates, which is the one error a figure cannot explain on its own axes.
SITE = 'MGB-MGH'


def save(fig, out, name, formats):
    out.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        f = out / f'{name}.{ext}'
        fig.savefig(f, dpi=600 if ext == 'png' else None, bbox_inches='tight')
        print(f'    {f.name}  ({f.stat().st_size // 1024} kB)')
    plt.close(fig)


def nothing(ax, msg):
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha='center', va='center',
            fontsize=9, color=PS.GREY_DARK, wrap=True)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ------------------------------------------------------------------------------- 1. loci
def fig_locus_replication(res, out, formats, ref):
    f = res / 'loci' / 'leads_per_stratum.csv'
    if not f.exists():
        return
    D = pd.read_csv(f)
    D = D[D['sample'] == (('unrelated') if (D['sample'] == 'unrelated').any() else D['sample'].iloc[0])]
    S = pd.read_csv(res / 'loci' / 'replication_summary.csv')
    strata = [s for s in L.STRATA if s in set(D.stratum)]
    n = len(strata)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.6), squeeze=False)
    for ax, st in zip(axes[0], strata):
        d = D[(D.stratum == st) & (D.status == 'tested')].copy()
        if not len(d):
            nothing(ax, f'{st}: no lead was testable'); continue
        x = d.disc_beta.abs().to_numpy()
        # the site's effect, SIGNED RELATIVE TO OURS: a discordant lead is a negative y, which on
        # a log axis cannot be drawn, so it is drawn at the floor and marked with a down caret.
        y = (d.site_beta * np.sign(d.disc_beta)).to_numpy()
        lo = max(1e-4, np.nanmin(np.abs(y[y > 0])) * 0.5 if (y > 0).any() else 1e-3)
        cls = d.class_plot.fillna('published ECG-trait locus')
        sig = d.site_p < 0.05
        for c, col in CLASS_COLOUR.items():
            m = (cls == c).to_numpy()
            if not m.any():
                continue
            for mm, face in ((m & sig.to_numpy(), col), (m & ~sig.to_numpy(), 'white')):
                if not mm.any():
                    continue
                pos = y[mm] > 0
                # concordant: a circle at its effect size.  Discordant: the site's effect points
                # the other way, which a log axis cannot hold, so it sits at the floor as a down
                # caret -- a different mark, so it is never read as a small positive effect.
                if pos.any():
                    ax.scatter(x[mm][pos], y[mm][pos], s=26, facecolor=face, edgecolor=col,
                               linewidth=1.1, zorder=3, marker='o')
                if (~pos).any():
                    ax.scatter(x[mm][~pos], np.full((~pos).sum(), lo), s=34, facecolor=face,
                               edgecolor=col, linewidth=1.1, zorder=4, marker='v')
        lim = (min(lo, np.nanmin(x) * 0.7), max(np.nanmax(x), np.nanmax(y)) * 1.5)
        ax.plot(lim, lim, ls='--', lw=1.0, color=PS.GREY, zorder=1)
        s = S[(S.stratum == st)].iloc[0] if len(S[S.stratum == st]) else None
        if s is not None and np.isfinite(s.get('slope', np.nan)):
            ax.plot(lim, [s.slope * lim[0], s.slope * lim[1]], lw=1.4, color=PS.INK, zorder=2)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        ax.set_xlabel('Allele effect size in discovery (SD)')
        ax.set_ylabel(f'Allele effect size at {SITE} (SD)')
        ttl = (f'{st}   ({int(s.n_tested)} of {int(s.get("n_leads", s.n_tested))} tested)'
               if s is not None else st)
        sub = ('' if s is None else
               f'{s.pct_concordant:.0f}% concordant, slope {s.slope:.2f} '
               f'[{s.slope_lo:.2f}, {s.slope_hi:.2f}]')
        ax.set_title(ttl + ('\n' + sub if sub else ''), fontsize=9.5, color=PS.INK)
        PS.despine(ax)
    h = [plt.Line2D([], [], marker='o', ls='', mfc=c, mec=c, ms=6, label=k.replace(' locus', ''))
         for k, c in CLASS_COLOUR.items()]
    h.append(plt.Line2D([], [], marker='o', ls='', mfc='white', mec=PS.GREY_DARK, ms=6,
                        label=f'P >= 0.05 at {SITE}'))
    h.append(plt.Line2D([], [], marker='v', ls='', mfc=PS.GREY_DARK, mec=PS.GREY_DARK, ms=6,
                        label='opposite direction (drawn at the floor)'))
    axes[0][0].legend(handles=h, loc='upper left', fontsize=7.4, frameon=False,
                      handlelength=1.0, labelspacing=0.3)
    fig.tight_layout()
    save(fig, out, 'fig_locus_replication', formats)


# ------------------------------------------------------------------------------- 2. manhattan
def fig_manhattan(res, out, formats, ref):
    base = res / 'scan'
    if not base.exists():
        return
    # the lead ticks are POSITIONS, so they must be in the build the scan was run on --
    # otherwise every tick sits a few hundred kb from the peak it is meant to mark and the
    # plate quietly argues against itself
    build = 'grch38'
    gp = res / 'geno' / 'geno_provenance.json'
    if gp.exists():
        try:
            build = L.build_suffix(json.loads(gp.read_text()).get('build', 'grch38'))
        except Exception:                                                # noqa: BLE001
            pass
    leads = pd.read_csv(L.ref_table(ref, None, 'leads', build), sep='\t', dtype={'CHR': str})
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        m = d / 'scan_manhattan.tsv'
        if not m.exists():
            continue
        M = pd.read_csv(m, sep='\t')
        chrcol = 'CHROM' if 'CHROM' in M.columns else 'CHR'
        M[chrcol] = pd.to_numeric(M[chrcol], errors='coerce')
        M = M.dropna(subset=[chrcol, 'POS', 'NEGLOG10P'])
        js = d / 'scan_gwas_summary.json'
        S = json.loads(js.read_text()) if js.exists() else {}
        off, ticks, x0 = {}, [], 0.0
        for c in range(1, 23):
            g = M[M[chrcol] == c]
            w = float(g.POS.max()) if len(g) else 1.0
            off[c] = x0; ticks.append(x0 + w / 2); x0 += w * 1.02
        fig, ax = plt.subplots(figsize=(11, 3.4))
        for c in range(1, 23):
            g = M[M[chrcol] == c]
            if not len(g):
                continue
            ax.scatter(g.POS + off[c], g.NEGLOG10P, s=2.0,
                       color=PS.GREY_DARK if c % 2 else PS.GREY, linewidths=0, rasterized=True)
        ax.axhline(-np.log10(SIG), lw=0.9, color=PS.INK)
        top = max(8.0, float(M.NEGLOG10P.max()) * 1.08)
        for _, r in leads.iterrows():
            c = int(r.CHR) if str(r.CHR).isdigit() else 0
            if c in off:
                ax.plot([r.POS + off[c]], [top * 0.985], marker='|', ms=4, color=PS.BLUE,
                        alpha=0.85, zorder=5)
        ax.set_ylim(0, top)
        ax.set_xticks(ticks); ax.set_xticklabels([str(c) for c in range(1, 23)], fontsize=8)
        ax.set_xlim(-0.01 * x0, x0)
        ax.set_xlabel('Chromosome'); ax.set_ylabel(r'$-\log_{10}(P)$')
        note = S.get('note', '')
        ax.set_title(f'GWAS of the ECG ancestry score at {SITE} -- {d.name}'
                     + (f"   (n = {S.get('n_people', '?'):,}" if isinstance(S.get('n_people'), int)
                        else '   (n = ?')
                     + (f", lambda_GC = {S['lambda_gc']:.3f})" if 'lambda_gc' in S else ')')
                     + ('\n' + note if note else ''), fontsize=9.5, color=PS.INK)
        ax.text(0.995, 0.93, 'ticks: the discovery leads', transform=ax.transAxes,
                ha='right', va='top', fontsize=7.5, color=PS.BLUE)
        PS.despine(ax)
        fig.tight_layout()
        save(fig, out, f'fig_manhattan_{d.name}', formats)


def fig_qq(res, out, formats):
    base = res / 'scan'
    dirs = [p for p in (base.iterdir() if base.exists() else []) if (p / 'scan_qq.tsv').exists()]
    if not dirs:
        return
    fig, axes = plt.subplots(1, len(dirs), figsize=(3.1 * len(dirs), 3.2), squeeze=False)
    for ax, d in zip(axes[0], dirs):
        Q = pd.read_csv(d / 'scan_qq.tsv', sep='\t')
        xc = 'EXPECTED' if 'EXPECTED' in Q.columns else Q.columns[0]
        yc = 'OBSERVED' if 'OBSERVED' in Q.columns else Q.columns[1]
        ax.plot(Q[xc], Q[yc], 'o', ms=2.5, color=PS.BLUE)
        m = float(max(Q[xc].max(), Q[yc].max()))
        ax.plot([0, m], [0, m], ls='--', lw=1.0, color=PS.GREY)
        js = d / 'scan_gwas_summary.json'
        lam = json.loads(js.read_text()).get('lambda_gc') if js.exists() else None
        ax.set_title(f'{d.name}' + (f'   $\\lambda_{{GC}}$ = {lam:.3f}' if lam else ''),
                     fontsize=9.5, color=PS.INK)
        ax.set_xlabel(r'expected $-\log_{10}(P)$'); ax.set_ylabel(r'observed $-\log_{10}(P)$')
        PS.despine(ax)
    fig.tight_layout()
    save(fig, out, 'fig_qq', formats)


# ------------------------------------------------------------------------------- 3. PGS
TRAIT_LABEL = {'PR': 'PR interval', 'QRS': 'QRS duration', 'QT': 'QT interval',
               'JT': 'JT interval', 'HR': 'Heart rate', 'QRSTangle': 'Spatial QRS-T',
               'QRSTangleF': 'Frontal QRS-T', 'QRSamp': 'QRS amplitude', 'Tamp': 'T amplitude'}
PGS_COLOUR = {'QRS': '#66C2A5', 'QRSTangleF': '#FC8D62', 'QRSTangle': '#8DA0CB',
              'HR': '#E78AC3', 'QRSamp': '#8FBF4D', 'QT': '#D9B23C',
              'PR': '#C08552', 'JT': '#9C9C9C'}


def fig_pgs_deciles(res, out, formats, ref, stratum='EUR'):
    f = res / 'pgs' / 'trait_pgs_deciles.csv'
    if not f.exists():
        return
    D = pd.read_csv(f)
    D = D[(D.stratum == stratum) & (D.score == 'trait_pgs') & D.mean_p.notna()]
    if not len(D):
        return
    ref_d = pd.read_csv(ref / 'discovery_pgs_deciles.tsv', sep='\t')
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(9.6, 3.9),
                                  gridspec_kw=dict(width_ratios=[1.35, 1.0], wspace=0.42))
    ends = []
    for t, g in D.groupby('trait'):
        g = g.sort_values('oriented_decile')
        top = t == 'Tamp'
        col = PS.INK if top else PGS_COLOUR.get(t, PS.GREY)
        ax.plot(g.oriented_decile, 100 * g.mean_p, '-', color=col, lw=1.9 if top else 1.3,
                zorder=3 if top else 2)
        ends.append((float(100 * g.mean_p.iloc[-1]), TRAIT_LABEL.get(t, t), top, col))
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.4, 13.9)
    ax.set_xlabel('Decile of ECG-trait PGS (oriented)')
    ax.set_ylabel('Mean predicted P(non-European), %')
    ax.set_title(f'{SITE}, {stratum} stratum', fontsize=9.5, color=PS.INK)
    if ends:
        ylo, yhi = ax.get_ylim()
        gap = (8.0 + 1.6) / 72.0 / (ax.bbox.height / fig.dpi) * (yhi - ylo)
        ends.sort()
        ys = [y for y, _, _, _ in ends]
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + gap)
        drop = max(0.0, ys[-1] - (yhi - 0.6 * gap))
        for (y0, name, top, col), y1 in zip(ends, ys):
            y1 -= drop
            ax.plot([10.05, 10.62], [y0, y1], color=col, lw=0.8, alpha=0.85, zorder=1)
            ax.text(10.75, y1, name, va='center', ha='left', fontsize=8, color=col,
                    fontweight='bold' if top else 'normal')
    PS.despine(ax); ax.spines['bottom'].set_bounds(1, 10)

    # the same curves in discovery, for scale -- the two cohorts have different baselines, so the
    # comparable quantity is the SPAN, which is what the right-hand panel plots.
    sp_site = (D.sort_values('oriented_decile').groupby('trait')
               .apply(lambda g: 100 * (g.mean_p.iloc[-1] - g.mean_p.iloc[0])))
    sp_disc = (ref_d.sort_values('oriented_decile').groupby('trait')
               .apply(lambda g: 100 * (g.mean_p.iloc[-1] - g.mean_p.iloc[0])))
    J = pd.concat([sp_disc.rename('discovery'), sp_site.rename('site')], axis=1).dropna()
    if len(J):
        m = float(np.nanmax(np.abs(J.to_numpy()))) * 1.25
        axr.plot([-m, m], [-m, m], ls='--', lw=1.0, color=PS.GREY)
        axr.axhline(0, lw=0.7, color=PS.GREY); axr.axvline(0, lw=0.7, color=PS.GREY)
        axr.set_xlim(-m, m); axr.set_ylim(-m, m)
        # labels are pushed apart vertically before they are drawn: with nine traits several
        # spans land within a fraction of a point of each other and the names overprint.
        Js = J.sort_values('site')
        lab_y, gap = [], 0.085 * 2 * m
        for v in Js.site.to_numpy():
            lab_y.append(v if not lab_y else max(v, lab_y[-1] + gap))
        drop = max(0.0, lab_y[-1] - (m - 0.5 * gap)) if lab_y else 0.0
        for (t, r), ly in zip(Js.iterrows(), lab_y):
            c = PS.INK if t == 'Tamp' else PGS_COLOUR.get(t, PS.GREY)
            ly -= drop
            axr.scatter([r.discovery], [r.site], s=34, color=c, zorder=3)
            if abs(ly - r.site) > 0.02 * m:
                axr.plot([r.discovery, r.discovery + 0.05 * m], [r.site, ly], lw=0.7,
                         color=c, alpha=0.8, zorder=2)
            axr.text(r.discovery + 0.07 * m, ly, TRAIT_LABEL.get(t, t), fontsize=7.2,
                     va='center', ha='left', color=c, zorder=4)
        r = float(np.corrcoef(J.discovery, J.site)[0, 1]) if len(J) > 2 else np.nan
        axr.set_title(f'Decile span, discovery vs {SITE}   (r = {r:.2f})',
                      fontsize=9.5, color=PS.INK)
        axr.set_xlabel('span in discovery (pp)'); axr.set_ylabel(f'span at {SITE} (pp)')
        PS.despine(axr)
    else:
        nothing(axr, 'no trait could be compared')
    for a_, l in ((ax, 'a'), (axr, 'b')):
        PS.panel(a_, l, dx=-0.13, dy=1.1, size=12)
    save(fig, out, 'fig_pgs_deciles', formats)


def fig_positive_control(res, out, formats):
    f = res / 'pgs' / 'trait_pgs.csv'
    if not f.exists():
        return
    D = pd.read_csv(f)
    if 'arm' not in D or 'beta_sd' not in D:
        return
    D = D[(D.arm == 'own_measured_trait') & D.beta_sd.notna()]
    if not len(D):
        return
    strata = [s for s in L.STRATA if s in set(D.stratum)]
    fig, ax = plt.subplots(figsize=(6.6, 0.32 * len(D) + 1.6))
    y, labs = 0, []
    # Two rulers may be present -- the cart's precomputed panel and the NeuroKit pass over the
    # same waveforms -- and they are NOT interchangeable: they agree on QRS, QT and heart rate
    # and diverge on PR and the amplitudes. Drawing both as the same marker would imply one
    # measurement; a filled marker is the device, an open one is NeuroKit.
    has_src = 'source' in D.columns
    sort_by = ['trait', 'source'] if has_src else ['trait']
    for st in strata:
        g = D[D.stratum == st].sort_values(sort_by)
        for _, r in g.iterrows():
            src = str(r.source) if has_src else 'device'
            col = PS.BLUE if st == 'EUR' else PS.GREY_DARK
            open_marker = src == 'neurokit'
            ax.errorbar(r.beta_sd, y, xerr=1.96 * r.se, fmt='o', ms=5,
                        color=col, ecolor=col,
                        markerfacecolor='none' if open_marker else col,
                        markeredgecolor=col, markeredgewidth=1.2,
                        elinewidth=1.1, capsize=2.4)
            tag = f'  [{src}]' if has_src and len(set(D.source)) > 1 else ''
            labs.append(f'{TRAIT_LABEL.get(r.trait, r.trait)}  ({st}){tag}')
            y += 1
    if has_src and len(set(D.source)) > 1:
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], marker='o', ls='none', color=PS.GREY_DARK,
                                  markerfacecolor=PS.GREY_DARK, label='device panel'),
                           Line2D([], [], marker='o', ls='none', color=PS.GREY_DARK,
                                  markerfacecolor='none', markeredgewidth=1.2,
                                  label='NeuroKit2')],
                  frameon=False, fontsize=8, loc='lower right')
    ax.axvline(0, ls=':', lw=1.0, color=PS.GREY_DARK)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=8)
    ax.set_ylim(-0.6, len(labs) - 0.4)
    ax.set_xlabel('measured trait (SD) per SD of its own PGS')
    ax.set_title('Positive control: each trait score on its own measured ECG trait\n'
                 '(the gate every other PGS panel depends on)', fontsize=9.5, color=PS.INK)
    PS.despine(ax)
    fig.tight_layout()
    save(fig, out, 'fig_positive_control', formats)


def fig_ancestry_pgs(res, out, formats, refj):
    f = res / 'pgs' / 'ancestry_pgs.csv'
    if not f.exists():
        return
    D = pd.read_csv(f)
    D = D[D.beta_sd.notna()] if 'beta_sd' in D else D
    if not len(D):
        return
    dec = res / 'pgs' / 'ancestry_pgs_deciles.csv'
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.5), gridspec_kw=dict(wspace=0.36))
    for st, g in D.groupby('stratum'):
        g = g.sort_values('p_threshold')
        a1.plot(g.p_threshold, 100 * g.delta_r2, '-o', ms=4.5, lw=1.7,
                color=PS.BLUE if st == 'EUR' else PS.GREY_DARK, label=st, mec='white', mew=0.8)
    lo = refj.get('ancestry_pgs', {}).get('loo_delta_r2_pct') or {}
    if lo:
        a1.axhspan(min(lo.values()), max(lo.values()), color=PS.GREY, alpha=0.18, zorder=0)
        a1.text(0.02, 0.96, 'shaded: discovery, leave-one-cohort-out\n'
                            f'({min(lo.values()):.1f}-{max(lo.values()):.1f}%)',
                transform=a1.transAxes, va='top', fontsize=7.4, color=PS.GREY_DARK)
    a1.set_xscale('log'); a1.set_ylim(bottom=0)
    a1.set_xlabel('variants kept at $p$ <'); a1.set_ylabel('incremental $R^2$ (%)')
    a1.set_title('Ancestry-score PGS: threshold ladder', fontsize=9.5, color=PS.INK)
    a1.legend(frameon=False, fontsize=8)
    PS.despine(a1)
    if dec.exists():
        E = pd.read_csv(dec)
        E = E[E.mean_p.notna()]
        for st, g in E.groupby('stratum'):
            g = g.sort_values('decile')
            a2.errorbar(g.decile, 100 * g.mean_p, yerr=100 * 1.96 * g.se, fmt='-o', ms=4.5,
                        lw=1.7, color=PS.BLUE if st == 'EUR' else PS.GREY_DARK, label=st,
                        mec='white', mew=0.8, capsize=2.2, elinewidth=1.0)
        a2.set_xticks(range(1, 11))
        a2.set_xlabel('decile of the ancestry-score PGS')
        a2.set_ylabel('mean P(non-European), %')
        a2.set_title('What the model says, by genotype', fontsize=9.5, color=PS.INK)
        a2.legend(frameon=False, fontsize=8)
        PS.despine(a2)
    else:
        nothing(a2, 'no decile table')
    for a_, l in ((a1, 'a'), (a2, 'b')):
        PS.panel(a_, l, dx=-0.15, dy=1.1, size=12)
    save(fig, out, 'fig_ancestry_pgs', formats)


# ------------------------------------------------------------------------------- 4. fairness
def fig_fairness(res, out, formats, ref):
    f = res / 'fairness' / 'gaps.csv'
    if not f.exists():
        return
    D = pd.read_csv(f)
    arm = 'matched' if (D.arm == 'matched').any() else D.arm.iloc[0]
    Q = D[(D.arm == arm) & (D.metric == 'fpr') & (D.threshold_rule == 'shared 0.5') &
          (D.plate_eligible == 1)]
    if not len(Q):
        Q = D[(D.arm == arm) & (D.metric == 'fpr') & (D.threshold_rule == 'shared 0.5')]
    if not len(Q):
        return
    order = (Q[Q.split == 'genetic'].set_index('finding').gap.sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(order) + 1.9))
    for i, fin in enumerate(order):
        yy = len(order) - 1 - i
        g = Q[(Q.finding == fin) & (Q.split == 'genetic')]
        n = Q[(Q.finding == fin) & (Q.split == 'inferred')]
        if len(g) and len(n):
            ax.plot([g.gap.iloc[0], n.gap.iloc[0]], [yy + 0.17, yy - 0.17], lw=1.0,
                    color=PS.GREY, zorder=1)
        for d, col, dy in ((g, GENETIC, 0.17), (n, INFERRED, -0.17)):
            if not len(d):
                continue
            r = d.iloc[0]
            ax.errorbar(r.gap, yy + dy,
                        xerr=[[max(0, r.gap - r.gap_lo)], [max(0, r.gap_hi - r.gap)]],
                        fmt='o', ms=6, color=col, ecolor=col, elinewidth=1.3, capsize=2.6,
                        capthick=1.1, zorder=3)
    ax.axvline(0, ls=':', lw=1.1, color=PS.GREY_DARK)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(list(order)[::-1], fontsize=9)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel('false-positive gap per 100 patients')
    ax.set_title(f'DeepECG-SSL at {SITE}, audited two ways   ({arm}, threshold 0.5)',
                 fontsize=9.5, color=PS.INK)
    h = [plt.Line2D([], [], marker='o', ls='', color=GENETIC, ms=7, label='true genetic ancestry'),
         plt.Line2D([], [], marker='o', ls='', color=INFERRED, ms=7,
                    label='ancestry predicted from the ECG')]
    ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, -0.12 - 0.5 / len(order)),
              ncol=2, frameon=False, fontsize=9)
    PS.despine(ax)
    fig.tight_layout()
    save(fig, out, 'fig_fairness_fpr', formats)

    # one against the other
    js = res / 'fairness' / 'auditability.json'
    A = json.loads(js.read_text()) if js.exists() else {}
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), gridspec_kw=dict(wspace=0.34))
    for ax, metric in zip(axes, ('fpr', 'fnr')):
        q = D[(D.arm == arm) & (D.metric == metric) & (D.threshold_rule == 'shared 0.5')]
        g = q[q.split == 'genetic'].set_index('finding').gap
        n = q[q.split == 'inferred'].set_index('finding').gap
        J = pd.concat([g.rename('g'), n.rename('n')], axis=1).dropna()
        if len(J) < 3:
            nothing(ax, f'too few findings for the {metric.upper()} panel'); continue
        ax.scatter(J.g, J.n, s=30, color=PS.BLUE if metric == 'fpr' else PS.GREY_DARK, zorder=3)
        m = float(np.nanmax(np.abs(J.to_numpy()))) * 1.2
        ax.plot([-m, m], [-m, m], ls='--', lw=1.0, color=PS.GREY)
        k = A.get('auditability', {}).get(f'{arm}_{metric}', {})
        if k.get('slope_through_origin') is not None:
            s = k['slope_through_origin']
            ax.plot([-m, m], [-m * s, m * s], lw=1.4, color=PS.INK)
            ax.set_title(f'{metric.upper()} gap   slope {s:.2f}, r {k.get("pearson_r", np.nan):.2f}'
                         f'\n{k.get("n_same_sign")}/{k.get("n_findings")} agree in sign',
                         fontsize=9.5, color=PS.INK)
        ax.axhline(0, lw=0.7, color=PS.GREY); ax.axvline(0, lw=0.7, color=PS.GREY)
        ax.set_xlim(-m, m); ax.set_ylim(-m, m)
        ax.set_xlabel('gap on true genetic ancestry')
        ax.set_ylabel('gap on ancestry predicted from the ECG')
        PS.despine(ax)
    for a_, l in zip(axes, 'ab'):
        PS.panel(a_, l, dx=-0.16, dy=1.12, size=12)
    save(fig, out, 'fig_fairness_auditability', formats)


# --------------------------------------------------------------------------- 9. deprivation
def fig_ses(res, out, formats):
    """The socioeconomic arm: the coupling, the four arms against their shams, the cut grid.

    Panel a is drawn first and largest on purpose. Wherever deprivation is nearly an ancestry
    proxy -- which at a US hospital it is -- the arms in panel b cannot be read without it, and a
    reader who sees only the forest will over-read the erasure bar.
    """
    d = res / 'ses'
    q = d / 'ses_quad.csv'
    if not q.exists():
        return
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.2))
    (a1, a2), (a3, a4) = axes
    Q = pd.read_csv(q)

    # (a) coupling: where each stratum sits on the deprivation axis
    try:
        C = pd.read_csv(d / 'ses_coupling.csv')
        C = C[C.n.astype(str).str.replace('<', '').str.isdigit() & C.mean_decile.notna()]
        cols = PS.ANCESTRY_ACCESSIBLE
        for i, r in enumerate(C.itertuples()):
            a1.barh(i, float(r.mean_decile), color=cols.get(r.stratum, PS.GREY), height=0.62)
            lab = ('' if r.stratum == 'EUR' else f'  SMD {float(r.smd_vs_eur):+.2f}')
            a1.text(float(r.mean_decile) + 0.12, i, f'{float(r.mean_decile):.1f}{lab}',
                    va='center', fontsize=9, color=PS.INK)
        a1.set_yticks(range(len(C))); a1.set_yticklabels(C.stratum)
        a1.set_xlim(0, 11.6); a1.set_xlabel('mean deprivation decile (10 = most deprived)')
        a1.axvline(5.5, color=PS.GREY_DARK, ls=':', lw=1)
        a1.set_title('a  deprivation by ancestry stratum', loc='left', fontsize=11)
        PS.despine(a1)
    except Exception as e:                                               # noqa: BLE001
        nothing(a1, f'coupling table unreadable\n{e}')

    # (b) the arms, with the equivalence margin shaded and the shams as open markers
    try:
        want = [('B_matched', 'cost of matching', 'B  matching'),
                ('B_matched', 'coupling-matched sham', 'B  sham (coupled)'),
                ('C_halves', 'difference', 'C  deprived - not'),
                ('D_erasure', 'cost of the erasure', 'D  erasure'),
                ('D_erasure', 'coupling-matched sham', 'D  sham (coupled)'),
                ('D_control', 'cost of erasing AGE', 'D  erase age (control)')]
        ys, labs = [], []
        for i, (arm, pre, lab) in enumerate(want):
            r = Q[(Q.arm == arm) & (Q.quantity.str.startswith(pre))]
            if not len(r) or pd.isna(r.estimate.iloc[0]):
                continue
            y = len(ys)
            e = float(r.estimate.iloc[0])
            lo = r.ci_lo.iloc[0]; hi = r.ci_hi.iloc[0]
            sham = 'sham' in lab or 'control' in lab
            if pd.notna(lo) and pd.notna(hi):
                a2.plot([float(lo), float(hi)], [y, y], color=PS.INK, lw=1.2, zorder=2)
            a2.plot([e], [y], 'o', ms=8, zorder=3, color='white' if sham else PS.BLUE,
                    mec=PS.GREY_DARK if sham else PS.BLUE, mew=1.4)
            ys.append(e); labs.append(lab)
        mgn = Q.tost_margin.dropna()
        m = float(mgn.iloc[0]) if len(mgn) else 0.02
        a2.axvspan(-m, m, color=PS.GREY, alpha=0.18, zorder=0,
                   label=f'equivalence margin  ${{\\pm}}${m:g}')
        a2.axvline(0, color=PS.INK, lw=0.9, zorder=1)
        a2.set_yticks(range(len(labs))); a2.set_yticklabels(labs, fontsize=9)
        a2.invert_yaxis()
        a2.set_xlabel('change in ancestry AUROC')
        a2.set_title('b  what adjusting for deprivation costs', loc='left', fontsize=11)
        a2.legend(frameon=False, fontsize=8.5, loc='upper left', bbox_to_anchor=(0.0, -0.13))
        PS.despine(a2, keep_left=True)
    except Exception as e:                                               # noqa: BLE001
        nothing(a2, f'arm table unreadable\n{e}')

    # (c) the ceiling: how well anything predicts deprivation at all
    try:
        A = Q[Q.arm == 'A_visibility'].dropna(subset=['estimate'])
        if not len(A):
            raise ValueError('arm A did not run')
        xs = np.arange(len(A))
        a3.bar(xs, A.estimate.astype(float), color=[PS.BLUE, PS.GREEN][:len(A)], width=0.55)
        for x, r in zip(xs, A.itertuples()):
            if pd.notna(r.ci_lo):
                a3.plot([x, x], [float(r.ci_lo), float(r.ci_hi)], color=PS.INK, lw=1.2)
            a3.text(x, float(r.estimate) + 0.015, f'{float(r.estimate):.3f}', ha='center',
                    fontsize=9.5)
        a3.set_xticks(xs)
        a3.set_xticklabels([q.replace('AUROC for deprived ', '').strip('()')
                            for q in A.quantity], fontsize=9)
        PS.chance_line(a3)
        a3.set_ylim(0.4, 1.0); a3.set_ylabel('AUROC for being deprived')
        a3.set_title('c  can the ECG see deprivation at all?', loc='left', fontsize=11)
        PS.despine(a3)
    except Exception as e:                                               # noqa: BLE001
        nothing(a3, f'arm A not available\n{e}')

    # (d) does anything depend on where the cut is drawn
    try:
        G = pd.read_csv(d / 'ses_cut_grid.csv')
        # the line is the decile sweep only; the named cuts (median, mean, ...) are rows in the
        # same table and would otherwise plot as duplicate points off the sweep.
        sweep = G[G.basis == 'decile sweep'] if 'basis' in G.columns else G
        for col, c, lab in (('auroc_deprived', PS.RUST, 'deprived half'),
                            ('auroc_not_deprived', PS.BLUE, 'not deprived')):
            m = sweep[col].notna()
            a4.plot(sweep.pctile[m], sweep[col][m], 'o-', color=c, ms=5, lw=1.6, label=lab)
        # where the median and the mean fall on that axis: if the two verdicts differ, they
        # differ between these two lines.
        if 'basis' in G.columns:
            for spec, ls, lab in (('cohort-median', '--', 'median'),
                                  ('cohort-mean', ':', 'mean')):
                r = G[G.basis == spec]
                if len(r) and pd.notna(r.pctile.iloc[0]):
                    x = float(r.pctile.iloc[0])
                    a4.axvline(x, color='0.45', ls=ls, lw=1.0, zorder=0)
                    a4.annotate(lab, xy=(x, 0.995), xycoords=('data', 'axes fraction'),
                                ha='center', va='top', fontsize=7.5, color='0.35')
        a4.set_xlabel('cut, percentile of the site\'s deprivation distribution')
        a4.set_ylabel('ancestry AUROC within half')
        a4.set_title('d  does the threshold matter?', loc='left', fontsize=11)
        a4.legend(frameon=False, fontsize=9)
        PS.despine(a4)
    except Exception as e:                                               # noqa: BLE001
        nothing(a4, f'cut grid unreadable\n{e}')

    fig.tight_layout()
    save(fig, out, 'fig_ses', formats)


def main():
    global SITE
    ap = argparse.ArgumentParser(prog='python -m ancestry figures', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=True,
                    help='the aggregate-only output directory; `package` zips exactly this')
    ap.add_argument('--out', default=None,
                    help='where to write (default: beside --results)')
    ap.add_argument('--ref', default=str(L.ref_dir()),
                    help='the frozen discovery tables (default: $REFERENCE_DIR, else the '
                         'reference/ folder of the pinned weights snapshot)')
    ap.add_argument('--formats', nargs='+', default=['png', 'pdf'],
                    help='image formats to write for every plate')
    ap.add_argument('--stratum', default='EUR',
                    help='the stratum the within-group panels are drawn for (default EUR)')
    ap.add_argument('--run-name', default=SITE,
                    help='the site label drawn on every plate; pass the same value as `package`')
    a = ap.parse_args()
    SITE = a.run_name
    res = Path(a.results); out = Path(a.out) if a.out else res / 'figures'
    ref = Path(a.ref)
    refj = json.loads((ref / 'discovery_reference.json').read_text())
    PS.use_paper_style()
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
    L.banner('figures')
    PLATES = ((fig_locus_replication, (res, out, a.formats, ref)),
              (fig_manhattan, (res, out, a.formats, ref)),
              (fig_qq, (res, out, a.formats)),
              (fig_pgs_deciles, (res, out, a.formats, ref, a.stratum)),
              (fig_positive_control, (res, out, a.formats)),
              (fig_ancestry_pgs, (res, out, a.formats, refj)),
              (fig_fairness, (res, out, a.formats, ref)),
              (fig_ses, (res, out, a.formats)))
    WHY = {
        'fig_locus_replication': '`loci`', 'fig_manhattan': '`scan`', 'fig_qq': '`scan`',
        'fig_pgs_deciles': '`pgs`', 'fig_positive_control': '`pgs` (needs --measurements '
                                                            'or `traits`)',
        'fig_ancestry_pgs': '`pgs` (GRCh38 only)', 'fig_fairness': '`fairness`',
        'fig_ses': '`ses`',
    }
    drew, skipped = [], []
    for fn, args in PLATES:
        before = {q.name for q in out.glob('*')} if out.exists() else set()
        try:
            fn(*args)
        except Exception as e:
            print(f'    {fn.__name__}: NOT DRAWN ({type(e).__name__}: {e})')
            skipped.append((fn.__name__, f'{type(e).__name__}: {e}'))
            continue
        after = {q.name for q in out.glob('*')} if out.exists() else set()
        # The plate functions return early and silently when their input is absent, which is
        # correct behaviour but invisible. Comparing the directory before and after is how this
        # loop can tell "nothing to draw" from "drew something", without each function having to
        # report it.
        if after - before:
            drew.append(fn.__name__)
        else:
            skipped.append((fn.__name__, f'no input; produced by {WHY.get(fn.__name__, "?")}'))
    # Saying "wrote <dir>/" after drawing nothing reads as success. A site that ran only some
    # of the arms needs to be told the difference between "this figure has no inputs" and
    # "this figure failed", and a site that ran everything needs to see a count it can check.
    if drew:
        print(f'\n  wrote {len(drew)} plate(s) to {out}/')
    else:
        print(f'\n  NO figures were drawn; {out}/ is empty.')
    for name, why in skipped:
        print(f'    not drawn: {name:24s} {why}')
    if skipped and not drew:
        print('\n  If you have not run those stages yet, this is expected. If you have, the '
              'reason above names the file each plate looked for.')


if __name__ == '__main__':
    main()
