# vendor/ — carried over unchanged

Three files from `scripts/lifelines_locus_validation/` (package v28), which ran at a live site and
were debugged there. They are copied byte-for-byte rather than rewritten, so a bug fixed at
Lifelines is not re-introduced here:

| file | was | what it does |
|---|---|---|
| `measure_neurokit.py` | `01_measure_neurokit.py` | NeuroKit2 delineation of every ECG → 60 trait columns per recording (PR, QRS, QT, JT, HR, the two QRS-T angles, and P/R/S/T amplitudes in all twelve leads). Self-contained; needs only `X.npy` and an id column. |
| `gwas_aggregate.py` | `04_gwas_aggregate.py` | Reduces a per-chromosome plink2 scan to the aggregate tables a Manhattan needs: the thinned point cloud, the QQ quantiles with λ_GC, the distance-clumped leads, and the discovery leads looked up in the full scan. The thinning rule is the discovery figure's. |
| `liftover_min.py` | `liftover_min.py` | Point liftOver from a UCSC chain file, no binary needed. Used only if the site's genotypes are GRCh37 and a build-37 ancestry-PGS weight file has to be made on site. |

Do not edit these here. Fix them in the Lifelines package and re-copy, so the two stay identical.
