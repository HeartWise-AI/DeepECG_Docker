"""DeepECG-Ancestry: external validation of the ECG ancestry score and its genetic signature.

One package, run as `python -m ancestry <mode>`, that takes a partner site's ECGs, genotypes and
ancestry calls and returns aggregate tables only. It is the external-validation layer of
DeepECG_Docker. Its separate model manifest pins the ancestry ensemble, while the shared main
manifest pins the released WCR backbone and findings head. The normal `main.py` task registry is
deliberately unchanged.

Two containers were merged into this package:

  deepecg-ancestry  the validation suite shipped to Lifelines/UGLI and MGB-MGH -- one inference
                    pass, the pre-specified analysis plan, no genotypes required
                    (`validate`, `metrics`, `admixed`, `ensemble`)
  deepecg-mgh       the genetic replication and fairness audit -- the discovery GWAS loci, nine trait
                    polygenic scores, the 77-finding fairness audit, the deprivation arm
                    (`prepare`, `score`, `geno`, `loci`, `pgs`, `scan`, `fairness`, `ses`)

They shared a lineage -- the inference core was byte-identical and the preprocessing had only
diverged -- so this is one pipeline, not two bolted together.

NOTHING PER-PATIENT EVER LEAVES. See `ancestry.privacy`.
"""

__all__ = ['VERSION']

VERSION = '1.0.0'
