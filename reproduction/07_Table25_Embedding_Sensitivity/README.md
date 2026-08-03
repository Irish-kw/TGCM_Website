# Appendix Table XXV — topic-encoder embedding sensitivity

Run `reproduce.ipynb` to evaluate seven topic-encoder checkpoint families, five seeds, and mixture sizes K=2 through K=6.

The notebook is standalone: it downloads the pinned public inference support code when needed and lists the required compact evaluation assets at the top. The full 822.44 GiB CAPture raw CSV archive is not required.

The reported statistic is Macro-F1 as mean and sample standard deviation over five seeds under the manuscript's fixed-slot evaluation (`without_hungarian`). No permutation/Hungarian label remapping is applied.
