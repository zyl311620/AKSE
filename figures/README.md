# Supplied historical visualization scripts

`source_scripts/` contains byte-preserved copies of the four plotting scripts
supplied in the workspace. Their hard-coded aggregate values come from those
files; they were not recomputed from raw model outputs by this implementation.

These scripts import each other from the same directory. Install the `plots`
extra and run, for example:

```bash
python figures/source_scripts/Refusal_graph_AKSE.py
python figures/source_scripts/CCFR_graph_AKSE.py
```

They save PDFs/PNGs alongside their source files according to the original code.
Confirm metric definitions, source provenance, licensing, and agreement with
the final manuscript before using their charts publicly.

For plots based on newly measured results, use `scripts/plot_metrics.py` instead.
