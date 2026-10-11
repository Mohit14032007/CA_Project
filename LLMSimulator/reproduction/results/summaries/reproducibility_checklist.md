# Reproducibility Checklist

- [x] **GPU baseline valid:** Yes. (`reproduction/results/csv/gpu_baseline/`)
- [x] **Duplex valid:** Yes. (`reproduction/results/csv/duplex/`)
- [x] **Duplex+PE valid:** Yes. (`reproduction/results/csv/duplex_pe/`)
- [x] **Duplex+PE+ET valid:** Yes. (`reproduction/results/csv/duplex_pe_et/`)
- [x] **Routing trace valid:** Yes. (Traces contain exactly 10,240 rows per trace).
- [x] **Processor trace valid:** Yes. (Successfully parsed and decoupled across GPU and Logic elements).
- [x] **Configuration differences audited:** Yes. (`final_configuration_diff.csv`).
- [x] **Numerical consistency verified:** Yes. (`numerical_consistency.md`).
- [x] **Performance CSVs complete:** Yes.
- [x] **Routing CSVs complete:** Yes.
- [x] **Processor CSVs complete:** Yes.
- [x] **Plots generated:** Yes. (`reproduction/plots/final/`).
- [x] **Figure index generated:** Yes. (`FIGURE_INDEX.md`).
- [x] **README complete:** Yes.
- [x] **Limitations documented:** Yes. (See `final_summary.md` and `FINAL_REPORT.md`).
- [x] **Deviations documented:** Yes. (`paper_deviation_report.md`).
- [x] **Reproduction commands documented:** Yes. (`reproduction/scripts/run_reproduction.sh`).
- [x] **Final summary generated:** Yes. (`FINAL_REPORT.md`).
