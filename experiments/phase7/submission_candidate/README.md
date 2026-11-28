# CS5489 Phase 7 Frozen Submission Candidate (`ConvNeXt-Small Detail-UNet`)

- **Native 500-image validation micro-IoU:** **81.01%** (mean image IoU **71.37%**, micro-Dice **89.51%**, Precision **94.66%**, Recall **84.89%**, PR-AUC **94.60%**).
- **Offline TA inference:** Place `model.pth` (SHA-256 recorded in `docs/provenance/local_artifact_SHA256SUMS.txt`) beside `test.ipynb` and `models.py`. Update `test_dir` in cell 1 of `test.ipynb` and Run All to export `submission.csv` (`ID,Predicted`, 1-based column-major RLE).
- **Project report:** `project_report.ipynb` contains full executed outputs, Phase 1–7 comparisons, ablations, failure analysis, and an optional `RETRAIN` entrypoint.
- **Before Canvas upload:** Fill in your Canvas group member names in the header cells of `project_report.ipynb` and `test.ipynb`.
