# CS5489 frozen Weighted BCE candidate

Open the notebooks with this directory as the working directory. Put test images in
test_images/ or set test_dir in test.ipynb. Leave model_path at model.pth when the
checkpoint is in this directory. Run all cells to write submission.csv in the
TA template format (ID,Predicted). Predictions use native-size, 1-based
column-major RLE. The notebook runs on RGB images without test labels.

project_report.ipynb displays the archived training and validation results and has an
optional training reproduction cell. Add the Canvas group name and members to
the submission copies. The course archive and weights are supplied separately.

model.pth is the Phase 2 epoch 7 checkpoint copied byte for byte. Keep project.py
beside the notebooks. Training reproduction also uses prepare_data.py,
train_candidate.py and split.json. python_environment.txt lists the versions.
The [checkpoint SHA-256 list](../../../docs/provenance/local_artifact_SHA256SUMS.txt) identifies the frozen weights.
