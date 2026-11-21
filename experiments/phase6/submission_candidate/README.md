# CS5489 frozen Weighted BCE candidate

Open the notebooks with this directory as the working directory. Put test images in
test_images/ or set test_dir in test.ipynb. Leave model_path at model.pth when the
checkpoint is in this directory. Run all cells to generate submission.csv. Test
images alone suffice. The CSV follows the TA notebook's ID,Predicted format and
uses native-size, 1-based column-major RLE. Confirm Kaggle's schema separately.

project_report.ipynb displays the archived training and validation results and has an
optional training reproduction cell. Fill in the group name, names and IDs before
submission. The course archive and caches are supplied separately.

model.pth is the Phase 2 epoch 7 checkpoint copied byte for byte. Keep project.py
beside the notebooks. Training reproduction also uses prepare_data.py,
train_candidate.py and split.json. python_environment.txt lists the versions.
SHA256SUMS.txt records the package file checksums.
