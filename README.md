# Pneumonia Detection

This project demonstrates training a convolutional neural network to detect pneumonia from chest X‑ray images.

## Project goals
* Train a deep learning model using TensorFlow/Keras on chest X‑ray data.
* Evaluate the model with cross‑validation and report metrics.
* Export the trained network to a TFLite file for on‑device inference.
* Produce a PDF synthesis summarizing the experiment.

## Dataset
The notebook expects the [Chest X‑ray dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) placed under:

```
chest_Xray/
  train/
  val/
  test/
```

Each subfolder must contain `NORMAL/` and `PNEUMONIA/` image folders.

## Directory layout
```
README.md                – this guide
pneumonet_pro_plus.ipynb – training and export notebook
chest_Xray/              – dataset (not included in repo)
outputs/
  pneumonia_model.tflite – exported model
  pneumonia_summary.pdf  – synthesis report
```

Create the `outputs/` directory before running the export cells.

## Environment setup
1. Create and activate a virtual environment
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies
   ```bash
   pip install tensorflow scikit-learn matplotlib numpy
   ```

TensorFlow 2.x is required for the export code.

## Running `pneumonet_pro_plus.ipynb`
1. Launch Jupyter
   ```bash
   jupyter notebook
   ```
2. Open the notebook and run each cell in order. Training will load the dataset and perform cross‑validation before fitting the final model.
3. After training completes, a cell near the end exports the model to `outputs/pneumonia_model.tflite`.
4. Convert the notebook to PDF via **File → Download as → PDF**. Save it to `outputs/pneumonia_summary.pdf`.

## Expected results
During training accuracy gradually improves toward ~0.90 with validation accuracy up to ~0.94. The ROC‑AUC curve is plotted using scikit‑learn and typically shows an AUC around 0.95. After running all cells you should obtain:

* `outputs/pneumonia_model.tflite` – TensorFlow Lite model (~several MB)
* `outputs/pneumonia_summary.pdf` – PDF report of the notebook

These outputs will appear in the `outputs/` directory.
