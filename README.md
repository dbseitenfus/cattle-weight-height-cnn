# cattle-weight-height-cnn

CNN-based regression that estimates a cow's **height (cm)** and **weight (kg)** from a single side-view image.
Developed for Assignment #2 of *Deep Learning* (UFSM00993, Federal University of Santa Maria, 2026).

## Overview

- **Data:** 513 annotated 256×256 RGB images (`dataset_8.zip`). The first 50 rows of `annotations.csv` are used for validation and the rest for training.
- **Preprocessing:** pixels scaled to [0, 1]; targets standardized (z-score) using training-set statistics.
- **Models** (all under 5M parameters):
  - **A**: residual CNN trained from scratch (~1.3M parameters)
  - **B**: ImageNet-pretrained EfficientNetB0 trained in two phases (frozen backbone, then fine-tuning) with two separate regression heads, one for height and one for weight (~4.4M parameters)
  - **C**: same backbone as B with a single `Dense(2)` output, to compare against the two-head setup
- **Training:** Huber loss, Adam, `ReduceLROnPlateau`, early stopping, data augmentation that preserves scale, dropout, and L2.
- **Hyperparameter search:** KerasTuner random search over learning rate, dropout, head width, L2, and loss function.
- **Evaluation:** MAE, RMSE, and MAPE reported separately for height and weight, compared against a baseline that always predicts the mean.

## Usage

Open `cattle_weight_height_cnn.ipynb` in Google Colab with a GPU runtime and run all cells. The notebook text is in Portuguese.

To generate predictions for a new dataset in the same format, set `URL_TESTE` and run the last section. It writes `predicoes.csv` with one `id,height,weight` row per image and no header.

## Requirements

TensorFlow/Keras 3, scikit-image, matplotlib, pandas, keras-tuner (all available or installed in Colab).
