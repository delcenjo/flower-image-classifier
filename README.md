# Flower Image Classifier

![CI](https://github.com/delcenjo/flower-image-classifier/actions/workflows/ci.yml/badge.svg)
[![Live demo](https://img.shields.io/badge/Live_demo-Spaces-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/spaces/delcenjo/flower-classifier-demo)
[![Model](https://img.shields.io/badge/Model-Hub-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/delcenjo/flower-image-classifier)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/delcenjo/flower-image-classifier/blob/main/notebooks/quickstart.ipynb)

This is a small PyTorch model that looks at a photo of a flower and tells you
which of five species it is: daisy, dandelion, roses, sunflowers, or tulips.
It is built with transfer learning on a ResNet-18 backbone, so it trains in a
few minutes on a laptop CPU and still gets reasonable accuracy.

The easiest way to see it in action is the
[demo on Hugging Face Spaces](https://huggingface.co/spaces/delcenjo/flower-classifier-demo):
drop in a flower photo and it returns the predicted species with confidence
scores. The trained weights are also on the
[Hugging Face Hub](https://huggingface.co/delcenjo/flower-image-classifier) if
you just want the checkpoint.

## How well does it do?

On a held-out test set the model lands at **0.77 accuracy**. For five roughly
balanced classes, random guessing would sit around 0.20, so the model is clearly
learning something useful. Here is the per-class breakdown:

| Class      | Precision | Recall | F1   |
| ---------- | --------- | ------ | ---- |
| daisy      | 0.76      | 0.65   | 0.70 |
| dandelion  | 0.70      | 0.87   | 0.78 |
| roses      | 0.65      | 0.71   | 0.68 |
| sunflowers | 0.93      | 0.84   | 0.88 |
| tulips     | 0.79      | 0.79   | 0.79 |

| Confusion matrix | Sample predictions |
| --- | --- |
| ![Confusion matrix](reports/figures/confusion_matrix.png) | ![Sample predictions](reports/figures/sample_predictions.png) |

Sunflowers are by far the easiest to spot. The errors that do happen are mostly
roses and tulips getting mixed up with each other, which is not surprising once
you look at a few of those photos side by side.

## How it was trained

The dataset is the TensorFlow flower_photos collection, loaded straight off disk
with `ImageFolder`. To keep CPU training quick, images are capped at 250 per
class (see `MAX_PER_CLASS` in `config.py`) and resized down to 128px. Training
images get a random horizontal flip and a small rotation; everything is
normalised with the usual ImageNet mean and standard deviation. The data is then
shuffled and split 70/15/15 into train, validation, and test.

For the model itself, the trick is to lean on a network that already knows how to
see. A ResNet-18 pretrained on ImageNet has its convolutional layers frozen, and
only the final fully connected layer is swapped out for a fresh five-way head.
That head is the only thing that gets trained: ten epochs with Adam and
cross-entropy loss, checkpointing whichever weights score best on validation.

Across those ten epochs validation accuracy climbs from about 0.59 up to roughly
0.79, and the test number quoted above comes from the saved best checkpoint.

## Running it yourself

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

Then the usual loop, from fetching data to predicting on a single image:

```bash
python scripts/download_data.py      # download and unpack the dataset
python -m flowerclf.train            # train, save the best checkpoint
python -m flowerclf.evaluate         # test metrics, confusion matrix, sample grid
python -m flowerclf.predict path/to/flower.jpg
```

`predict` prints each class with its probability, sorted most likely first.
Tests live under `tests/` and run with `pytest`.

## Where things live

```
src/flowerclf/
  config.py      paths and hyperparameters
  data.py        transforms, per-class capping, dataloaders
  model.py       transfer-learning model factory
  train.py       training loop with validation and checkpointing
  evaluate.py    test metrics and figures
  predict.py     single-image inference CLI
tests/           unit tests for the data and model code
scripts/         dataset download
reports/         metrics.json and the figures above
notebooks/       a Colab quickstart
```

If you want to push accuracy further, the obvious lever is to stop freezing the
backbone: unfreeze the last ResNet block, train at 224px, and use a lower
learning rate.
