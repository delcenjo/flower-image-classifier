# Data

Download the dataset with:

```bash
python scripts/download_data.py
```

This fetches the **TensorFlow Flowers** dataset (~3,670 photos across five
species: daisy, dandelion, roses, sunflowers, tulips) into
`data/raw/flower_photos/`, organised as one folder per class so it can be loaded
directly with `torchvision.datasets.ImageFolder`.

The images are not tracked in version control.
