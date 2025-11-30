## Neural Style Transfer - Gatys et al.

A from-scratch, fully-modular PyTorch implementation based on the original paper -
[Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)

---

### Features

* **Modular Feature Extraction:** Uses VGG-based feature hierarchies with selectable layers for controlling content and style fidelity.
* **Style Representation via Feature Statistics:** Computes and matches feature correlations through Gram matrices to achieve robust texture and style transfer.
* **Optimizable Image Generation:** Synthesizes the stylized image directly in pixel space using gradient-based optimization with support for both L-BFGS and Adam.
* **Configurable Loss Framework:** Allows independent tuning of content, style, and total-variation losses, enabling different stylization strengths and smoothness profiles.
* **Flexible Initialization:** Supports initializing the generated image from the content image, the style image, or random noise to control convergence behavior.
* **Reproducible Experimentation:** Provides structured argument parsing, deterministic configuration, and consistent preprocessing for reliable results.
* **Clean Image I/O Pipeline:** Includes standardized loading, normalization, and post-processing to ensure output correctness across models and resolutions.
* **Extensible Codebase:** Organized into independent modules, making it straightforward to add new loss terms, new models, or alternative style representations.

---

### Example Results
| Content Image | Style Image | Stylized Output |
| ------------- | ----------- | --------------- |
| <img src="./src/data/content-images/green_bridge.jpg" width="200"> | <img src="./src/data/style-images/wave_crop.jpg" width="200"> | <img src="./src/data/output/green_bridge_wave_crop.jpg" width="200"> |

---

### Project Structure
```
├── src/
│   ├── main.py                   # Entry point and main pipeline
│   ├── models/
│   │   ├── vgg.py                # VGG16/VGG19 feature extractor
│   ├── utils/
│   │   ├── image.py           # Image loading, saving, preprocessing
│   │   ├── losses.py         # Loss computations
│   │   └── model.py            # Model preparation and gram matrix calculation  
│   └── data/
│       ├── content-images/       # Input content images
│       ├── style-images/         # Input style references
│       └── output/               # Stylized results
├── requirements.txt
├── README.md                     
└── .gitignore
```

---

### Setup

1. Install Dependencies
```bash
pip install -r requirements.txt
```

2. Run Style Transfer
```bash
python main.py
```

3. Optional Flags

| Flag                       | Default             | Description                                                                     |
| -------------------------- | ------------------- | ------------------------------------------------------------------------------- |
| `--content_image`          | `"golden_gate.jpg"` | Name of the content image                                                     |
| `--style_image`            | `"vg_la_cafe.jpg"`  | Name of the style image                                                        |
| `--content_weight`         | `1`                 | Weight factor for content loss                                                 |
| `--style_weight`           | `1e6`               | Weight factor for style loss                                                 |
| `--total_variation_weight` | `1e-6`              | Weight factor for total variation loss                                        |
| `--optimizer`              | `'lbfgs'`           | `lbfgs`, `adam`                                |
| `--model`                  | `'vgg19'`           | `vgg16`, `vgg19`                                |
| `--init_method`            | `'random'`          | `content`, `style`, `random` |
