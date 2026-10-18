# SlumNet

An end-to-end deep learning framework for detecting and mapping informal
settlements (slums) from high-resolution satellite imagery. Hybrid
CNN + Vision Transformer backbone, attention modules, multi-scale
(BiFPN) fusion, a Mixture-of-Experts refinement stage, and a U-Net-style
decoder with attention gates — plus GIS export, explainability, a
Streamlit dashboard, and a FastAPI backend.

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

GDAL system libraries are required for `rasterio`/`geopandas`. On Ubuntu:

```bash
sudo apt-get install gdal-bin libgdal-dev
```

## 2. Dataset

Place georeferenced scenes and matching masks here:

```
dataset/
  images/<scene_id>.tif   (or .png/.jpg)
  masks/<scene_id>.png    (single-channel; pixel value = class index)
```

Good public sources to seed this from:

- **Sentinel-2 / Landsat-8** — free multispectral imagery (10–30 m/px);
  export RGB (or false-color) composites as GeoTIFFs via Copernicus
  Browser or Google Earth Engine.
- **SpaceNet** (AWS Open Data) — very-high-resolution imagery with
  building footprints, useful for building-density labels.
- **Maxar Open Data** — sub-meter imagery released for specific
  disaster/humanitarian events.
- **Google Open Buildings** — building footprint polygons (where
  licensing permits) to help bootstrap slum vs. planned-residential
  labels.
- Manually labeled slum datasets from research partners/NGOs, if
  available, generally give the strongest signal since "slum" is a
  socioeconomic label, not a purely visual one — treat model output as
  decision support, not ground truth.

`train.py` handles patch extraction (256×256 default, configurable),
class balancing, and the train/val split automatically from whatever
full scenes you drop into `dataset/`.

## 3. Configuration

All hyperparameters (backbone choice, transformer, attention type,
fusion strategy, MoE settings, loss weights, training schedule) live in
`configs/config.yaml`. Edit this before training rather than the code.

## 4. Train

```bash
python train.py --config configs/config.yaml
```

Checkpoints are saved to `checkpoints/`, TensorBoard logs to
`outputs/logs/`:

```bash
tensorboard --logdir outputs/logs
```

## 5. Evaluate

```bash
python test.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth
```

Produces `outputs/test_report.json` with mIoU, Dice, precision/recall/F1,
boundary IoU, and a full confusion matrix.

## 6. Predict on a single image

```bash
python predict.py --config configs/config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --image path/to/scene.tif \
    --out-dir outputs/prediction_001 \
    --pixel-area-m2 100 \
    --gradcam
```

Writes a mask, RGB overlay, confidence map, GeoTIFF/GeoJSON exports,
area/cluster statistics, and (optionally) a Grad-CAM heatmap.

## 7. Interactive dashboard

```bash
streamlit run dashboard/app.py
```

Upload an image, view the mask/overlay/confidence side by side, inspect
Grad-CAM and Mixture-of-Experts activation, and export mask/report files.

## 8. REST API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- `POST /upload` — store an image, get back an `image_id`
- `POST /predict` — run inference on an uploaded file, get mask/overlay/
  report download links plus area statistics inline
- `GET /report/{result_id}` — fetch a prior report
- `GET /health` — liveness check

## 9. Docker

```bash
docker compose up --build
```

Starts the API on `:8000` and the dashboard on `:8501`. Mount
`checkpoints/` and `dataset/` as needed (already wired in
`docker-compose.yml`).

## 10. Project layout

```
slumnet/
  dataset/            preprocessing, augmentation, PyTorch Dataset
  models/
    backbones/         CNN backbone (ResNet50 / EfficientNet / ConvNeXt)
    transformers/       Swin / ViT / SegFormer branch
    attention/          CBAM, channel, spatial, multi-head self-attention
    experts/             Mixture-of-Experts block
    decoder/             fusion (FPN/BiFPN/cross-attn) + U-Net decoder
    slumnet.py           top-level model wiring
  losses/              Dice / Focal / CE / Boundary combined loss
  metrics/             mIoU, Dice, precision/recall/F1, boundary IoU
  utils/               config/checkpoint helpers, GIS export, Grad-CAM/XAI
  configs/             config.yaml
  api/                 FastAPI backend
  dashboard/           Streamlit app
  train.py / test.py / predict.py
```

## Notes & caveats

- "Slum" is a socioeconomic and legal designation as much as a visual
  one; a model trained on visual/morphological cues (roof density,
  road irregularity, building packing) will pick up correlated
  patterns but can misclassify dense-but-planned settlements or miss
  slums with irregular but sparse layouts. Outputs are best used as
  triage/prioritization input for follow-up by domain experts, not as
  a sole basis for policy or resource-allocation decisions.
- Real-world area statistics require georeferenced input imagery
  (`.tif` with a valid CRS/transform); plain PNG/JPEG uploads only
  produce pixel-space statistics unless you supply `--pixel-area-m2`.
- The cloud-masking function in `dataset/preprocessing.py` is a
  brightness/saturation heuristic; swap in a dedicated model (e.g.
  `s2cloudless`) for production Sentinel-2 pipelines.
