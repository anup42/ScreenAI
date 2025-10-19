# ScreenAI Icon Annotator

Utilities for naming UI icons inside screenshots using large multimodal models. The project supports both the original Qwen crop-based workflow and a ViP-LLaVA overlay pipeline for screenshots where bounding boxes and IDs are embedded directly in the image.

> **Model size warning**: The Qwen 72B and ViP-LLaVA checkpoints are heavy. Expect to provision multiple high-memory GPUs, enable 4-bit quantization, or rely on a hosted inference endpoint.

## Repository Layout

- `src/annotate_icons.py` – Crop-based annotation workflow that queries `Qwen/Qwen2.5-VL-72B-Instruct`.
- `src/annotate_icons_vip_llava.py` – Overlay workflow that prompts ViP-LLaVA (7B or 13B) to name labelled icons.
- `src/export_vip_llava_to_yolo.py` – Converts ViP-LLaVA outputs plus bounding boxes into YOLO label files.
- `src/visualize_icon_annotations.py` – Draws predicted names beside each bounding box for visual QA.
- `src/icon_box_utils.py` – Shared helpers for reading and normalising bounding boxes.
- `requirements.txt` – Python dependencies for local execution.

Folders such as `screenshots/` and `annotations/` can be arranged as you see fit; the CLIs accept explicit paths.

## Installation

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Install `bitsandbytes` if you plan to enable `--load-in-4bit` on CUDA hardware.
4. Log in to Hugging Face when required:

   ```bash
   huggingface-cli login
   ```

## Expected Bounding Box JSON Format

For each screenshot `example.png`, create a JSON file named `example.json` in the bounding-box directory. Any of the formats below are accepted:

```json
[
  {"id": 1, "bbox": [120, 45, 40, 40]},
  {"id": 2, "bbox": [200, 45, 42, 42]}
]
```

```json
{
  "boxes": [
    {"id": 1, "bbox": [0.15, 0.08, 0.22, 0.16]},
    {"id": 2, "bbox": [0.30, 0.08, 0.37, 0.16]}
  ]
}
```

- Bounding boxes can be `[x, y, width, height]` or `[x1, y1, x2, y2]` in pixels.
- Normalised coordinates in `[0, 1]` are also supported.
- Missing IDs are filled sequentially, starting at `1`.

## Usage (Qwen Crop-Based)

```bash
python src/annotate_icons.py \
  --screenshots-dir data/screenshots \
  --boxes-dir data/boxes \
  --output-dir results \
  --model-name Qwen/Qwen2.5-VL-72B-Instruct \
  --device-map auto \
  --max-new-tokens 32 \
  --crop-padding 0.05
```

### Overlay Mode (Qwen)

When screenshots already include drawn boxes with numeric IDs, you can skip the external JSON files:

```bash
python src/annotate_icons.py \
  --screenshots-dir data/screenshots_with_boxes \
  --output-dir results \
  --overlay-mode
```

The script sends the entire screenshot alongside the overlay prompt (`--overlay-user-prompt` by default) and stores the raw response plus a lowercased variant when `--lowercase-output` is set.

Example output (`results/example_labels.json`):

```json
{
  "image": "example.png",
  "model": "Qwen/Qwen2.5-VL-72B-Instruct",
  "annotations": [
    {
      "id": 1,
      "bbox": [120, 45, 160, 85],
      "padded_bbox": [118, 43, 162, 87],
      "raw_response": "delete icon",
      "label": "delete icon"
    },
    {
      "id": 2,
      "bbox": [200, 45, 242, 87],
      "padded_bbox": [198, 43, 244, 89],
      "raw_response": "home",
      "label": "home"
    }
  ]
}
```

`raw_response` stores the exact model text, while `label` is a normalised string ready for downstream use.

## ViP-LLaVA Overlay Workflow

Choose this pipeline when icons are already boxed and labelled with IDs inside the screenshot.

### 1. Run ViP-LLaVA annotations

```bash
python src/annotate_icons_vip_llava.py \
  --images-dir data/screenshots_with_ids \
  --output-dir results/vip_llava \
  --model-size 13b \
  --max-new-tokens 256
```

- Use `--model-size 7b` (default) or `13b`, or override with `--model-id <repo>`.
- Enable `--load-in-4bit` for bitsandbytes quantisation (CUDA + Linux required).
- Customise the instruction via `--prompt` and normalise names with `--lowercase-output`.
- The script produces files like `results/vip_llava/example_vip_llava.json`, including the raw response, parsed annotations, and any unparsed lines.

### 2. Convert to YOLO labels

```bash
python src/export_vip_llava_to_yolo.py \
  --annotations-dir results/vip_llava \
  --boxes-dir data/boxes \
  --images-dir data/screenshots_with_ids \
  --output-dir results/yolo_labels
```

- Bounding box JSON files must mirror the image names (same format as above).
- A `classes.txt` file is created or appended in the output directory listing every discovered label.
- Pass `--strict` to raise if an ID is missing from the box metadata instead of skipping the annotation.

### 3. Visualise predicted names

```bash
python src/visualize_icon_annotations.py \
  --annotations-dir results/vip_llava \
  --boxes-dir data/boxes \
  --images-dir data/screenshots_with_ids \
  --output-dir results/visualized \
  --include-id
```

- Provide a TTF font through `--font-path` and tune `--font-size` / `--line-width` to suit the UI scale.
- The script draws coloured rectangles for each box and writes the icon name (and optionally the ID) nearby.

## Scaling Notes

- Expect the 72B Qwen model to require multiple 80 GB GPUs for full precision. For smaller hardware, use hosted endpoints or quantisation.
- ViP-LLaVA 7B/13B still benefit greatly from GPUs with ≥24 GB VRAM. Quantisation and `device_map="auto"` help reduce memory pressure.
- Adjust `--max-new-tokens`, prompts, and sampling parameters in both pipelines to balance accuracy and performance.

## Development Tips

- Run syntax checks with `python -m py_compile src/annotate_icons.py src/annotate_icons_vip_llava.py src/export_vip_llava_to_yolo.py src/visualize_icon_annotations.py` before committing.
- Use `--quiet` for long batch runs to reduce log noise.
- Keep a small set of representative screenshots handy for smoke testing changes.

Enjoy annotating!
