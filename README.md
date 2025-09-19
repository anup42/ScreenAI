# ScreenAI Icon Annotator

A small utility that queries the multimodal model `Qwen/Qwen2.5-VL-72B-Instruct` to name icons highlighted inside application screenshots. Each screenshot is paired with a JSON file that contains the bounding boxes of the icons to be described. The script crops every icon region, sends it to the model, and stores the predicted icon names.

> **Note**: The Qwen 72B vision-language model is extremely large. Running it locally requires multiple high-memory GPUs or an inference endpoint that exposes the model. The provided script focuses on automating the inference/responses; provisioning the compute is left to you.

## Repository Layout

- `src/annotate_icons.py` – CLI that performs the annotation workflow.
- `requirements.txt` – Python dependencies for local execution.

You can create additional folders such as `screenshots/` and `annotations/` alongside the script. The CLI accepts paths, so the exact layout is flexible.

## Installation

1. Create and activate a Python 3.10+ environment.
2. Install the requirements:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) If you plan to enable `--load-in-4bit`, install `bitsandbytes` on a Linux machine with CUDA support.

4. Make sure you are authenticated with Hugging Face if the model requires it:

   ```bash
   huggingface-cli login
   ```

## Expected Bounding Box JSON Format

For each screenshot `example.png`, place a JSON file named `example.json` in the bounding-box directory. The script accepts either of the formats below:

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

- BBoxes can be `[x, y, width, height]` in pixels or `[x1, y1, x2, y2]` in pixels.
- Normalised coordinates in the range `[0, 1]` are also supported.
- If an `id` is missing, the script assigns sequential IDs starting at `1`.

## Usage

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

### Overlay Mode

If your screenshots already contain drawn bounding boxes with numeric IDs, you can skip
the separate JSON annotations and ask the model to describe each numbered icon directly:

```bash
python src/annotate_icons.py \
  --screenshots-dir data/screenshots_with_boxes \
  --output-dir results \
  --overlay-mode
```

In this mode the script sends the entire screenshot to the model along with the prompt
defined by `--overlay-user-prompt` (customisable). The resulting JSON contains the raw
model response plus a lowercased variant when `--lowercase-output` is enabled.

Command-line options worth highlighting:

- `--boxes-dir` – Location of the JSON files. Defaults to the screenshots directory.
- `--boxes-suffix` – File extension used for the JSON files (default: `.json`).
- `--crop-padding` – Adds a percentage of padding around each bounding box before sending it to the model (default: 5%).
- `--load-in-4bit` – Enables 4-bit quantisation (Linux + CUDA only).
- `--temperature` / `--top-p` – Control sampling. The default is deterministic output.
- `--system-prompt` / `--user-prompt` – Customise the prompt template.

Each run produces JSON files such as `results/example_labels.json`:

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

The `raw_response` field stores the exact model output, while `label` is a normalised version that can be used directly.

## Scaling Notes

- Expect the 72B model to need significant GPU memory (multiple 80 GB cards for full precision). For smaller hardware, consider running the script against a hosted inference endpoint or enabling `--load-in-4bit`.
- When using remote endpoints (for example, Hugging Face Inference Endpoints or vLLM), update `--model-name` to match the deployed model or adapt the script to call your endpoint.
- You can tune `--max-new-tokens` and the prompt templates to balance accuracy and runtime.

## Development Tips

- Run a syntax check with `python -m py_compile src/annotate_icons.py` before pushing changes.
- Use `--quiet` during batch runs to reduce console noise, relying on the summarised progress bar instead.
- Consider keeping a subset of screenshots locally for quick smoke tests before tackling the full dataset.

Enjoy annotating!
