---
base_model: Qwen/Qwen2.5-VL-3B-Instruct
pipeline_tag: image-to-text
library_name: peft
tags:
- vision-language
- lora
- floor-plan
- vectorization
- structured-json
- cubicasa
- sft
datasets:
- Claudio9701/cubicasa5k
- Forceless/Zenodo10K
license: apache-2.0
language:
- en
---

# Qwen2.5-VL floor plan SFT adapter (stage 1)

**Hub:** [`mudasir13cs/qwen25-vl-3b-floorplan-sft`](https://huggingface.co/mudasir13cs/qwen25-vl-3b-floorplan-sft)

**Improved using Qwen** — LoRA **supervised fine-tuning (SFT)** on [**CubiCasa5K**](https://arxiv.org/abs/1904.01920) ([CC BY‑NC 4.0](https://github.com/CubiCasa/CubiCasa5k/blob/master/LICENSE)), built on [**Qwen2.5-VL-3B-Instruct**](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) ([LICENSE](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE)).

Intended **non-commercial / research** use, consistent with the Qwen research license and CubiCasa **NC** terms.

**Stage 2 (optional):** GRPO fine-tune — [`mudasir13cs/qwen25-vl-3b-floorplan-grpo`](https://huggingface.co/mudasir13cs/qwen25-vl-3b-floorplan-grpo). In a local training checkout, see `floorplan-vlm-grpo/README.md` for the same style of usage doc.

## Paper & upstream training material

- **Method:** [FloorplanVLM (arXiv:2602.06507)](https://arxiv.org/abs/2602.06507)
- **Original Hub collection:** **[manitocross/floorplan-vlm-training](https://huggingface.co/manitocross/floorplan-vlm-training)** — readme, Zenodo→CubiCasa5K download, SVG→JSON labels, GRPO overview.
- **SFT recipe (canonical source on Hub):** [`train_floorplan_vlm.py`](https://huggingface.co/manitocross/floorplan-vlm-training/blob/main/train_floorplan_vlm.py) — `MODEL_ID`, `HUB_MODEL_ID`, `OUTPUT_DIR`, `DATA_DIR`, epochs, LR, LoRA layout, **`SYSTEM_PROMPT` / `USER_PROMPT`** (use these strings at inference).

If you are working inside a **clone of your training repo**, the same script may exist locally as `train_floorplan_vlm.py`.

## Quick install

**Inference / loading only:**

```bash
pip install torch torchvision transformers peft accelerate pillow
```

**Full SFT training** (matches script docstring; includes data prep deps):

```bash
pip install torch torchvision transformers trl peft datasets accelerate shapely Pillow lxml numpy tqdm huggingface_hub
# optional GPU attention: pip install flash-attn
```

## Loading the adapter

Use **Hub** IDs (recommended):

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

BASE = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER = "mudasir13cs/qwen25-vl-3b-floorplan-sft"

processor = AutoProcessor.from_pretrained(BASE)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE, torch_dtype="auto", device_map="auto"
)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
```

If you cloned weights into **this folder** locally: set `ADAPTER = "./floorplan-vlm-sft"` (or an absolute path) instead of the Hub repo id.

## Using the model (inference)

This stage defines the **recommended** prompts: the **system** message embeds the output JSON schema. Keep `SYSTEM_PROMPT` and `USER_PROMPT` aligned with [`train_floorplan_vlm.py`](https://huggingface.co/manitocross/floorplan-vlm-training/blob/main/train_floorplan_vlm.py) constants.

**User** text: *“Vectorize this floor plan into structured JSON with all walls, doors, windows, and rooms.”*

End-to-end pattern (matches the inference test at the end of `train_floorplan_vlm.py`):

```python
import json, re, torch
from PIL import Image

SYSTEM_PROMPT = (
    "You are a floor plan vectorization expert. Extract wall, door, window geometry "
    "from floor plan images into structured JSON.\n\n"
    "Output ONLY valid JSON with this schema:\n"
    '{"walls":[{"id":"wall_N","start":[x,y],"end":[x,y],"thickness":T,"curvature":0,'
    '"openings":[{"type":"door"|"window","center":D,"width":W}]}],'
    '"rooms":[{"label":"room_type","walls":["wall_N",...]}]}\n\n'
    "Coordinates normalized so longer image edge = 1024."
)
USER_PROMPT = "Vectorize this floor plan into structured JSON with all walls, doors, windows, and rooms."

image = Image.open("plan.png").convert("RGB")

messages = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_PROMPT}]},
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=4096, do_sample=False)

raw = processor.batch_decode(out[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)[0]
m = re.search(r"\{[\s\S]*\}", raw)
plan = json.loads(m.group()) if m else None
```

**Output shape:** top-level `walls` (with optional `openings`) and `rooms`. Example JSON is documented under **Output JSON Schema** in the [Manitocross training README](https://huggingface.co/manitocross/floorplan-vlm-training/blob/main/README.md).

## Reproducing stage 1

1. `huggingface-cli login` if you push to Hub (`PUSH_TO_HUB` in script).
2. Run [`train_floorplan_vlm.py`](https://huggingface.co/manitocross/floorplan-vlm-training/blob/main/train_floorplan_vlm.py): first run downloads **CubiCasa5K** from Zenodo into `./cubicasa_data` (~5GB). Tune `NUM_EPOCHS`, `MAX_SAMPLES`, `LEARNING_RATE`, `HUB_MODEL_ID`, etc. in the configuration block at the top of that file.

## Training details

- **Base model:** `Qwen/Qwen2.5-VL-3B-Instruct`
- **Dataset:** [CubiCasa5K](https://arxiv.org/abs/1904.01920) · [Zenodo](https://doi.org/10.5281/zenodo.2613548)
- **Method:** LoRA SFT (`SFTTrainer` / TRL; see script)
- **Supervision targets:** SVG-derived structured JSON (`walls`, `rooms`, openings)

_Add your own run metadata: epochs, LR, seed, GPU, date, partial `MAX_SAMPLES` if applicable._

## Citation

```bibtex
@article{floorplanvlm2026,
  title={FloorplanVLM: A Vision-Language Model for Floorplan Vectorization},
  journal={arXiv preprint arXiv:2602.06507},
  year={2026}
}
```

## Acknowledgments

- [FloorplanVLM (arXiv:2602.06507)](https://arxiv.org/abs/2602.06507)
- [CubiCasa5K (arXiv:1904.01920)](https://arxiv.org/abs/1904.01920)
- [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- Upstream training reference: [manitocross/floorplan-vlm-training](https://huggingface.co/manitocross/floorplan-vlm-training)
- Stage 2 (optional): [`mudasir13cs/qwen25-vl-3b-floorplan-grpo`](https://huggingface.co/mudasir13cs/qwen25-vl-3b-floorplan-grpo)

## Author / contact

**Mudasir** — multimodal AI, VLM fine-tuning, retrieval/RAG research, and engineering; **MS AI Convergence**, [숭실대학교 — Soongsil University](https://ssu.ac.kr/), Seoul. More credentials, publications, and projects: **[mudasir13cs.github.io](https://mudasir13cs.github.io/)**

- **Hugging Face:** [@mudasir13cs](https://huggingface.co/mudasir13cs)  
- **GitHub:** [@mudasir13cs](https://github.com/mudasir13cs)  
- **Email:** mudasir13cs@gmail.com