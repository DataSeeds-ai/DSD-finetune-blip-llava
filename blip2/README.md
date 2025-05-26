# BLIP2-OPT-2.7B Fine-tuned on GuruShots Dataset

This model is a fine-tuned version of [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b) specialized for photography scene analysis and technical description generation. The model was fine-tuned on the [GuruShots Sample Dataset (GSD)](https://huggingface.co/datasets/Dataseeds/GuruShots-Sample-Dataset-GSD) to enhance its capabilities in generating detailed photographic descriptions with focus on composition, lighting, and technical aspects.

## Model Description

- **Base Model**: [BLIP2-OPT-2.7B](https://huggingface.co/Salesforce/blip2-opt-2.7b)
- **Vision Encoder**: EVA-CLIP ViT-g/14
- **Language Model**: OPT-2.7B (2.7 billion parameters)
- **Architecture**: BLIP-2 with Q-Former bridging vision and language
- **Fine-tuning Approach**: Full model fine-tuning (ViT unfrozen)
- **Task**: Photography scene analysis and detailed image captioning
- **Precision**: Mixed Precision (AMP enabled)
- **Image Resolution**: 364×364 pixels

## Training Details

### Dataset
The model was fine-tuned on the GuruShots Sample Dataset, containing 10,610 curated photography images with comprehensive annotations:

- **Training Set**: 9,549 image-text pairs (90%)
- **Validation Set**: 1,061 image-text pairs (10%)
- **Content Focus**: Technical scene descriptions, compositional analysis, lighting conditions
- **Annotation Quality**: Professional photography standards with 15+ word descriptions

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Learning Rate** | 1e-5 |
| **Optimizer** | AdamW |
| **LR Schedule** | Linear warmup + Cosine decay |
| **Warmup Steps** | 17 |
| **Weight Decay** | 0.01 |
| **Batch Size** | 8 |
| **Gradient Accumulation** | 1 |
| **Training Epochs** | 10 |
| **Mixed Precision** | AMP enabled |
| **Hardware** | Single NVIDIA A100 80GB |
| **Vision Encoder** | Unfrozen (trainable) |
| **Gradient Checkpointing** | Enabled |

### Generation Configuration

| Parameter | Value |
|-----------|-------|
| **Max Length** | 100 tokens |
| **Min Length** | 8 tokens |
| **Num Beams** | 5 |
| **Beam Search** | Enabled |
| **Task** | Captioning |

### Checkpoint Selection
- **Selection Metric**: Aggregate validation score (CIDEr + BLEU-4)
- **Best Epoch**: Epoch 1 (aggregate score: 0.2626)
- **Training Loss**: Decreased from 2.780 (epoch 1) to 1.692 (epoch 10)
- **Final Model**: `checkpoint_best.pth` (selected based on validation performance)

## Performance

### Quantitative Results

The fine-tuned model shows significant improvements in lexical overlap metrics, with notable trade-offs in semantic understanding:

| Metric | Base Model | Fine-tuned | Absolute Δ | Relative Δ |
|--------|------------|------------|------------|------------|
| **BLEU-4** | 0.001 | **0.047** | +0.046 | **+4600%*** |
| **ROUGE-L** | 0.126 | **0.242** | +0.116 | **+92.06%** |
| **BERTScore F1** | 0.0545 | **-0.0537** | -0.1082 | **-198.53%** |
| **CLIPScore** | 0.2854 | **0.2583** | -0.0271 | **-9.49%** |

*_Note: The extreme BLEU-4 improvement is due to the very low baseline (0.001), making relative improvements appear dramatic._

### Key Observations

**Strengths:**
- **Enhanced Lexical Matching**: Substantial improvements in BLEU-4 and ROUGE-L indicate better n-gram alignment with reference descriptions
- **Photography Terminology**: Model learned to incorporate photographic vocabulary and technical terms
- **Compositional Awareness**: Improved description of camera angles, lighting conditions, and visual elements

**Trade-offs:**
- **Semantic Coherence**: Negative BERTScore suggests divergence from reference semantic patterns
- **Visual-Text Alignment**: Moderate decrease in CLIPScore indicates reduced image-text semantic alignment
- **Repetitive Patterns**: Tendency toward formulaic descriptions with some repetition

### Performance Characteristics

The fine-tuning results reveal a model that has specialized in lexical pattern matching for photography descriptions but with trade-offs in semantic understanding. This suggests the model is particularly suited for applications requiring technical photography terminology rather than general-purpose image captioning.

## Usage

### Installation

```bash
pip install transformers torch pillow
```

### Basic Usage

```python
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image

# Load model and processor
processor = Blip2Processor.from_pretrained("Dataseeds/BLIP2-opt-2.7b-GSD-FineTune")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Dataseeds/BLIP2-opt-2.7b-GSD-FineTune",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load and process image
image = Image.open("your_image.jpg")
inputs = processor(image, return_tensors="pt").to(model.device, torch.float16)

# Generate caption
generated_ids = model.generate(
    **inputs,
    max_length=100,
    min_length=8,
    num_beams=5,
    do_sample=False
)

caption = processor.decode(generated_ids[0], skip_special_tokens=True)
print(f"Generated caption: {caption}")
```

## Model Architecture

The model maintains the BLIP-2 architecture with the following components:

### Core Architecture
- **Vision Encoder**: EVA-CLIP ViT-g/14 (unfrozen during fine-tuning)
- **Q-Former**: 32-layer transformer bridging vision and language modalities
- **Language Model**: OPT-2.7B decoder-only transformer
- **Bootstrapping**: Two-stage pre-training methodology preserved

### Technical Specifications
- **Vision Resolution**: 364×364 pixels
- **Vision Patch Size**: 14×14
- **Q-Former Queries**: 32 learnable queries
- **Language Model Layers**: 32
- **Total Parameters**: ~2.7B (language model) + vision components
- **Precision**: Mixed precision (FP16/FP32)

### Fine-tuning Approach
- **Vision Encoder**: Trainable (freeze_vit: False)
- **Q-Former**: Trainable
- **Language Model**: Trainable
- **Full Model**: End-to-end fine-tuning enabled
- **Gradient Checkpointing**: Memory optimization enabled

## Training Data & Methodology

### Dataset Characteristics
The GuruShots Sample Dataset focuses on:
- **Technical Photography**: Camera settings, composition analysis
- **Lighting Descriptions**: Ambient, directional, studio lighting analysis  
- **Subject Matter**: Diverse photographic subjects and styles
- **Annotation Style**: Technical scene descriptions (20-30 words typical)

### Data Processing
- **Image Preprocessing**: `blip2_image_train` (364×364 resolution)
- **Text Processing**: `blip_caption` processor
- **Evaluation**: `blip_image_eval` for consistent validation
- **Format**: Input-output pairs with scene analysis prompts

### Recommended Use Cases
- ✅ Photography scene analysis and technical descriptions
- ✅ Camera composition and lighting analysis
- ✅ Product photography captioning
- ✅ Photography education and training applications
- ⚠️ General-purpose image captioning (may show repetitive patterns)
- ❌ Non-photographic content analysis (not optimized)

## Citation

If you use this model in your research or applications, please cite:

```bibtex
@misc{blip2-opt-gsd-finetune-2024,
  title={BLIP2-OPT-2.7B Fine-tuned on GuruShots Dataset for Photography Analysis},
  author={Dataseeds},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/Dataseeds/BLIP2-opt-2.7b-GSD-FineTune},
  note={Fine-tuned model for photography scene analysis and technical description}
}

@inproceedings{li2023blip2,
  title={BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models},
  author={Li, Junnan and Li, Dongxu and Xiong, Caiming and Hoi, Steven},
  booktitle={International Conference on Machine Learning},
  pages={19730--19742},
  year={2023},
  organization={PMLR}
}

@article{zhang2022opt,
  title={OPT: Open Pre-trained Transformer Language Models},
  author={Zhang, Susan and Roller, Stephen and Goyal, Naman and Artetxe, Mikel and Chen, Moya and Chen, Shuohui and Dewan, Christopher and Diab, Mona and Li, Xian and Lin, Xi Victoria and others},
  journal={arXiv preprint arXiv:2205.01068},
  year={2022}
}
```

## License

This model is released under the MIT license, consistent with the base BLIP2 model licensing terms.

## Acknowledgments

- **Base Model**: Salesforce Research for BLIP2 architecture and pre-training
- **Language Model**: Meta AI for OPT-2.7B foundation
- **Vision Encoder**: OpenAI for EVA-CLIP vision components
- **Dataset**: GuruShots photography community for source imagery
- **Framework**: Hugging Face Transformers for model infrastructure

## Training Artifacts

This repository includes comprehensive training artifacts:
- **Checkpoints**: All epoch checkpoints (0-9) plus best performing checkpoint
- **Evaluation Results**: Validation metrics for each epoch
- **Training Logs**: Complete training and evaluation logs
- **Configuration**: Original training configuration and hyperparameters

---

*For questions, issues, or collaboration opportunities, please visit the [model repository](https://huggingface.co/Dataseeds/BLIP2-opt-2.7b-GSD-FineTune) or contact the Dataseeds team.* 