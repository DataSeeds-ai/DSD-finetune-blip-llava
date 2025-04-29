import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
from torchmetrics.text import BERTScore
from transformers import logging as hf_logging

lavis_repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'LAVIS'))
if lavis_repo_path not in sys.path:
    sys.path.insert(0, lavis_repo_path)
from lavis.common.registry import registry
from lavis.models import load_model_and_preprocess
from lavis.common.config import Config

hf_logging.set_verbosity_error() # suppress transformer warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*torchvision.*")

long_clip_repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Long_CLIP'))
if long_clip_repo_path not in sys.path:
    sys.path.insert(0, long_clip_repo_path)
from Long_CLIP.model import longclip

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

lavis_root = "./LAVIS"
config_path = os.path.join("./", lavis_root, "lavis/projects/blip2/train/gurushots_technical_ft.yaml")
checkpoint_path = os.path.join("./", lavis_root, "lavis/output/BLIP2/Caption_Gurushots_OPT2.7b_Run1/20250415092/checkpoint_best.pth")
longclip_checkpoint_path = os.path.join(long_clip_repo_path, "checkpoints/longclip-L.pt")
if not os.path.exists(config_path):
    print(f"Error: Config file not found at {config_path}")
    exit()
if not os.path.exists(checkpoint_path):
    print(f"Error: Checkpoint file not found at {checkpoint_path}")
    exit()
if not os.path.exists(longclip_checkpoint_path):
    print(f"Error: LongCLIP checkpoint file not found at {longclip_checkpoint_path}")
    exit()

base_model_name = "blip2_opt"
base_model_type = "caption_coco_opt2.7b"

batch_size = 8
num_workers = 4

num_beams = 5
max_len = 30
min_len = 8

bert_score_lang = 'en'

# --- Load Config ---
print(f"Loading configuration from: {config_path}")
ft_cfg_omega = OmegaConf.load(config_path)

correct_base_model_config_name = "blip2_caption_opt2.7b.yaml"
base_config_path = os.path.join(lavis_root, "lavis/configs/models/blip2", correct_base_model_config_name)

if not os.path.exists(base_config_path):
    print(f"Error: Base model config not found at the expected path: {base_config_path}")
    exit()

print(f"Loading base model config from: {base_config_path}")
base_cfg_omega = OmegaConf.load(base_config_path)
merged_cfg_omega = OmegaConf.merge(base_cfg_omega, ft_cfg_omega)
cfg_omega = merged_cfg_omega

finetuned_model_arch = cfg_omega.model.arch
print(f"Fine-tuned model architecture (from merged config): {finetuned_model_arch}")

batch_size = cfg_omega.run.get("batch_size_eval", batch_size)
num_workers = cfg_omega.run.get("num_workers", num_workers)
num_beams = cfg_omega.run.get("num_beams", num_beams)
max_len = cfg_omega.run.get("max_len", max_len)
min_len = cfg_omega.run.get("min_len", min_len)
print(f"Using Eval Batch Size: {batch_size}, Num Workers: {num_workers}")
print(f"Generation Params: Beams={num_beams}, MinLen={min_len}, MaxLen={max_len}")


# --- Load Models ---
print(f"Loading base model ({base_model_name} - {base_model_type})...")
base_model, vis_processors, _ = load_model_and_preprocess(
    name=base_model_name,
    model_type=base_model_type,
    is_eval=True,
    device=device
)
vis_processor_eval = vis_processors["eval"]
print("Base model loaded.")

print(f"Loading fine-tuned model ({finetuned_model_arch})...")
cfg_model_ft = cfg_omega.model

eval_img_size = cfg_omega.datasets[list(cfg_omega.datasets.keys())[0]].vis_processor.eval.image_size
cfg_model_ft.image_size = eval_img_size
print(f"Added image_size={eval_img_size} to fine-tuned model config.")

# Create model instance based on the MERGED config
model_cls_ft = registry.get_model_class(finetuned_model_arch)
my_model = model_cls_ft.from_config(cfg_model_ft)

print(f"Loading fine-tuned weights from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Find the state dict within the checkpoint file
if "model" in checkpoint:
    state_dict = checkpoint["model"]
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
elif isinstance(checkpoint, dict) and "epoch" in checkpoint and len(checkpoint) > 1 :
     # Common structure: epoch, model, optimizer, ...
     keys_without_epoch = {k:v for k,v in checkpoint.items() if k != 'epoch'}
     if len(keys_without_epoch) == 1:
         state_dict = list(keys_without_epoch.values())[0]
     else: 
         state_dict = {k:v for k,v in checkpoint.items() if k not in ['epoch', 'optimizer', 'scaler', 'lr_scheduler']}
         if not state_dict:
              state_dict = checkpoint
else:
    state_dict = checkpoint

new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("module."):
        new_state_dict[k[len("module."):]] = v
    else:
        new_state_dict[k] = v

msg = my_model.load_state_dict(new_state_dict, strict=False)

my_model.to(device)
my_model.eval()
print("Fine-tuned model loaded and set to eval mode.")

# --- Load Long-CLIP Model ---
print(f"Loading Long-CLIP model from: {longclip_checkpoint_path}")
try:
    longclip_model, longclip_preprocess = longclip.load(longclip_checkpoint_path, device=device)
    longclip_model.eval()
    print("Long-CLIP model loaded successfully.")
except Exception as e:
    print(f"Error loading Long-CLIP model: {e}")
    exit()

# --- Load Dataset ---
print("Loading validation dataset...")
dataset_name = list(cfg_omega.datasets.keys())[0]
dataset_config = cfg_omega.datasets[dataset_name]

# get the dataset builder class from the registry
# --- Modify LAVIS import ---
# ensure it's registered - also modify this import
from lavis.datasets.builders.gurushots_technical_builder import GurushotsTechnicalBuilder
# --- End modification ---
dataset_builder_cls = registry.get_builder_class(dataset_name)

# --- Instantiate the builder WITH its specific configuration ---
print("Instantiating dataset builder...")
dataset_builder = dataset_builder_cls(cfg=dataset_config)

# --- Build datasets using the instantiated builder ---
print("Building datasets...")
datasets = dataset_builder.build()
valid_split = cfg_omega.run.valid_splits[0] if cfg_omega.run.valid_splits else "val"
val_dataset = datasets[valid_split]
print(f"Validation dataset ('{valid_split}') loaded with {len(val_dataset)} samples.")

def get_pil_image_from_dataset(dataset, index):
    if hasattr(dataset, 'annotation') and index < len(dataset.annotation):
         ann = dataset.annotation[index]
         if 'image' in ann and isinstance(ann['image'], str):
             img_path = ann['image']
             if hasattr(dataset, 'vis_root') and not os.path.isabs(img_path):
                 img_path = os.path.join(dataset.vis_root, img_path)
             if os.path.exists(img_path):
                 return Image.open(img_path).convert("RGB")

    print(f"Error: Could not load PIL image via expected path (annotation) for index {index}.")
    # raise ValueError(f"Could not load image for index {index}")
    return None


# --- DataLoader ---
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=val_dataset.collater
)

# --- Setup Metrics ---
print("Initializing BERTScore...")
try:
    bert_score_metric = BERTScore(lang=bert_score_lang, rescale_with_baseline=True, device=device)
except Exception as e:
    print(f"Error initializing BERTScore: {e}")
    exit()
print("BERTScore initialized.")

all_gt_captions = []
all_base_preds = []
all_ft_preds = []
all_base_longclip_scores = []
all_ft_longclip_scores = []

# --- Evaluation Loop ---
print("Starting evaluation loop...")
for i, batch in enumerate(tqdm(val_loader, desc="Evaluating Batches")):
    images = batch["image"].to(device)
    instance_ids = batch["instance_id"]

    gt_captions_batch = []
    original_pil_images_batch = []
    valid_indices_for_batch = []
    for idx, instance_id_str in enumerate(instance_ids):
        try:
            item_index = int(instance_id_str)
            annotation = val_dataset.annotation[item_index]
            pil_image = get_pil_image_from_dataset(val_dataset, item_index)

            if pil_image is None:
                print(f"Skipping index {item_index} due to image loading failure.")
                continue

            if 'caption' in annotation:
                gt_captions_batch.append(annotation['caption'])
                original_pil_images_batch.append(pil_image)
                valid_indices_for_batch.append(idx)
            else:
                print(f"'caption' key missing in annotation at index {item_index}. Skipping.")
        except (ValueError, IndexError) as e:
             print(f"Could not process instance_id '{instance_id_str}' (expected index): {e}. Skipping.")
        except KeyError as e:
            print(f"Missing expected key '{e}' in annotation at index {item_index} for instance_id '{instance_id_str}'. Skipping.")

    # Filter images and predictions based on valid ground truths found
    if len(valid_indices_for_batch) < len(instance_ids):
        print(f"Some ground truth captions were missing or IDs not found in batch {i}.")
        if not valid_indices_for_batch:
            print(f"Skipping batch {i} due to no valid ground truth found.")
            continue
        filtered_images = images[valid_indices_for_batch]
    else:
        filtered_images = images

    # --- Generation ---
    # Models expect dict input: {"image": images, "prompt": ...}
    model_input = {"image": filtered_images}

    with torch.no_grad():
        # Generate captions with base model
        base_preds_batch = base_model.generate(
            model_input,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len
        )
        # Generate captions with fine-tuned model
        ft_preds_batch = my_model.generate(
            model_input,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len
        )

    # --- Store results for metrics ---
    if gt_captions_batch:
        # Ensure lengths match after filtering
        if gt_captions_batch and len(gt_captions_batch) == len(base_preds_batch) == len(ft_preds_batch) == filtered_images.size(0):
            all_gt_captions.extend(gt_captions_batch)
            all_base_preds.extend(base_preds_batch)
            all_ft_preds.extend(ft_preds_batch)
        else:
            print(f"Length mismatch AFTER generation in batch {i}. GT: {len(gt_captions_batch)}, Base: {len(base_preds_batch)}, FT: {len(ft_preds_batch)}, Imgs: {filtered_images.size(0)}. Skipping batch storage.")

    if original_pil_images_batch and len(original_pil_images_batch) == len(base_preds_batch) == len(ft_preds_batch):
        try:
            with torch.no_grad():
                # 1. Preprocess PIL images for Long-CLIP
                # Apply preprocess individually and stack
                processed_images_longclip = torch.stack(
                    [longclip_preprocess(img) for img in original_pil_images_batch]
                ).to(device)

                # 2. Tokenize texts for Long-CLIP
                base_tokens = longclip.tokenize(base_preds_batch).to(device)
                ft_tokens = longclip.tokenize(ft_preds_batch).to(device)

                # 3. Encode images and texts using Long-CLIP
                image_features = longclip_model.encode_image(processed_images_longclip)
                base_text_features = longclip_model.encode_text(base_tokens)
                ft_text_features = longclip_model.encode_text(ft_tokens)

                # 4. L2-normalize features
                image_features = F.normalize(image_features, p=2, dim=-1)
                base_text_features = F.normalize(base_text_features, p=2, dim=-1)
                ft_text_features = F.normalize(ft_text_features, p=2, dim=-1)

                # 5. Calculate Cosine Similarity (dot product of normalized vectors)
                # Output shape: [batch_size]
                base_cos_sim = torch.sum(image_features * base_text_features, dim=1)
                ft_cos_sim = torch.sum(image_features * ft_text_features, dim=1)

                # 6. Scale score by 100
                base_scores_batch = 100.0 * base_cos_sim
                ft_scores_batch = 100.0 * ft_cos_sim

                # 7. Store individual scores
                all_base_longclip_scores.extend(base_scores_batch.cpu().tolist())
                all_ft_longclip_scores.extend(ft_scores_batch.cpu().tolist())
        except Exception as e:
            print(f"Error calculating Long-CLIP score in batch {i}: {e}")
            import traceback
            traceback.print_exc()

# --- Compute and Print Final Scores ---
# --- BERT Scores ---
print(f"Calculating BERTScore using {len(all_gt_captions)} samples...")
try:
    print("Calculating for Base Model...")
    bert_scores_base = bert_score_metric(all_base_preds, all_gt_captions)
    avg_bert_p_base = torch.mean(bert_scores_base['precision']).item()
    avg_bert_r_base = torch.mean(bert_scores_base['recall']).item()
    avg_bert_f1_base = torch.mean(bert_scores_base['f1']).item()
    print(f"  Base Model Avg BERTScore: P={avg_bert_p_base:.4f} R={avg_bert_r_base:.4f} F1={avg_bert_f1_base:.4f}")

    print("Calculating for Fine-tuned Model...")
    bert_scores_ft = bert_score_metric(all_ft_preds, all_gt_captions)
    avg_bert_p_ft = torch.mean(bert_scores_ft['precision']).item()
    avg_bert_r_ft = torch.mean(bert_scores_ft['recall']).item()
    avg_bert_f1_ft = torch.mean(bert_scores_ft['f1']).item()
    print(f"  Fine-tuned Model Avg BERTScore: P={avg_bert_p_ft:.4f} R={avg_bert_r_ft:.4f} F1={avg_bert_f1_ft:.4f}")

except Exception as e:
    print(f"Error during BERTScore calculation: {e}")
    import traceback
    traceback.print_exc()

# --- CLIP Scores ---
print(f"\n--- CLIP Score (based on {len(all_base_longclip_scores)} samples) ---")
if all_base_longclip_scores:
    avg_longclip_base = sum(all_base_longclip_scores) / len(all_base_longclip_scores)
    print(f"  Base Model Avg CLIP Score: {avg_longclip_base:.4f}")
else:
    print("  Base Model Avg CLIP Score: N/A (no scores calculated)")

if all_ft_longclip_scores:
    avg_longclip_ft = sum(all_ft_longclip_scores) / len(all_ft_longclip_scores)
    print(f"  Fine-tuned Model Avg CLIP Score: {avg_longclip_ft:.4f}")
else:
    print("  Fine-tuned Model Avg CLIP Score: N/A (no scores calculated)")

print("Evaluation finished.")