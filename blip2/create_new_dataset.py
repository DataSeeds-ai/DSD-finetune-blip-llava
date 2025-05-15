import json
import os
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
import urllib.request
import urllib.parse

# --- Configuration ---
INPUT_DATA_ROOT = Path("./gurushot_finalized/pipeline")
OUTPUT_LAVIS_DIR = Path("./output/gurushots_technical/")

OUTPUT_ANNOTATIONS_DIR = OUTPUT_LAVIS_DIR / "annotations"
OUTPUT_IMAGES_DIR = OUTPUT_LAVIS_DIR / "images"

FAILED_ITEMS_LOG = INPUT_DATA_ROOT / "failed_items_missing_description.txt"
PROBLEMATIC_ENTRIES_LOG = INPUT_DATA_ROOT / "problematic_entries.json"

# --- Train/Validation Split ---
TRAIN_RATIO = 0.9
RANDOM_SEED = 22

def extract_technical_description(annotation_data):
    """Extracts the technical scene description from the annotation dict."""
    # --- Find Scene_Description attribute ID ---
    scene_desc_spec_id = None
    for label in annotation_data["labels"]:
        if label.get("name") == "description":
            for attr in label.get("attributes", []):
                if attr.get("name") == "Scene_Description":
                    scene_desc_spec_id = attr.get("id")
                    break
            if scene_desc_spec_id:
                break
    
    if not scene_desc_spec_id:
        return None  # Return None instead of raising an exception
        
    # --- Find the attribute value in annotations ---
    for tag in annotation_data["annotations"]["tags"]:
        for attribute in tag.get("attributes", []):
            if attribute.get("spec_id") == scene_desc_spec_id:
                value = attribute.get("value")
                if value:
                    return value.strip()
                else:
                    # Return None instead of raising an exception
                    return None
    return None

# --- Main Processing Logic ---
print("Starting dataset preparation...")

OUTPUT_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

all_task_data = []
problematic_entries = []  # List to track problematic entries

# Find all JSON files
json_files = list(INPUT_DATA_ROOT.glob("gurushot-batch-*.json"))
print(f"Found {len(json_files)} batch JSON files in {INPUT_DATA_ROOT}.")

processed_item_count = 0
failed_item_count = 0

# Process each JSON file
for json_path in json_files:
    print(f"Processing file: {json_path.name}...")
    try:
        with open(json_path, 'r') as f:
            batch_data = json.load(f)
    except Exception as e:
        print(f"Error reading file {json_path}: {e}")
        problematic_entries.append({
            "batch_file": json_path.name,
            "reason": f"File reading error: {str(e)}"
        })
        continue

    # Process each item in the batch
    for item in batch_data:
        try:
            item_id = item["id"]
            
            # Extract necessary data
            try:
                image_url = item["dataset_url"]
                cvat_annotations_str = item["cvat_annotations"]
            except KeyError as e:
                print(f"Item {item_id} - Missing required field: {e}")
                problematic_entries.append({
                    "id": item_id,
                    "batch_file": json_path.name,
                    "reason": f"Missing required field: {str(e)}"
                })
                failed_item_count += 1
                continue

            # Parse annotation string
            try:
                annotation_data = json.loads(cvat_annotations_str)
            except json.JSONDecodeError as e:
                print(f"Item {item_id} - Invalid JSON in annotations: {e}")
                problematic_entries.append({
                    "id": item_id,
                    "batch_file": json_path.name,
                    "reason": f"Invalid JSON in annotations: {str(e)}"
                })
                failed_item_count += 1
                continue
                
            # Extract description
            description = extract_technical_description(annotation_data)
            
            if description is None:
                print(f"Item {item_id} - Scene_Description value not found or empty.")
                problematic_entries.append({
                    "id": item_id,
                    "batch_file": json_path.name,
                    "reason": "Missing or empty Scene_Description"
                })
                try:
                    with open(FAILED_ITEMS_LOG, 'a') as log_f:
                        log_f.write(f"{item_id}\n")
                    failed_item_count += 1
                except OSError as e:
                    print(f"Error: Could not write to log file {FAILED_ITEMS_LOG}: {e}.")
                continue

            # Prepare image path & download
            parsed_url = urllib.parse.urlparse(image_url)
            image_ext = Path(parsed_url.path).suffix or '.jpg'

            image_filename = f"{item_id}{image_ext}"
            image_path_dest = OUTPUT_IMAGES_DIR / image_filename
            image_path_relative_to_output = image_filename

            if not image_path_dest.exists():
                print(f"Downloading image for {item_id} to {image_path_dest}")
                try:
                    urllib.request.urlretrieve(image_url, image_path_dest)
                except Exception as e:
                    print(f"Error downloading image for {item_id}: {e}")
                    problematic_entries.append({
                        "id": item_id,
                        "batch_file": json_path.name,
                        "reason": f"Image download failed: {str(e)}"
                    })
                    failed_item_count += 1
                    continue
                
            # Add to list
            all_task_data.append({
                "image": image_path_relative_to_output,
                "caption": description,
                "image_id": item_id
            })
            processed_item_count += 1
            
        except Exception as e:
            # Catch-all for any unexpected errors during item processing
            print(f"Unexpected error processing item: {e}")
            problematic_entries.append({
                "batch_file": json_path.name,
                "reason": f"Unexpected error: {str(e)}"
            })
            failed_item_count += 1
            continue

# Save the problematic entries list to a JSON file
try:
    with open(PROBLEMATIC_ENTRIES_LOG, 'w') as f:
        json.dump(problematic_entries, f, indent=2)
    print(f"Saved {len(problematic_entries)} problematic entries to {PROBLEMATIC_ENTRIES_LOG}")
except Exception as e:
    print(f"Failed to save problematic entries: {e}")

print(f"\nFinished processing.")
print(f"Successfully processed {processed_item_count} items.")
print(f"{failed_item_count} items failed and were logged.")

# --- Split ---
if all_task_data:  # Only split if we have data
    train_data, val_data = train_test_split(
        all_task_data,
        test_size=1-TRAIN_RATIO,
        random_state=RANDOM_SEED
    )
    test_data = []

    print(f"Splitting into: {len(train_data)} Train, {len(val_data)} Validation, {len(test_data)} Test items.")

    # --- Save JSON files ---
    def save_json(data, path):
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved annotation file: {path}")
        except Exception as e:
            print(f"Error saving {path}: {e}")

    save_json(train_data, OUTPUT_ANNOTATIONS_DIR / "train.json")
    save_json(val_data, OUTPUT_ANNOTATIONS_DIR / "val.json")
    if test_data:
        save_json(test_data, OUTPUT_ANNOTATIONS_DIR / "test.json")

    print("Dataset preparation finished!")
    print(f"    -> Annotations: {OUTPUT_ANNOTATIONS_DIR}")
    print(f"    -> Images: {OUTPUT_IMAGES_DIR}")
else:
    print("No valid data was processed. Check the problematic entries log.")