#!/bin/bash

# Download necessary NLTK data
python -c "import nltk; nltk.download('punkt')"

# Run the evaluation script
python scripts/inf/evaluation_script_base.py \
    --model-path "lmms-lab/llava-onevision-qwen2-0.5b-ov" \
    --eval-data-path "/home/sajjad/AI_proj/example_data/val_data_guru.json" \
    --image-dir "/home/sajjad/AI_proj/example_data" \
    --output-file "evaluation_results.json" \
    --detailed-output "detailed_results.json"

# Print the results
echo "Evaluation complete. Results saved to evaluation_results.json"
echo "Detailed outputs saved to detailed_results.json"