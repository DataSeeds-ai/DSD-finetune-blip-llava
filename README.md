# BLIP2 Training Setup

## Prerequisites

1.  **Clone Long_CLIP:** Ensure that the [Long_CLIP](https://github.com/beichenzbc/Long-CLIP) repository is cloned *inside* the `blip2/` directory. Also rename it to `Long_CLIP`

## Training

The core training logic and BLIP workings are detailed in the [LAVIS documentation](https://github.com/salesforce/LAVIS).

To start training a model using the Gurushots Technical dataset configuration:

1.  Navigate to the project root directory.
2.  Execute the training script:

    ```bash
    bash run_scripts/blip2/train/train_gurushots_technical.sh
    ```

The specific training configuration can be found in:
`blip2/LAVIS/lavis/projects/blip2/train/gurushots_technical_ft.yaml`