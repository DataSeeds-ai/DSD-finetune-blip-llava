# Peer-Ranked Precision

This repository contains the code, configurations, and training scripts associated with our white paper, "Peer-Ranked Precision: Creating a Foundational Dataset for Fine-Tuning Vision Models from GuruShots' Annotated Imagery".

Our work utilizes and adapts components from two main projects:
*   **LLaVA-NeXT**: For details on our LLaVA-NeXT setup, experiments, and scripts, please see the [LLaVA-NeXT Project README](./LLaVA-NeXT/README.md).
*   **BLIP2**: For details on our BLIP2 setup, training configurations, and scripts, please see the [BLIP2 Project README](./blip2/README.md).

## Overview
The development of modern Artificial Intelligence (AI) models, particularly diffusion-based models employed in computer vision and image generation tasks, is undergoing a paradigmatic shift in development methodologies. Traditionally dominated by a "Model Centric" approach, wherein performance gains were primarily pursued through increasingly complex model architectures and hyperparameter optimization, the field is now recognizing a more nuanced "Data-Centric" approach. This emergent framework foregrounds the quality, structure, and relevance of training data as the principal driver of model performance.

To operationalize this paradigm shift, we introduce the DataSeeds.AI Sample Dataset (the "DSD"), comprised of approximately 10,610 high-quality human peer-ranked photography images accompanied by extensive multi-tier annotations. The DSD is a foundational computer vision dataset designed to usher in a new standard for commercial image datasets. Representing a small fraction of GuruShots' 100 million-plus image catalogue, the DSD provides scalable foundation necessary for robust commercial and multimodal AI development.

This repository makes publicly available the code and trained models used in our evaluation, as documented in the white paper. This allows users to reproduce experiments and utilize our findings.

## Repository Structure
- `/LLaVA-NeXT`: Contains code, scripts, and configurations related to the LLaVA-NeXT models used in our research. This includes the forked LLaVA-NeXT project.
- `/blip2`: Contains code, scripts, and configurations related to the BLIP2 models used in our research.

## Citation
If you find this work useful, please consider citing our white paper:
```bibtex
@article{[TBD],
  title={Peer-Ranked Precision: Creating a Foundational Dataset for Fine-Tuning Vision Models from GuruShots' Annotated Imagery},
  author={Sajjad Abdoli, Freeman Lewin, Gediminas Vasiliauskas, and Fabian Schonholz},
  journal={[add]},
  year={[2025]},
  url={[link]}
}
```
