# MultiSpecVision Architecture Documentation

This document helps you quickly understand the `multispecvision/MultiSpecVision-Core` project source code and distinguish the roles of various file types.

## Core Model Files (models/)

These files define the network structures of MultiSpecVision and its variants, serving as the "heart" of the entire project:
- `models/swin_transformer.py`: **MultiSpecVision backbone model implementation**, the most core model code.
- `models/swin_transformer_v2.py`: MultiSpecVision V2 implementation.
- `models/swin_transformer_moe.py`: MultiSpecVision-MoE (Mixture of Experts) model implementation.
- `models/swin_mlp.py`: MultiSpecVision-MLP model implementation.
- `models/multi_channel_swin_transformer.py`: **Multi-channel MultiSpecVision model**, extended version supporting 3-20 channel input.

## Training and Inference Entry Points (Main Directory)

These are the actual entry files for running training, evaluation, and testing:
- `main.py`: Main training/evaluation entry (image classification), standard MultiSpecVision training and testing script.
- `main_moe.py`: Training/evaluation for MultiSpecVision-MoE (Mixture of Experts) model.
- `main_simmim_pt.py` and `main_simmim_ft.py`: Self-supervised pre-training and fine-tuning scripts.
- `app.py`: Basic Web application entry, providing single image recognition functionality.
- `multi_channel_app.py`: **Multi-channel Web application entry**, providing multi-channel image segmentation functionality.
- `test_inference.py`: Model inference testing script.

## Configuration Files (configs/)

- `configs/swin/`: **MultiSpecVision variant configurations** (such as tiny, small, base, large, etc.), including training, fine-tuning, different resolution configurations.
- `configs/simmim/`: Self-supervised training configuration files.
- `configs/swinmlp/`, `configs/swinmoe/`, `configs/swinv2/`: Corresponding configurations for MultiSpecVision-MLP, MultiSpecVision-MoE, MultiSpecVision V2.

## Fine-tuning Related Configuration Files

If you want to **fine-tune based on pre-trained models**, pay attention to:
- In directories like `configs/swin/`, `configs/simmim/`, yaml files with `finetune` in the filename (such as `swin_base_patch4_window12_384_finetune.yaml`, `swin_base_patch4_window7_224_22kto1k_finetune.yaml`) **are the configuration files for fine-tuning**, used to continue training from pre-trained models to new tasks or new resolutions.

## Data Processing (data/)

- `data/build.py`: Data loading and preprocessing build script.
- `data/cached_image_folder.py`: Optimized image folder dataset.
- `data/samplers.py`: Sampling strategies during training.
- `data/data_simmim_*.py`: Data processing for self-supervised training.
- `models/dng_processor.py`: **DNG raw image processor**, used for processing multi-channel camera data.

## Tools and Optimization (utils & kernels/)

- `utils.py`, `optimizer.py`, `lr_scheduler.py`: Utility functions, optimizers, learning rate schedulers for training.
- `logger.py`: Logging utility.
- `kernels/window_process/`: Efficient window processing CUDA acceleration code.

## Web Application Related

- `templates/index.html`: Basic image recognition Web interface.
- `templates/multi_channel_index.html`: **Multi-channel image segmentation Web interface**.
- `static/`: Static resource files for Web application.

## File Classification Summary Table

| File Type | File Path | Purpose |
|-----------|-----------|---------|
| Core Models | models/swin_transformer.py | MultiSpecVision backbone network |
| Extended Models | models/multi_channel_swin_transformer.py | Multi-channel MultiSpecVision model |
| Configuration Files | configs/swin/*.yaml, configs/simmim/*.yaml | Training, fine-tuning, structure parameter configurations |
| Training Entry | main.py, main_moe.py, main_simmim_*.py | Training and evaluation scripts |
| Fine-tuning Config | configs/swin/*finetune*.yaml, configs/simmim/*finetune*.yaml | Configuration files for fine-tuning |
| Web Applications | app.py, multi_channel_app.py | Web interface application entries |
| Data Processing | data/*.py, models/dng_processor.py | Data loading, preprocessing |
| Utility Functions | utils.py, optimizer.py, lr_scheduler.py | Training utilities |

---

**📖 Reading Recommendations**:
1. **Beginners**: Start with `models/swin_transformer.py` to understand core architecture, then look at `app.py` to understand usage.
2. **Researchers**: Focus on model implementations in `models/` directory and configuration files in `configs/` directory.
3. **Engineers**: Pay attention to application entries like `app.py`, `multi_channel_app.py` and deployment-related files.
4. **Multi-channel Application Developers**: Focus on studying `models/multi_channel_swin_transformer.py` and `multi_channel_app.py`. 