# DOCUMENT.md

<!--
This file lives in the root of every forked repo.
Fill it in as you go. Do not reconstruct it after the fact.
Keep entries factual and brief. The audience is a future person
reproducing your setup on a different machine or the HPC cluster.
-->

---

## Model Info

<!--
Copy this information from the upstream repo's README and paper.
"Paired or unpaired" refers to whether the model assumes paired training data.
If the model is domain-specific to virtual staining, note the exact staining task (e.g. H&E to HER2 IHC).
-->

- **Model name:** UGATIT (pytorch)
- **Upstream repo URL:** https://github.com/znxlwm/UGATIT-pytorch
- **Fork URL:** https://github.com/InViLabVirtualStainingBenchmark/UGATIT-pytorch
- **Upstream last commit date:** Oct 15, 2019
- **Paper / citation:** [U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation](https://arxiv.org/abs/1907.10830)
- **Paired or unpaired assumption:** Unpaired 
- **Intended staining task (if domain-specific):** general (unpaired domain translation, not VS-specific)

---

## Environment Claimed by Authors

<!--
Record exactly what the authors say in their README or requirements file.
Do not adjust or interpret -- copy their stated versions.
"Requirements file present" should note the filename if it exists.
If no version is specified for Python or PyTorch, write "not specified".
-->

- **Python version:** not specified
- **PyTorch version:** not specified
- **CUDA version:** not specified
- **Installation method:** not specified (no requirements file present)
- **Requirements file present:** none
- **Pretrained weights available:** no (PyTorch repo has no pretrained weights; TF repo has selfie2anime weights incompatible with PyTorch)
- **Pretrained weights notes:** TF checkpoint format, selfie2anime domain, incompatible with this codebase
<!-- Where are they hosted? Are they behind a login? Is the link likely to rot (GDrive, Dropbox, personal server)? -->
---

## Environment Actually Used

<!--
Record the environment you actually created and tested in.
If you deviated from what the authors specified, briefly note why (e.g. "authors' version not compatible with CUDA 12.1").
Conda env name should follow the convention: the model's short name.
-->

- **Python version:** 3.10
- **PyTorch version:** 2.5.1+cu121
- **CUDA version:** 12.1
- **Conda environment name:** ugatit
- **Date tested:** 08.05.2026
- **Hardware:** RTX 4090, WSL2 on Windows 11

---

## Installation

<!--
Follow the authors' README exactly before making any changes.
Record the commands you ran in order.
If an error occurred, paste the key line of the error (not the full traceback) and then record the fix.
If installation succeeded without issues, write "No issues."
-->

### Commands Run

```bash
# paste the installation commands here in order
conda create -n ugatit python=3.10 -y
conda activate ugatit
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "from UGATIT import UGATIT; from networks import *; from utils import *; from dataset import ImageFolder; print('all imports OK')”
```

### Issues and Fixes

<!--
Format: problem encountered -> fix applied.
If no issues, write "None."
-->

| Issue | Fix Applied |
| --- | --- |
|  |  |

### GPU Confirmation

<!--
Paste the output of the check below so there is proof the GPU was visible.
Command: python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
-->

```
2.5.1+cu121 True NVIDIA GeForce RTX 4090
```

---

## Dataset Preparation

<!--
Record how the dataset was prepared for this specific model.
"Format expected" means what folder layout or file structure the model's data loader assumes
(e.g. side-by-side paired images, separate A/B folders, CSV manifest, etc.).
"Conversion applied" means any script or command you ran to reformat the standard BCI/MIST-HER2
download into the format this model needs.
If no conversion was needed, write "None -- dataset used as downloaded."
-->

- **Dataset used:** BCI & MIST-HER2 
- **Format expected by model:** trainA/ trainB/ testA/ testB/ (flat image folders, one domain per folder; pairing order within folders does not matter)
- **Conversion applied:**
    
    ```bash
    mkdir -p ~/internship-models/datasets/ugatit/BCI-ugatit/{trainA,trainB,testA,testB}
    mkdir -p ~/internship-models/datasets/ugatit/MIST-ugatit/{trainA,trainB,testA,testB} 
    
    # Symlink or copy H&E images into trainA/testA and IHC images into trainB/testB
    ```
    
- **Final folder layout used:**
    
    ```
    # sketch the folder tree here, e.g.:
    datasets/ugatit/
        BCI-ugatit/
            trainA/   <-- H&E source images (smoke: ~100)
            trainB/   <-- IHC target images (smoke: ~100)
            testA/    <-- H&E test images (smoke: 20)
            testB/    <-- IHC reference images for evaluate.py (smoke: 20)
        MIST-ugatit/
            trainA/
            trainB/
            testA/
            testB/
    ```
    
- **Number of images used for smoke test (train / test):** ~100 / 20 per domain per dataset

---

## Pretrained Weights

<!--
Only fill this section if pretrained weights exist.
Record the exact download source. Flag any link that is not on a stable host
(Zenodo and HuggingFace are stable; Google Drive, Dropbox, and personal servers are at risk).
Record where you placed the weights relative to the repo root.
-->

- **Download source URL:**
- **Host stability:** stable (Zenodo / HuggingFace) / at-risk (GDrive / Dropbox / personal server) / N/A
- **Weights placed at (relative path):**
- **Size on disk:**

---


## Training Smoke Test

<!--
Run training for 5 epochs minimum. The goal is a clean exit, not a useful model.
Use the smallest viable batch size and the model's default resolution unless that causes an OOM error.
Always set checkpoint saving to every epoch (e.g. --save_epoch_freq 1 for pix2pix-style repos)
so there is proof a checkpoint was written.
Monitor GPU memory with: watch -n 1 nvidia-smi (run in a second terminal).
-->

- **Script / command run:**
    
    ```bash
    BCI
    # Train
    python main.py --dataset BCI-ugatit --dataroot ~/internship-models/datasets/ugatit --iteration 200 --save_freq 200 --print_freq 100000 --img_size 512 --load_size 1024 --light True --result_dir results 2>&1 | tee smoke_bci.log
    # Test
    python main.py --phase test --dataset BCI-ugatit --dataroot ~/internship-models/datasets/ugatit --img_size 512 --load_size 1024 --light True --result_dir results 2>&1 | tee infer_bci.log
    # Evaluation
    conda activate vs-benchmark
    python ~/internship-models/evaluate/evaluate.py --pred ~/internship-models/ugatit/results/BCI-ugatit/test --gt ~/internship-models/datasets/ugatit/BCI-ugatit/testB --model_name ugatit --dataset_name BCI --split_name smoke-200iter --match_by sort --cellpose --cellpose_model cyto2 --cellpose_n 10 --output ~/internship-models/results.csv

    MIST-HER2
    # Train
    python main.py --dataset MIST-ugatit --dataroot ~/internship-models/datasets/ugatit --iteration 200 --save_freq 200 --print_freq 100000 --img_size 512 --load_size 1024 --light True --result_dir results 2>&1 | tee smoke_mist.log
    # Test
    python main.py --phase test --dataset MIST-ugatit --dataroot ~/internship-models/datasets/ugatit --img_size 512 --load_size 1024 --light True --result_dir results 2>&1 | tee infer_mist.log
    # Evaluation
    conda activate vs-benchmark
    python ~/internship-models/evaluate/evaluate.py --pred ~/internship-models/ugatit/results/MIST-ugatit/test --gt ~/internship-models/datasets/ugatit/MIST-ugatit/testB --model_name ugatit --dataset_name MIST-HER2 --split_name smoke-200iter --match_by sort --cellpose --cellpose_model cyto2 --cellpose_n 1 --output ~/internship-models/results.csv
    ```
    
- **Dataset used:** BCI & MIST-HER2
- **Epochs run:** 200 interaions
- **Batch size:** 1
- **Input resolution:** 1024, crop 512
- **Time per epoch (approx):**
- **Peak GPU memory (approx, from nvidia-smi):**
- **Checkpoint saved:** yes
- **Checkpoint path:** /results/BCI-ugatit/model/BCI-ugatit_params_0000200.pt
- **Crash or error during training:**
<!-- "None" if clean. Otherwise paste the key error line and the fix applied. -->

---

## Output Verification

<!--
Open 3-5 output images and compare them visually against the ground-truth target.
This is not a metric -- just a check that the model produced something in the right domain.
"Expected domain" for BCI would be IHC HER2-stained tissue with brown DAB staining on a light background.
Record one or two example output filenames so the check is reproducible.
-->

- **Output folder:** results/BCI-ugatit/test
- **Example output filenames:** fake_B_0005.png
- **Dimensions match input:** yes
- **Visual sanity check:** Outputs show a domain shift toward IHC-like coloring. As an unpaired model, spatial alignment with the H&E input is not expected. Domain plausibility (light background, some structure) is the only criterion. 
<!-- e.g. "outputs show IHC-like staining, structures roughly aligned with H&E input" -->
- **Any obvious artifacts or failure modes:**

---

## Changes Made to Original Code

<!--
Record every change made to the original repo, no matter how small.
Do not make changes that alter model architecture or training logic.
Only changes needed for the code to run in the benchmark environment are allowed.
Add rows as needed.
-->

| File | Change Description | Reason |
| --- | --- | --- |
| main.py | Added --dataroot CLI argument (default: 'dataset') | Hardcoded 'dataset' relative path incompatible with benchmark dataset layout |
| UGATIT.py | Stored self.dataroot = args.dataroot; replaced hardcoded 'dataset' prefix in four ImageFolder calls in build_model() | Same as above |
| main.py   | Added --load_size argument (default: 0) | Allow crop-from-original-resolution training without resize; 0 preserves original behavior |
| UGATIT.py | Stored self.load_size = args.load_size | Same as above |
| UGATIT.py | Replaced train_transform and test_transform blocks to respect --load_size | Same as above |
| UGATIT.py | Replaced test() method to save individual fake_A2B PNG files instead of 7-panel contact sheets | Contact sheet format incompatible with evaluate.py; panel concatenation also crashes at load_size != img_size |

<!--
Common examples of acceptable changes:

- Pinning a dependency version in requirements.txt (e.g. torch==2.1.0) because no version was specified
- Replacing a hardcoded absolute path with a command-line argument
- Removing an import that is not used and is not installable in the benchmark environment
- Adapting the data loader to accept BCI/MIST-HER2 folder structure
-->

---

## Frozen Environment

<!--
After the smoke test passes, export and commit the environment file.
Command: conda env export > environment_<model-name>.yml
This file is what gets adapted for the HPC migration later.
Note any packages that are unusual, very large, or likely to cause conflicts on the cluster.
-->

- **Environment file:** `environment_ugatit.yml`
- **Committed to fork:** yes
- **Notes on unusual or heavy dependencies:**
<!-- e.g. "requires openslide-python which needs a system-level apt install" -->

---

## HPC Readiness Notes

<!--
Fill this in after the local smoke test passes.
Flag anything that will need attention before running on the VSC cluster.
Common issues: GUI/display dependencies (matplotlib backends), hardcoded CUDA package versions,
dependencies that require apt/system installs, very large model downloads.
Leave blank until local test is complete.
-->

- **Display/GUI dependencies to remove or neutralize:** None identified. UGATIT does not use visdom or matplotlib display calls.
- **System-level dependencies (non-pip/conda):** None
- **Estimated GPU memory requirement:** ~12-16 GB at img_size 512, batch size 1
- **Estimated storage requirement (weights + data):** ~500 MB checkpoints per run (no pretrained weights); dataset mirrors from BCI/MIST-HER2 (~same size as originals)
- **Other notes for cluster adaptation:**
    - UGATIT is iteration-based, not epoch-based. The full benchmark run epoch count must be converted to iterations (default: 1,000,000 iterations).
    - --dataroot and --load_size are custom additions not present in the upstream repo; confirm they are present in the forked version on the cluster.
    - No runtime downloads required. No internet access needed on compute nodes.

---

## Summary

<!--
Write 2-4 sentences summarizing what worked, what did not, and what the next step is.
Be specific. Include the overall pass/fail verdict.
This is the first thing someone reads when picking this model back up.
-->

UGATIT smoke test completed on 08.05.2026 (BCI) and (MIST-HER2). Training ran for
200 iterations without crash on both datasets; checkpoints were written and inference produced
20 output images per dataset. Six changes were made to the original codebase: adding --dataroot
and --load_size CLI arguments, modifying train and test transforms to use them, and replacing
the test() method to output individual PNG files compatible with evaluate.py. As an unpaired
model, SSIM, PSNR, MS-SSIM, and MAE are not reported; only FID and LPIPS are used for comparison.

**Overall result:** PASS

<!-- Example pass:
"[Model] smoke test completed on [date]. Inference with pretrained weights passed on 10 BCI test images.
Training ran for 5 epochs without crash. One change was made to the data loader to accept separate
source/target folders. Frozen environment committed. Ready for full benchmark run."

Example fail:
"[Model] smoke test failed at the environment step. The required PyTorch version (1.4) is not
compatible with CUDA 12.1. Blocked until a workaround is identified. Do not schedule for HPC."
-->