# Background Matting V2 Reimplementation

This repository contains a reimplementation of the paper:

**Background Matting V2: Real-Time High-Resolution Background Matting**  
*Peter Lin, Cem Keskin, Shih-En Wei, Yaser Sheikh*  
[arXiv preprint arXiv:2012.07810](https://arxiv.org/abs/2012.07810)

The reimplementation introduces the following modifications:

- **Datasets**: Uses the [P3M](https://paperswithcode.com/dataset/p3m-10k) dataset for portrait images and the [BG20K](https://paperswithcode.com/dataset/bg-20k) dataset for background images for base training data. Uses the [VideoMatte240K](https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets), [PhotoMatte85](https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets) and [Backgrounds](https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets) for refine training data.
- **Backbone Network**: Employs **MobileNetV3** instead of MobileNetV2 for improved performance.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
  - [Local Testing](#local-testing)
  - [Training](#Training)
  - [Running on Great Lakes Cluster](#running-on-great-lakes-cluster)
- [Batch Script Example](#batch-script-example)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)
- [References](#references)

---

## Overview

This project focuses on reimplementing the Background Matting V2 model with enhancements:

- **Datasets**: Incorporates the P3M dataset for high-quality portrait images and the BG20K dataset for diverse background images.
- **Backbone Network**: Upgrades the backbone network to MobileNetV3 for better efficiency and accuracy.

The implementation is adapted to run on the University of Michigan's Great Lakes High-Performance Computing Cluster. 

You can test the trained model online on [huggingface](https://huggingface.co/spaces/Shangyunle/Background-Matting).

---

## Features

- High-resolution background matting.
- Real-time inference capabilities.
- Utilizes **MobileNetV3** as the backbone network.
- Uses **P3M** and **BG20K** datasets for base training.
- Uses the **VideoMatte240K**, **PhotoMatte85** and **Backgrounds** for refine training.
- Adapted for use on HPC clusters, specifically the UMich Great Lakes cluster.
- Supports multiple inference backends:
  - PyTorch (Research)
  - TorchScript (Production)
  - ONNX (Experimental)
- Pretrained Models: The project has released pretrained models, you can access on [Google Drive](https://drive.google.com/drive/folders/1hC5u12Mqqc3u9LAHWV4OZhztLEOxKpxW?usp=sharing) for further research.
- Huggingface Instance: You can test the trained model online on [huggingface](https://huggingface.co/spaces/Shangyunle/Background-Matting).

---

## Results

In this project, I tested the performance of the trained model in [Background Matting and Background Matting V2 Footage](https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets) ,and compared it with the results of the original model. Here are some key results:

<table style="border-collapse: collapse; border-spacing: 0; width: 100%;">
<tr>
    <th style="width: 50%; text-align: center; padding: 10px;">Origin Model</th>
    <th style="width: 50%; text-align: center; padding: 10px;">Reimplementation</th>
</tr>
<tr>
    <td style="width: 50%; text-align: center; padding: 10px;"><img src="https://image.pipzza.pw/1733304202635.webp" width="200" style="display: block; margin: auto;"/></td>
    <td style="width: 50%; text-align: center; padding: 10px;"><img src="https://image.pipzza.pw/1733304204815.webp" width="200" style="display: block; margin: auto;"/></td>
</tr>
<tr>
    <td style="width: 50%; text-align: center; padding: 10px;"><img src="https://image.pipzza.pw/1733304207530.webp" width="200" style="display: block; margin: auto;"/></td>
    <td style="width: 50%; text-align: center; padding: 10px;"><img src="https://image.pipzza.pw/1733304210721.webp" width="200" style="display: block; margin: auto;"/></td>
</tr>
</table>

All [test results](https://drive.google.com/drive/folders/1sZodmiO97JVyyjpT9H3UYuR-1Y4AlY9P?usp=sharing) and [train log](https://drive.google.com/drive/folders/1ORbrEgXIreVlJdIhpaDxcKbGvInevzvs?usp=sharing) are available on Google Drive.

## Requirements

- Python 3.8 or higher
- PyTorch 1.7 or higher
- CUDA 11.0 or higher (if using GPU acceleration)
- Additional Python packages:
  - `torchvision`
  - `onnxruntime`
  - `numpy`
  - `opencv-python`

---

## Installation

### Local Environment

1. **Clone the Repository**

   ```bash
   git clone https://github.com/joycexjl/BackgroundMatting.git
   cd BackgroundMatting
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Great Lakes Cluster Environment

1. **Connect to the Cluster**

   ```bash
   ssh your_umich_username@greatlakes.arc-ts.umich.edu
   ```

2. **Load Modules**

   ```bash
   module purge
   module load python/3.8.2
   module load cuda/11.0
   ```

3. **Create a Virtual Environment**

   ```bash
   cd /home/your_umich_username/
   python -m venv matting_env
   source matting_env/bin/activate
   ```

4. **Install Dependencies**

   ```bash
   pip install --upgrade pip
   pip install torch torchvision onnxruntime numpy opencv-python
   ```

---

## Dataset Preparation

1. **Download the Datasets**

   - **P3M Dataset**: [Download Link](https://paperswithcode.com/dataset/p3m-10k)
   - **BG20K Dataset**: [Download Link](https://paperswithcode.com/dataset/bg-20k)

   Place the datasets on your local machine and change `data_path.py` to the corresponding path.

2. **Transfer the Datasets to the Cluster**

   ```bash
   scp -r /path/to/dataset your_umich_username@greatlakes.arc-ts.umich.edu:/scratch/your_umich_username/dataset/
   ```

   - **Note**: Use the `/scratch` directory for large datasets on the cluster.

3. **Verify Data Integrity**

   Ensure that the data is correctly transferred and accessible.

---

## Usage

### Local Testing

You can test the model locally using the provided scripts.

#### Inference

```bash
python inference_images.py \
    --model-type torchscript \
    --model-backbone mobilenetv3 \
    --model-backbone-scale 0.25 \
    --model-refine-sample-pixels 80000 \
    --model-path ./models/model.pth \
    --src ./images/src_image.png \
    --bgr ./images/bgr_image.png \
    --output ./results/output.png
```



### Training

For basenet training:

```bash
python train_base.py \
        --dataset-name p3m10k \
        --background-dataset bg20k \
        --model-backbone mobilenetv3 \
        --model-name mattingbase-mobilenetv3-p3m10k \
        --epoch-end 50
```

For refinenet training:

```bash
python train_refine.py \
        --dataset-name videomatte240k \
        --model-backbone mobilenetv3 \
        --model-name mattingrefine-mobilenetv3-videomatte240k \
        --model-last-checkpoint "checkpoints/checkpoint-xx.pth" \
        --background-dataset backgrounds \
        --batch-size 4
```



### Running on Great Lakes Cluster

#### 1. Transfer Code and Scripts

```bash
scp -r /path/to/BackgroundMatting your_umich_username@greatlakes.arc-ts.umich.edu:/home/your_umich_username/
```

#### 2. Submit a Batch Job

Create a batch script (e.g., `run_matting.sh`) as described in the [Batch Script Example](#batch-script-example) section.

Submit the job:

```bash
sbatch run_matting.sh
```

---

## Batch Script Example

Below is an example of a Slurm batch script for running the model on the Great Lakes cluster.

```bash
#!/bin/bash
#SBATCH --job-name=matting_job          # Job name
#SBATCH --account=your_slurm_account    # Slurm account
#SBATCH --partition=standard            # Partition (queue)
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=8               # Number of CPU cores per task
#SBATCH --mem=32G                       # Total memory per node
#SBATCH --gres=gpu:1                    # Number of GPUs per node
#SBATCH --time=24:00:00                 # Time limit hrs:min:sec
#SBATCH --output=matting_%j.out         # Standard output and error log

# Print some info
echo "Running on host $(hostname)"
echo "Job started at $(date)"
echo "Directory is $(pwd)"

# Load modules
module purge
module load python/3.8.2
module load cuda/11.0

# Activate virtual environment
source /home/your_umich_username/matting_env/bin/activate

# Navigate to project directory
cd /home/your_umich_username/BackgroundMatting/

# Run inference
python inference_images.py \
    --model-type torchscript \
    --model-backbone mobilenetv3 \
    --model-backbone-scale 0.25 \
    --model-refine-sample-pixels 80000 \
    --model-path /home/your_umich_username/BackgroundMatting/models/model.pth \
    --src /scratch/your_umich_username/dataset/P3M/test/src_image.png \
    --bgr /scratch/your_umich_username/dataset/BG20K/test/bgr_image.png \
    --output /scratch/your_umich_username/results/output.png

echo "Job ended at $(date)"
```

**Instructions:**

- Replace `your_slurm_account` with your actual Slurm account name.
- Ensure that the paths to the source image, background image, and model are correct.
- Submit the script using `sbatch run_matting.sh`.

---

## Project Structure

```
BackgroundMatting/
├── dataset/                 # Contains datasets (P3M, BG20K)
├── doc/                     # Documentation and notes
├── eval/                    # Evaluation scripts and metrics
├── images/                  # Sample images for testing
├── model/                   # Model architecture scripts
│   ├── __init__.py
│   ├── MattingBase.py       # Base matting model
│   ├── MattingRefine.py     # Refined matting model
│   └── ...                  # Additional model files
├── .gitignore
├── LICENSE
├── README.md                # Project documentation
├── data_path.py             # Script to manage dataset paths
├── export_onnx.py           # Script to export model to ONNX
├── export_torchscript.py    # Script to export model to TorchScript
├── inference_images.py      # Script for image inference
├── inference_speed_test.py  # Script to test inference speed
├── inference_utils.py       # Utility functions for inference
├── inference_video.py       # Script for video inference
├── inference_webcam.py      # Script for webcam inference
├── requirements.txt         # Python dependencies
├── train_base.py            # Training script for base model
├── train_refine.py          # Training script for refined model
└── ...                      # Additional scripts and files
```

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

## License

This project is licensed under the terms of the MIT license.

---

## Acknowledgments

- **Original Authors**: Peter Lin, Cem Keskin, Shih-En Wei, Yaser Sheikh
- **Original Repository**: [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMatting.git)
- **Datasets**:
  - **P3M Dataset**: [https://paperswithcode.com/dataset/p3m-10k](https://paperswithcode.com/dataset/p3m-10k)
  - **BG20K Dataset**: [https://paperswithcode.com/dataset/bg-20k](https://paperswithcode.com/dataset/bg-20k)
- **University of Michigan ARC**: For providing the Great Lakes cluster resources.

---

## Contact

For any questions or issues, please contact:

- **Name**: Joyce Liu
- **Email**: joycexjl@umich.edu

---

## References

- [Background Matting V2 Paper](https://arxiv.org/abs/2012.07810)
- [Original GitHub Repository](https://github.com/PeterL1n/BackgroundMattingV2)
- [UMich Great Lakes User Guide](https://documentation.its.umich.edu/arc-hpc/greatlakes/user-guide)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [P3M Dataset](https://paperswithcode.com/dataset/p3m-10k)
- [BG20K Dataset](https://paperswithcode.com/dataset/bg-20k)
- [MobileNetV3 Paper](https://arxiv.org/abs/1905.02244)

---

## Additional Notes

- **Data Privacy**: Ensure that any data used complies with data usage agreements and privacy laws.
- **Resource Management**: Be mindful of the resources requested when submitting jobs to the cluster to optimize scheduling and efficiency.
- **Environment Modules**: Use the module system on Great Lakes to manage software dependencies effectively.

---

# Quick Start Guide

1. **Clone the Repository**

   ```bash
   git clone https://github.com/joycexjl/BackgroundMatting.git
   cd BackgroundMatting
   ```

2. **Set Up Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Prepare Datasets**

   - Download the P3M and BG20K datasets.
   - Place them in the `dataset/` directory.

4. **Train the Model**

   ```bash
    python train_base.py \
        --dataset-name p3m10k \
        --background-dataset bg20k \
        --model-backbone mobilenetv3 \
        --model-name mattingbase-mobilenetv3-p3m10k \
        --epoch-end 50
   ```

5. **Run Inference Locally**

   ```bash
   python inference_images.py \
       --model-type torchscript \
       --model-backbone mobilenetv3 \
       --model-backbone-scale 0.25 \
       --model-refine-sample-pixels 80000 \
       --model-path ./models/model.pth \
       --src ./images/src_image.png \
       --bgr ./images/bgr_image.png \
       --output ./results/output.png
   ```

6. **Prepare for Cluster Execution**

   - Transfer data and code to the cluster.
   - Create and submit a batch script.

7. **Monitor Job**

   ```bash
   squeue -u your_umich_username
   tail -f matting_JOBID.out
   ```

8. **Retrieve Results**

   ```bash
   scp your_umich_username@greatlakes.arc-ts.umich.edu:/scratch/your_umich_username/results/output.png /local/path/to/save/
   ```

---

Thank you for using this reimplementation. We hope it aids in your research and projects!

---

## Frequently Asked Questions

### **Q1: Why use MobileNetV3 instead of MobileNetV2?**

**A1**: MobileNetV3 offers improved performance and efficiency over MobileNetV2 due to architectural advancements. It achieves better accuracy with lower computational cost, making it suitable for real-time applications.

### **Q2: How do I change the backbone network?**

**A2**: In the training and inference scripts, specify the `--model-backbone` parameter:

```bash
--model-backbone mobilenetv3
```

You can replace `mobilenetv3` with other supported backbones if desired.

### **Q3: Can I use this model for videos?**

**A3**: Yes, use the `inference_video.py` script to perform matting on videos:

```bash
python inference_video.py \
    --model-type torchscript \
    --model-backbone mobilenetv3 \
    --model-path ./models/model.pth \
    --src-video ./videos/src_video.mp4 \
    --bgr-video ./videos/bgr_video.mp4 \
    --output ./results/output_video.mp4
```

### **Q4: How do I export the model to ONNX or TorchScript?**

**A4**: Use the provided scripts:

- **Export to TorchScript**:

  ```bash
  python export_torchscript.py \
      --model-backbone mobilenetv3 \
      --model-path ./models/model.pth \
      --output-path ./models/model_scripted.pt
  ```

- **Export to ONNX**:

  ```bash
  python export_onnx.py \
      --model-backbone mobilenetv3 \
      --model-path ./models/model.pth \
      --output-path ./models/model.onnx
  ```
