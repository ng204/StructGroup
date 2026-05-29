## 1. Environment Setup

### 1.1 Basic Environment Requirements
- Python >= 3.8
- PyTorch >= 1.10
- CUDA >= 11.1 (GPU acceleration is recommended)
- Operating System: Linux / Windows / macOS

### 1.2 Install Dependencies
```bash
# Clone the repository
git clone https://github.com/ng204/StructGroup.git
cd StructGroup

**Note**: If your CUDA version is incompatible with the default PyTorch version, please install the corresponding PyTorch version from the [PyTorch Official Website](https://pytorch.org/) first.

## 2. Dataset Preparation

### 2.1 Dataset Structure
Please organize the dataset according to the following directory structure:
```
StructGroup/
└── dataset/
    └── Panax/
        └── Panax_data/
            ├── Area_1/
            ├── Area_2/
            ├── Area_3/
            └── ...
```

### 2.2 Dataset Access
The Panax notoginseng plant point cloud dataset used in this study is large and has not been uploaded to GitHub. If you need the complete dataset, please contact the author via the email address provided below.

## 3. Pre-trained Model Weights

### 3.1 Weight File Description
- **File Format**: PyTorch (.pth)
- **Training Environment**: PyTorch 2.0 + CUDA 11.7
- **Model Performance**: Achieves [XX.X%] accuracy on the Panax notoginseng point cloud test set

### 3.2 Weight Download
Due to the large size of the model weight file, it has been uploaded to Baidu Netdisk:
> Baidu Netdisk Link: [https://pan.baidu.com/s/10-bgNOQ3pt305gOA4eXBwg?pwd=gsdu]
> Extraction Code: [gsdu]

After downloading, please place the weight file in the root directory of the project.

## 4. Model Testing (Reproduce Paper Results)

**Run the following command to load the pre-trained weights and test the model**:
```bash
python test.py
```

### 4.1 Testing Instructions
- The script will automatically load the pre-trained weight file from the project root directory
- Automatically read test data from the `./dataset/Panax/Panax_data/` directory
- No retraining is required; running directly will produce results consistent with the paper
- Test results will be output to the terminal and automatically saved to the `./results/` directory

### 4.2 Expected Output
```
Loading model weights...
Model loaded successfully.
Testing on Panax notoginseng point cloud test set...
Test Accuracy: XX.X%
Test Precision: XX.X%
Test Recall: XX.X%
Test F1-Score: XX.X%
Results saved to ./results/test_results.txt
```

## 5. Model Training

If you need to retrain the model, run directly:
```bash
python train.py
```

### 5.1 Training Configuration
- All training parameters are configured in the `configs/StructGroup/default.yaml` file
- You can adjust parameters such as batch_size and learning_rate according to your hardware conditions
- Checkpoints will be automatically saved to the `./checkpoints/` directory during training
- Training logs will be saved to the `./logs/` directory

### 5.2 Training Results
- After training is completed, the optimal model weights will be saved as `./checkpoints/best_model.pth`
- You can use this new weight file to replace the downloaded pre-trained weights for testing

## 6. Contact Information

If you have any questions or need to obtain the complete dataset, please contact us via:
- GitHub Issues: https://github.com/ng204/StructGroup/issues
- Email: [yangling@kust.edu.cn]

---

**Note**: This code and dataset are for academic research purposes only.
```
