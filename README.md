## 1. 环境配置

### 1.1 基础环境要求
- Python >= 3.8
- PyTorch >= 1.10
- CUDA >= 11.1 (推荐使用GPU加速)
- 操作系统: Linux / Windows / macOS

### 1.2 安装依赖
```bash
# 克隆仓库
git clone https://github.com/ng204/StructGroup.git
cd StructGroup

**注意**: 如果你的CUDA版本与默认PyTorch版本不兼容，请先从[PyTorch官网](https://pytorch.org/)安装对应版本的PyTorch。

## 2. 数据集准备

### 2.1 数据集结构
请将数据集按照以下目录结构组织：
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

### 2.2 数据集获取
本研究使用的三七植物点云数据集较大，未上传至GitHub。如需获取完整数据集，请通过下方邮箱联系作者。

## 3. 预训练模型权重

### 3.1 权重文件说明
- **文件格式**: PyTorch (.pth)
- **训练环境**: PyTorch 2.0 + CUDA 11.7
- **模型性能**: 在三七植物点云测试集上达到[XX.X%]的精度

### 3.2 权重获取
由于模型权重文件较大，已上传至百度网盘：
> 百度网盘链接: [通过网盘分享的文件：work_dirs
链接: https://pan.baidu.com/s/10-bgNOQ3pt305gOA4eXBwg?pwd=gsdu 提取码: gsdu 
--来自百度网盘超级会员v5的分享]
> 提取码: [gsdu]

下载完成后，请将权重文件放置在项目根目录下。

## 4. 模型测试（复现论文结果）

**运行以下命令加载预训练权重并测试模型**：
```bash
python test.py
```

### 4.1 测试说明
- 脚本会自动加载项目根目录下的预训练权重文件
- 自动读取`./dataset/Panax/Panax_data/`目录下的测试数据
- 无需重新训练，直接运行即可得到与论文一致的结果
- 测试结果将输出到终端，并自动保存到`./results/`目录下

### 4.2 预期输出
```
Loading model weights...
Model loaded successfully.
Testing on Panax point cloud test set...
Test Accuracy: XX.X%
Test Precision: XX.X%
Test Recall: XX.X%
Test F1-Score: XX.X%
Results saved to ./results/test_results.txt
```

## 5. 模型训练

如需重新训练模型，直接运行：
```bash
python train.py
```

### 5.1 训练配置
- 所有训练参数均在`configs/StructGroup/default.yaml`文件中配置
- 可根据硬件条件调整batch_size、learning_rate等参数
- 训练过程中会自动保存checkpoint到`./checkpoints/`目录
- 训练日志会保存到`./logs/`目录

### 5.2 训练结果
- 训练完成后，最优模型权重会保存为`./checkpoints/best_model.pth`
- 可使用该权重文件替换下载的预训练权重进行测试



## 6. 联系方式

如有任何问题或需要获取完整数据集，请通过以下方式联系：
- 邮箱: [yangling@kust.edu.cn]

---

**注意**: 本代码和数据集仅用于学术研究目的。
```
