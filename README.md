# 任务2：场景目标检测与视频多目标跟踪

## 任务内容

本任务主要包括以下四部分：

1. 使用 **Road Vehicle Images Dataset** 数据集微调训练 YOLOv8 检测模型
2. 使用训练好的模型对测试视频进行逐帧检测与多目标跟踪
3. 分析遮挡场景下的目标丢失或 ID 跳变现象
4. 基于 Tracking ID 和虚拟线实现越线计数

## 环境配置

建议使用 **Google Colab + GPU** 运行。

安装依赖：

```bash
pip install ultralytics opencv-python
```

## 数据集说明

数据集采用 YOLO 检测格式，主要文件结构如下：

```text
trafic_data/
├── data_1.yaml
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```

其中：

- 训练集：2704 张图像
- 验证集：300 张图像
- 类别数：21

## 主要代码文件

- `task2_yolo_pipeline.py`：任务 2 主脚本

该脚本包含三个主要功能：

- `train`：训练检测模型
- `val`：验证模型
- `track`：视频跟踪与越线计数

## 模型训练

训练命令如下：

```bash
python task2_yolo_pipeline.py train \
  --data trafic_data/data_1.yaml \
  --model yolov8n.pt \
  --epochs 30 \
  --imgsz 640 \
  --batch 16 \
  --project runs/task2 \
  --name exp1
```

## 模型验证

在 Ultralytics YOLOv8 中，训练过程中会自动在验证集上进行评估。

如果需要手动再次验证，可使用：

```bash
python task2_yolo_pipeline.py val \
  --data trafic_data/data_1.yaml \
  --weights /content/runs/detect/runs/task2/exp1/weights/best.pt
```

## 视频跟踪与越线计数

跟踪命令如下：

```bash
python task2_yolo_pipeline.py track \
  --weights /content/runs/detect/runs/task2/exp1/weights/best.pt \
  --source test_video.mp4 \
  --output outputs/tracked_output.mp4 \
  --line 200 350 1000 350 \
  --show-labels
```

运行后可以得到：

- 目标检测框
- 目标类别
- 跟踪 ID
- 越线计数结果

## 实验结果

本实验使用 **YOLOv8n** 作为检测模型，训练完成后的主要结果如下：

- 模型：YOLOv8n
- 输入尺寸：640
- Batch Size：16
- Epoch：30
- 验证集结果：`mAP50 = 0.419`
- 验证集结果：`mAP50-95 = 0.259`
- 测试视频越线计数结果：`7`

## 模型权重下载

模型权重文件 `best.pt` 下载链接如下：

- Google Drive 链接：https://drive.google.com/file/d/1VLSRhuArsgIhKLf2XgmNcnDvY2gVeMP3/view?usp=sharing

## 说明

- 本实验训练过程在 Google Colab GPU 环境中完成。
- 测试视频用于验证检测、跟踪和越线计数功能。
- 遮挡与 ID 跳变分析请结合实验报告中的连续关键帧进行说明。

