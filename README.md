# 3D 项目简介

这是一个 3D 项目，旨在实现三维数据处理和分析功能。该项目提供了多种工具和库来支持不同类型的3D任务，包括数据加载、预处理、模型训练和推理。

## 文件树结构

```sh
3D/
├── data                         # 数据集存储目录
├── libraries                    # 第三方库(Astra SDK)
├── models                       # 模型文件存储目录
├── qtResource                   # Qt图形界面资源文件（图标、界面定义等）
├── results                      # 处理结果输出目录
│   ├── human                    # 人工可读结果
│   └── machine                  # 机器可读结果
├── test                         # pytest测试脚本目录
├── src                          # 项目主要的运行文件
├── tools                        # 快捷操作的脚本工具
├── README.md                    # 项目说明文档
```

## 安装说明

1. 克隆仓库：
   ```bash
   git clone https://github.com/lovedream-ms/3D.git
   cd 3D
   ```

2. 安装依赖项：
   ```bash
   uv sync
   ```

## 使用示例

### 示例 1: 运行测试

```bash
python test/test_astra_resolution.py
```
此脚本用于测试Astra分辨率算法，验证其性能和准确性。

### 示例 2: 批量推理

```bash
python test/batch_infer_test.py
```
此脚本用于批量处理和推理3D数据集，生成预测结果。
