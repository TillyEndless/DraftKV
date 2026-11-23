# 快速开始

[English](QUICKSTART.md) | **简体中文**

DraftKV 提供四组针对不同模型的在线初始化命令，并对来源文件执行校验。Native 安装器要求 **Linux x86_64 + NVIDIA CUDA GPU**、Python 3.10，以及兼容的 CUDA Toolkit/编译器。Hopper SM90 自动选择 FA3；其他 GPU 架构（包括 RTX 5090）自动选择 FA2。

使用 Conda（Miniconda 或 Anaconda）提供 Python 3.10，无需预装系统级 Python 3.10。宿主机必须已具备 NVIDIA 驱动、CUDA Toolkit/编译器、Git 和 curl；Conda 不会安装驱动或系统 CUDA Toolkit。安装器使用已有的 Hugging Face 登录状态或 `HF_TOKEN`。对于**在线初始化**，无需手动准备本地 FlashAttention wheel 或组装模型目录：安装器会下载经验证的官方 FA2 wheel，并仅在检测到 Hopper SM90 时额外构建当前参考版本的 FA3。项目特定的 B01 仍属于独立阶段。

## 克隆仓库

```bash
git clone https://github.com/TillyEndless/DraftKV.git
cd DraftKV
```

## Step 1：创建并激活 Conda 环境

```bash
conda create -n draftkv python=3.10 pip -y
conda activate draftkv
```

## Step 2：安装 Backbone 的 Python 依赖

以 LingBot-World 1.3B 为例：

```bash
./scripts/draftkv.sh environment --backbone lingbot1.3b --python "$(which python)"
```

Python/PyTorch 依赖会**直接安装到已激活的非 `base` Conda 环境**中，不会创建嵌套 venv。CLI 在后续运行阶段使用相同的解释器；Native 产物与安装记录仍保存在 `~/.cache/draftkv-install/<backbone>/`。每个 Backbone 应使用不同的 Conda 环境，以免固定版本的依赖相互冲突。如需使用旧版独立虚拟环境，可指定 `--env-mode venv`。

## Step 3：下载模型与数据集资产

```bash
./scripts/draftkv.sh assets --backbone lingbot1.3b \
  --asset-root "$HOME/.cache/draftkv/assets" --with-dataset
```

## 一键在线初始化

完成 Step 1 并激活 Conda 环境后，`bootstrap` 可合并 Python 环境、模型与数据资产准备阶段，同时自动安装 FA2/FA3 组件并下载公开的 Native 源码。默认自动使用当前 Conda 环境中的 `bin/python`；若显式提供 `--python`，其路径必须与当前解释器一致：

```bash
./scripts/draftkv.sh bootstrap --backbone lingbot1.3b --with-dataset \
  --python "$(which python)"
```

根据需要运行以下 Backbone 对应的命令。`--with-dataset` 还会下载已公开且可校验的数据集文件。

**LongLive-1.3B**

```bash
./scripts/draftkv.sh bootstrap --backbone longlive1.3b --with-dataset
```

**LingBot-World 1.3B**

```bash
./scripts/draftkv.sh bootstrap --backbone lingbot1.3b --with-dataset
```

**LingBot-World 14B**

```bash
./scripts/draftkv.sh bootstrap --backbone lingbot14b --with-dataset
```

**SolarWM-33B**

```bash
./scripts/draftkv.sh bootstrap --backbone solarwm33b --with-dataset
```

默认情况下，Python 包安装在当前 Conda 环境中，各模型的 Native 构建产物与缓存存于 `~/.cache/draftkv-install/`，模型权重和在线数据集存于 `~/.cache/draftkv/assets/`。通过 `--env-root`、`--asset-root` 可以改用其他外部磁盘。重复运行时会校验并复用已下载资产。

## 检查 Native 源码可用性

```bash
./scripts/draftkv.sh native-deps --backbone lingbot1.3b --dry-run
```

源码选择器识别通过 SHA-256 校验的公开上游文件，下载能够访问的文件并报告缺失的原始修改文件。不能使用同名但内容不同的上游代码替代缺失文件。各 Backbone 当前的源码需求见[安装指南：Native 依赖预检](INSTALL_CN.md#native-依赖预检)。

## 端到端推理的边界

**在线 bootstrap 与端到端视频生成是不同阶段。** `bootstrap` 成功意味着固定版本 Python 依赖、所选 FA2/FA3 安装组件和可获取的公开资产已经准备完成；并不意味着已拥有定制 Native/B01 源码归档、完整的原始 ORBIA 短案例或 LongLive 基准输入，也不意味着已完成 B01 Native 编译或 GPU 视频生成。

完整的 `install` / `prepare` / `run` 流程对源码进行校验，缺少原始文件时会明确拒绝继续。未来若公开剩余 Native 和输入资产的可信版本，可将它们纳入下载器，而无需修改科学计算代码。

有关环境与前置条件，请参阅[安装指南](INSTALL_CN.md)；有关 Native 启动检查、视频生成与部署诊断的区别，请参阅[性能分析](PROFILING_CN.md)。
