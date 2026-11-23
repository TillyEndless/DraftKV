# 安装指南

[English](INSTALL.md) | **简体中文**

DraftKV 支持四组模型各自独立的 Python 环境，以及经过源码和 SHA-256 校验的在线资产下载。本指南区分**在线初始化（bootstrap）**与**端到端推理**：部分原始 Native 源码和基准输入尚无公开下载渠道。

## 环境要求

- **Linux x86_64 + NVIDIA CUDA GPU**，配备兼容的驱动、CUDA Toolkit 和编译工具链。Hopper SM90 自动选择 FA3；其他 NVIDIA 架构自动选择 FA2，不会编译 Hopper 专用组件。
- 使用 Conda（Miniconda 或 Anaconda）提供 Python 3.10 和 `pip`，无需系统级 Python 3.10。还需 Git、curl，并为模型和数据集预留足够磁盘空间。
- 网络能够访问 PyPI、PyTorch wheel 索引、上游官方 GitHub Commit 及 Hugging Face。
- 若模型资产受 Hugging Face 访问控制限制，需提前取得访问授权并登录 Hugging Face，或设置 `HF_TOKEN`。安装器不会代替用户申请或绕过访问权限。

安装器**不负责**安装操作系统、系统级 CUDA Toolkit 或 GPU 驱动。

## 在原生 Conda 环境中安装（无嵌套 venv）

克隆仓库：

```bash
git clone https://github.com/TillyEndless/DraftKV.git
cd DraftKV
```

以 LingBot-World 1.3B 为例，按以下步骤安装：

```bash
# Step 1：创建并激活专属的 Python 3.10 Conda 环境
conda create -n draftkv python=3.10 pip -y
conda activate draftkv

# Step 2：直接在当前 Conda 环境中安装固定版本的 Python 依赖
./scripts/draftkv.sh environment --backbone lingbot1.3b --python "$(which python)"

# Step 3：下载可获取的模型及数据集资产
./scripts/draftkv.sh assets --backbone lingbot1.3b \
  --asset-root "$HOME/.cache/draftkv/assets" --with-dataset
```

**Conda 环境本身就是包安装与运行环境**，而非仅提供基础解释器。安装器会将锁定版本的 PyTorch 和其他 Python 包安装到已激活的非 `base` Conda 环境中，不会再创建嵌套 venv。`environment`、`bootstrap`、`install`、`prepare`、`run` 均使用该 Python 解释器。Native 构建产物、缓存和校验记录仍存于 Conda 之外的 `~/.cache/draftkv-install/<backbone>/`。**不同 Backbone 必须使用独立 Conda 环境**，因为其依赖版本可能冲突，例如 LongLive 与 LingBot 使用不同的 Torch/CUDA 包。Conda 不负责安装驱动或系统 CUDA Toolkit。

如需使用旧版独立虚拟环境流程，可显式指定 `--env-mode venv`（即使当前已激活 Conda 环境）。默认 `--env-mode auto` 会优先选用已激活的非 `base` Conda 环境；没有专用 Conda 环境时，回退到 venv。已完成的旧版 venv 缓存**不要**直接用作 Conda 部署缓存；必要时指定新的 `--env-root`。

## 按 Backbone 执行在线初始化

克隆仓库并激活相应 Conda 环境后，选择下列**其中一组**命令：

| Backbone | CLI 标识 |
| --- | --- |
| LongLive-1.3B | `longlive1.3b` |
| LingBot-World 1.3B | `lingbot1.3b` |
| LingBot-World 14B | `lingbot14b` |
| SolarWM-33B | `solarwm33b` |

### LongLive-1.3B

```bash
./scripts/draftkv.sh bootstrap --backbone longlive1.3b --with-dataset
```

### LingBot-World 1.3B

```bash
./scripts/draftkv.sh bootstrap --backbone lingbot1.3b --with-dataset
```

### LingBot-World 14B

```bash
./scripts/draftkv.sh bootstrap --backbone lingbot14b --with-dataset
```

### SolarWM-33B

```bash
./scripts/draftkv.sh bootstrap --backbone solarwm33b --with-dataset
```

`bootstrap` 默认使用已激活 Conda 环境中的 `bin/python`。`--python` 为可选参数；如指定，必须指向同一解释器：

```bash
./scripts/draftkv.sh bootstrap --backbone lingbot1.3b --with-dataset \
  --python "$(which python)"
```

`bootstrap` 会安装固定版本的 Python/PyTorch 依赖，下载并校验模型原生依赖使用的官方 FA2 wheel，**仅在 Hopper SM90 上额外构建当前参考版本的 FA3**；同时下载经 SHA-256 验证的公开模型/数据资产，并从锁定的上游 Commit 获取字节完全一致的公开 Native 源码。`--with-dataset` **仅下载已公开、且可以通过校验和验证的数据集资产**。

默认目录为：Python 包放入当前 Conda 环境；Native 产物、缓存和安装记录放入 `~/.cache/draftkv-install/<backbone>/`；模型与数据集资产放入 `~/.cache/draftkv/assets/`。磁盘空间不足时，可通过 `--env-root` 和 `--asset-root` 分别调整外部存储位置。下载中断后，如服务器支持 Range 请求，安装器可尝试断点续传；已校验的文件会直接复用。

## Native 依赖预检

原始源码完整性锁保持不变。[按 Backbone 划分的源码选择清单](../scripts/native_sources.json)仅在文件字节与原始内容完全一致时，才将固定的官方 Git Commit 作为下载来源。不同 SHA-256 的上游文件不能替代 DraftKV 修改过的源码。

| Backbone | 原始 SM90 Native 文件数 | 已验证的公开上游文件 | 无可信 URL 的原始文件 |
| --- | ---: | ---: | ---: |
| LongLive-1.3B | 49 | 28 | 21 |
| LingBot-World 1.3B | 53 | 20 | 33 |
| LingBot-World 14B | 53 | 20 | 33 |
| SolarWM-33B | 4 | 0 | 4 |

检查实际可用和缺失的文件：

```bash
./scripts/draftkv.sh native-deps --backbone lingbot1.3b --dry-run
```

上述数量来自当前冻结的**源码锁清单**，不是模型权重数量。完整的 500 行 `EXTERNAL_DEPENDENCY_SHA256.json` 仍予保留，以兼容原始启动器。按 Backbone 选择源码时，会在 H200 环境下排除无关模型源码和 SM120 头文件，但保留完整原始模型源码锁引用的辅助文件。B01 完整性验证还需要原始证据归档。

## Bootstrap 不包含的内容

原始修改版 Native 源码、定制 B01 归档、部分 SolarWM Native 代码，以及完整的 ORBIA 短案例和 LongLive 基准输入，目前没有已确认的、字节完全一致的公开下载位置。在线 `bootstrap` **不会**重建这些文件、构建项目特定的 B01，也不会生成视频。其 Hopper FA3 构建使用固定的公开参考版本，不等同于还原任何不可获取的历史二进制文件。

当用户具备所需的源码锁定原件与输入案例时，可以分别使用 `install`、`prepare` 和 `run`。在缺少这些输入的情况下，不应将它们描述为可仅凭网络下载就完成的零输入端到端流程。

`bootstrap` 输出会明确报告 `native_sources_unavailable` 和 `end_to_end_generation: NOT_EXECUTED`。未来若提供完整 E2E 发布版，需要先公开剩余 Native 源码和基准输入的可验证文件，再重新执行真实的 H200 完整推理验收。

## 延伸阅读

- [快速开始](QUICKSTART_CN.md) — 模型选择与在线下载。
- [性能分析](PROFILING_CN.md) — Native 启动检查、功能性生成及诊断信息之间的区别。
- [第三方许可声明](THIRD_PARTY_NOTICES.md) — 上游许可证及致谢（英文）。
