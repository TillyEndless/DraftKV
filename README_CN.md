# DraftKV

[English](README.md) | **简体中文**

**DraftKV: Protecting Persistent Worlds through Hierarchical Streaming Retrieval**

DraftKV 是面向持久化长视频生成和交互式世界模型的**免训练历史 KV 检索框架**。它通过两级检索筛选相关的历史上下文，并将选中的全分辨率 KV 从 CPU 内存流式传输至 GPU 执行 Attention。

## 方法概览

DraftKV 包含三个核心模块：

- **Draft Grounding：** 基于相机视锥和几何对应关系筛选历史候选。
- **Draft Refinement：** 使用紧凑的 Draft Attention 分数对候选 KV Blocks 进一步排序。
- **Streaming CG-Attention：** 重叠执行 CPU→GPU KV 传输与分块 Attention 计算。

全分辨率历史 KV 可以存放在 CPU 内存中，紧凑的 Persistent Draft Bank 则保留在 GPU 上，整个检索流程无需重新训练模型。

### 支持的 Backbone

| 模型 | 源码目录 | CLI 标识 |
| --- | --- | --- |
| LongLive-1.3B | `versions/longlive` | `longlive1.3b` |
| LingBot-World 1.3B | `versions/lingbot13` | `lingbot1.3b` |
| LingBot-World 14B | `versions/h200_large` | `lingbot14b` |
| SolarWM-33B | `versions/h200_large` | `solarwm33b` |

## 安装

**部署平台：** Linux x86_64 + NVIDIA CUDA GPU。请先安装 **Conda（Miniconda 或 Anaconda）**，由 Conda 提供 Python 3.10，无需预先安装系统级 Python 3.10。宿主机需要已经安装可用的驱动、兼容的 CUDA Toolkit/编译工具链、Git 和 curl。Conda 不会自动安装 GPU 驱动或系统级 CUDA Toolkit。在线安装命令负责模型专属 Python 包、权重及可下载的数据资产，**不负责安装操作系统或 GPU 驱动**。

克隆仓库：

```bash
git clone https://github.com/TillyEndless/DraftKV-public.git
cd DraftKV-public
```

### Step 1：创建并激活 Conda 环境

Conda 负责提供 Python 3.10 基础解释器。运行安装命令前，先创建并激活环境：

```bash
conda create -n draftkv python=3.10 pip -y
conda activate draftkv
```

### Step 2：安装对应 Backbone 的 Python 依赖

以 LingBot-World 1.3B 为例，显式传入当前 Conda 环境的解释器：

```bash
./scripts/draftkv.sh environment --backbone lingbot1.3b --python "$(which python)"
```

现在安装器会将固定版本的 Python/PyTorch 依赖**直接安装到已激活的 `draftkv` Conda 环境中**，不再创建嵌套 venv。命令 `environment`、`bootstrap`、`install`、`prepare`、`run` 均使用同一 Conda 解释器；Native 编译产物、安装记录和缓存仍保存在 `~/.cache/draftkv-install/lingbot1.3b/`。要求 Conda 环境使用 Python 3.10，且不能为 `base`。

**每个 Backbone 需要独立的 Conda 环境。** 四组模型的 PyTorch/CUDA 和部分依赖版本并不一致；其他模型请分别使用 `draftkv-longlive`、`draftkv-lingbot14`、`draftkv-solarwm` 等环境名称。安装器会阻止将其他 Backbone 安装进已绑定的环境。

### Step 3：下载模型和数据集资产

```bash
./scripts/draftkv.sh assets --backbone lingbot1.3b \
  --asset-root "$HOME/.cache/draftkv/assets" --with-dataset
```

基础环境安装不包括 native FlashAttention 和 B01/FA3 编译。正式原生 `install` 阶段会检测所选 GPU（物理编号 `--gpu`，默认 `0`）：**Hopper SM90（H100/H200）选择 FA3，其他 NVIDIA 架构（包括 RTX 5090/SM120）选择 FA2 并跳过 FA3 编译**。Hopper 上仍保留模型原生调用所需的 FA2 依赖。Streaming CG-Attention 的 autotune 不受修改；自动选择安装后端不代表已通过 GPU 推理验收。

### 四组 Backbone 的一键网络部署

必须先完成 Step 1，创建并激活专用 Conda 环境。`bootstrap` 会自动选用该环境的 `bin/python`；也可以显式传入 `--python`，但必须与当前激活的 Conda Python 一致：

```bash
./scripts/draftkv.sh bootstrap --backbone lingbot1.3b --with-dataset \
  --python "$(which python)"
```

正常部署时，`bootstrap` 合并环境安装与在线资产下载，**自动为各类 GPU 安装 FA2，在 Hopper SM90 上额外编译并安装 FA3**，并获取与原始 SHA-256 一致的公开 native 源码。安装器使用已有 Hugging Face 登录状态或 `HF_TOKEN`，校验公开文件哈希；所有缓存均在 Git 仓库之外。

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

以上**网络准备命令**无需提供本地 wheel、原生依赖压缩包、手动计算的 SHA-256、数据集索引或其他占位文件路径。不同 Backbone 的下载体积差异较大，特别是 14B 和 33B 模型，请确保有足够的磁盘空间。重复执行会校验并复用已经下载的文件。

Python 包直接安装在已激活的 Conda 环境中；Native 构建、安装记录与缓存默认放在 `~/.cache/draftkv-install/<backbone>/`，模型资产在 `~/.cache/draftkv/assets/`。`--env-root` 与 `--asset-root` 只调整这些外部目录，不改变 Conda 环境路径。

### 网络部署的范围与限制

| 部署内容 | 当前网络准备流程 |
| --- | --- |
| 固定版本 Python / PyTorch 环境 | 从在线软件包仓库安装 |
| 模型权重及可下载的辅助资产 | 自动下载并校验 SHA-256 |
| 已公开的数据集资产（`--with-dataset`） | 存在经验证的 URL 时自动下载 |
| 与原始源码锁完全一致的上游 native 文件 | 从固定官方 Commit 自动下载 |
| 未公开的定制 native / B01 源码归档 | **尚无可确认的在线来源** |
| 原始 ORBIA 短案例或未发布的评测输入 | **尚无可确认的在线来源** |
| FlashAttention-2（全部 GPU，模型原生依赖） | 自动下载并验证固定版本官方 wheel 后安装 |
| FlashAttention-3（仅 Hopper SM90） | 从固定的官方源码构建并安装当前参考版本 |
| 项目特定 B01 构建和完整视频生成 | **`bootstrap` 不会执行** |

**重要：** `bootstrap` 成功代表在线 Python 环境和可获取资产准备完成，**不代表已经完成端到端视频生成部署**。原始推理路径仍需要部分尚无可信公开下载地址的定制 native 文件及可校验的原始输入案例。安装器不会用同名但哈希不同的上游文件替代，也不会在没有生成视频的情况下声称 E2E 成功。剩余前置条件见[原生依赖要求](docs/INSTALL.md#native-dependency-preflight)和[详细 Quickstart](docs/QUICKSTART.md)。

## 命令行参数说明

| 命令 / 参数 | 用途 |
| --- | --- |
| `bootstrap --backbone MODEL` | 安装模型专属 Python 环境，下载可用的在线资产及精确匹配的公开 native 源码 |
| `--with-dataset` | 同时下载已公开、带 SHA-256 锁定的数据集资产；不代表包含未发布的原始案例 |
| `--env-root DIR` / `--asset-root DIR` | 指定外部环境、模型资产与缓存的存储目录 |
| `--python "$(which python)"` | 可选：显式指定当前激活的 Conda Python 3.10 |
| `--env-mode {auto,conda,venv}` | 默认自动使用已激活的非 base Conda 环境，`venv` 保留旧版独立虚拟环境流程 |
| `--dry-run` | 仅查看安装计划，不创建环境或下载 |
| `environment` / `assets` | 单独执行 Python 环境安装或在线资产下载 |
| `native-deps` | 检查指定 Backbone 的原始 native 源码锁依赖 |
| `install` / `prepare` / `run` | 高级原生构建、来源校验后的输入准备和视频推理（还需其他原始前置资产） |

完整参数请运行 `./scripts/draftkv.sh bootstrap --help`。进一步使用说明见[安装文档](docs/INSTALL.md)、[Quickstart](docs/QUICKSTART.md) 和 [Profiling](docs/PROFILING.md)。

## 仓库结构

- `versions/` — 按 Backbone 保留的原始实现和配置。
- `scripts/draftkv.sh` — 公开命令行入口。
- `scripts/` 和 `tools/` — 环境安装、校验下载、源码绑定和原生编译工具。
- `docs/` — 部署说明、Profiling 选项与[第三方许可声明](docs/THIRD_PARTY_NOTICES.md)。
- `tests/` — 离线单元测试和集成测试。

## 许可证与致谢

项目自有代码采用 [Apache-2.0](LICENSE) 许可证。第三方源码、模型权重和数据集遵循各自的上游许可与访问要求，详见[第三方许可声明](docs/THIRD_PARTY_NOTICES.md)。

DraftKV 基于 [LongLive](https://github.com/NVlabs/LongLive)、[LingBot-World](https://github.com/Robbyant/lingbot-world-v2) 和 [SolarWM](https://github.com/Junchao-cs/SolarWM) 等模型体系构建。
