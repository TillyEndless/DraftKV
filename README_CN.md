# DraftKV

[English](README.md) | **简体中文**

**DraftKV: Protecting Persistent Worlds through Hierarchical Streaming Retrieval**

DraftKV 是面向长时视频生成和交互式世界模型的**免训练历史 KV 检索框架**。它结合相机几何与 Attention 相关性，从长历史中找回有用的上下文，并将选中的全分辨率 KV 从 CPU 显存外存储高效传输至 GPU 执行 Attention。

仓库提供 LongLive-1.3B、LingBot-World 1.3B/14B 和 SolarWM-33B 的原始实现，以及用于环境安装、输入准备和推理执行的命令行工具。

## 方法概览

DraftKV 包含三个核心模块：

- **Draft Grounding：** 基于相机视锥和几何关系筛选空间相关的历史候选。
- **Draft Refinement：** 使用紧凑的 Draft Attention 相关性分数，进一步选择历史 KV Blocks。
- **Streaming CG-Attention：** 将 CPU 到 GPU 的 KV 传输与分块 Attention 计算重叠执行。

检索过程无需重新训练模型。全分辨率历史 KV 可以存储在 CPU 内存中，紧凑 Draft 描述则常驻 GPU。

### 支持的模型

| 模型 | 源码目录 |
| --- | --- |
| LongLive-1.3B | `versions/longlive` |
| LingBot-World 1.3B | `versions/lingbot13` |
| LingBot-World 14B | `versions/h200_large` |
| SolarWM-33B | `versions/h200_large` |

运行 `./scripts/draftkv.sh list` 可以查看对应的 CLI 模型标识。

## 安装

当前文档中的原生部署环境为 **Linux x86_64 + NVIDIA H200（SM90）**。安装前需准备 Python 3.10、兼容的 CUDA Toolkit 和编译器、Ninja，以及对应版本的 CPython 开发头文件。模型权重、受限数据集、原始 native 依赖和 ABI 兼容的 FlashAttention-2 wheel 需依据各自的许可单独获取。

克隆仓库：

```bash
git clone https://github.com/TillyEndless/DraftKV-public.git
cd DraftKV-public
./scripts/draftkv.sh list
```

以下以 LingBot-World 1.3B 为例安装环境和模型资产：

```bash
./scripts/draftkv.sh install \
  --backbone lingbot1.3b \
  --python python3.10 \
  --cuda-home /usr/local/cuda-12.6 \
  --env-root /external/draftkv-envs \
  --asset-root /external/draftkv-assets \
  --dependency-root /external/authorized-native-sources \
  --fa2-wheel /external/wheels/flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl \
  --fa2-sha256 REPLACE_WITH_ACTUAL_SHA256 \
  --python-dev-root /external/python-dev/usr/include \
  --authorized-gated
```

请将示例路径和 SHA-256 占位符替换为实际值。FA2 wheel 必须与选择的 Python/PyTorch ABI 兼容。`--authorized-gated` 表示用户已经获得上游授权，并不会自动申请权限。添加 `--dry-run` 可以在不执行安装的情况下查看配置。

安装器会构建固定版本的参考 FA3 和所需的 B01 native 组件；这不意味着所编译的二进制文件与历史修改版本完全一致。依赖要求及验证命令见[安装说明](docs/INSTALL.md)。

## LingBot-World 快速开始

安装完成后，从**获得授权的原始 ORBIA 数据集**中准备一个案例：

```bash
./scripts/draftkv.sh prepare \
  --backbone lingbot1.3b \
  --env-root /external/draftkv-envs \
  --dataset-root /external/orbia \
  --index /external/orbia/installed-files.json \
  --index-sha256 REPLACE_WITH_ACTUAL_INDEX_SHA256 \
  --lock /external/orbia/orbia480.lock.json \
  --case-id YOUR_ORBIA_CASE_ID \
  --output /external/prepared/lingbot13
```

当源码仓库中没有原始数据分发锁时，需要提供 `--lock`。输入必须包含经过源码校验的真实 Prompt、视频帧和相机/控制信息；不能直接以任意图片或虚构相机轨迹替代。LongLive 使用单独的原始 WorldAttention 清单，可通过 `prepare --manifest` 指定。

可以先检查 native 扩展是否能够编译并导入，**不进行视频生成**：

```bash
./scripts/draftkv.sh run \
  --backbone lingbot1.3b \
  --env-root /external/draftkv-envs \
  --asset-root /external/draftkv-assets \
  --input /external/prepared/lingbot13/cases/YOUR_ORBIA_CASE_ID/request.json \
  --output /external/runs/lingbot13-startup/native.mp4 \
  --gpu 0 --profile off --startup-only
```

检查通过时会返回 `PORTABLE_NATIVE_STARTUP_READY`，但**不会加载模型或生成视频**。正式生成时使用**新的输出目录**，去掉 `--startup-only`：

```bash
./scripts/draftkv.sh run \
  --backbone lingbot1.3b \
  --env-root /external/draftkv-envs \
  --asset-root /external/draftkv-assets \
  --input /external/prepared/lingbot13/cases/YOUR_ORBIA_CASE_ID/request.json \
  --output /external/runs/lingbot13-generation/native.mp4 \
  --gpu 0 --profile off
```

虚拟环境、模型权重、数据集、native 编译缓存、准备后的输入和生成结果均应存储在 **Git 仓库目录之外**。模型专用输入要求见[快速开始指南](docs/QUICKSTART.md)。

### 主要命令与参数

| 命令或参数 | 说明 |
| --- | --- |
| `list` / `install` / `prepare` / `run` | 列出模型 / 安装依赖 / 准备已验证输入 / 检查 native 扩展或生成视频。 |
| `--backbone` | `longlive1.3b`、`lingbot1.3b`、`lingbot14b` 或 `solarwm33b`。 |
| `--env-root` / `--asset-root` | 外部模型环境与资产目录。 |
| `--python` / `--cuda-home` / `--python-dev-root` | Python 3.10、CUDA Toolkit 路径和对应的 CPython 头文件（安装阶段）。 |
| `--dependency-root` | 获授权的原始 native 源码，通过 SHA-256 校验（安装阶段）。 |
| `--fa2-wheel` / `--fa2-sha256` | ABI 兼容的 FA2 wheel 和实际小写 SHA-256（安装阶段）。 |
| `--authorized-gated` / `--with-dataset` | 确认已获得上游权限 / 同时安装数据集资产（安装阶段）。 |
| `--dataset-root` / `--case-id` | 原始数据集目录和指定案例（准备输入阶段）。 |
| `--index` / `--index-sha256` / `--lock` | ORBIA 原始索引、哈希及必要时的外部分发锁（准备输入阶段）。 |
| `--manifest` / `--native-outputs` | LongLive 原始清单与输出策略（准备输入阶段）。 |
| `--input` / `--output` | 经过源码校验的 Request/Plan 和新的输出路径（运行阶段）。 |
| `--native-source` | 必要时指定获授权的 SolarWM 原始 native 源码目录（运行阶段）。 |
| `--gpu` | 物理 GPU 编号，默认 `0`（运行阶段）。 |
| `--profile off|summary` | `off` 用于正常运行；`summary` 用于部署诊断，不等同于论文基准测试。 |
| `--startup-only` | 构建/导入 native 扩展后退出，不执行模型推理（运行阶段）。 |
| `--dry-run` / `--jobs` | 预览安装/运行配置 / 设置 native 编译并行数（适用时）。 |

完整参数见 `./scripts/draftkv.sh install --help`、`prepare --help` 和 `run --help`。Profiling 模式说明见 [Profiling](docs/PROFILING.md)。

## 仓库结构

- `versions/` — 按 Backbone 保留的源码实现和科学配置。
- `scripts/draftkv.sh` — 对外提供的安装、输入准备和推理入口。
- `scripts/` 与 `tools/` — 依赖安装和运行时集成工具。
- `docs/` — [Quickstart](docs/QUICKSTART.md)、[安装说明](docs/INSTALL.md)、[Profiling](docs/PROFILING.md) 和[第三方许可声明](docs/THIRD_PARTY_NOTICES.md)。
- `tests/` — CPU 单元测试和集成测试。

仓库还提供校验与部署清单，用于验证原始源码、外部依赖、安装资产和原始执行入口。

## 许可证与致谢

项目自有代码采用 [Apache-2.0](LICENSE) 许可证。第三方组件遵循各自的许可条款，部分包含非商业使用限制；分发或使用前请阅读[第三方许可声明](docs/THIRD_PARTY_NOTICES.md)。

DraftKV 集成了 [LongLive](https://github.com/NVlabs/LongLive)、[LingBot-World](https://github.com/robbyant/lingbot-world-v2) 和 [SolarWM](https://github.com/Junchao-cs/SolarWM) 等模型。该源码发布版本不包含模型权重、受限数据集和私有 native 源码包。
