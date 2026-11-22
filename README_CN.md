# DraftKV

[English](README.md) | **简体中文**

**DraftKV: Protecting Persistent Worlds through Hierarchical Streaming Retrieval**

DraftKV 是面向长时视频生成与交互式世界模型的**免训练历史 KV 检索框架**，通过 **Draft Grounding** 与 **Draft Refinement** 实现分层历史检索，并借助 **Streaming CG-Attention** 高效完成 CPU 到 GPU 的 KV 访问与 Attention 计算。

## 支持的模型

| 模型 | 源码目录 |
|---|---|
| LongLive-1.3B | `versions/longlive` |
| LingBot-World 1.3B | `versions/lingbot13` |
| LingBot-World 14B | `versions/h200_large` |
| SolarWM-33B | `versions/h200_large` |

## 快速部署（Quick Start）

以下以 **Linux x86_64 + NVIDIA H200（SM90）+ Python 3.10 + LingBot-World 1.3B** 为例。需要事先准备对应 CUDA Toolkit、获授权的模型与数据集，以及**匹配 Python/PyTorch ABI 的 FlashAttention-2 wheel** 和原始 native 依赖。本仓库只发布源码，不提供完整权重与第三方依赖。

**1. 克隆仓库并查看模型列表**

```bash
git clone https://github.com/TillyEndless/DraftKV-public.git
cd DraftKV-public
./scripts/draftkv.sh list
```

**2. 安装模型环境、原生依赖与模型资产**

```bash
./scripts/draftkv.sh install \
  --backbone lingbot1.3b \
  --python python3.10 \
  --cuda-home /usr/local/cuda-12.6 \
  --env-root /external/draftkv-envs \
  --asset-root /external/draftkv-assets \
  --dependency-root /external/authorized-native-sources \
  --fa2-wheel /external/wheels/flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl \
  --fa2-sha256 REPLACE_WITH_ACTUAL_64_CHARACTER_SHA256 \
  --python-dev-root /external/python-dev/usr/include \
  --authorized-gated
```

将以上路径及 wheel SHA-256 替换为真实值；wheel 文件名只是示例。`--authorized-gated` 表示用户已自行获得上游访问许可。安装器会编译固定版本的参考 FA3 和带有独立校验记录的 B01，**不代表恢复论文历史修改版 FA3 二进制文件**。可以添加 `--dry-run` 先检查安装参数。

**3. 准备经过源码校验的 ORBIA 案例**

```bash
./scripts/draftkv.sh prepare \
  --backbone lingbot1.3b \
  --env-root /external/draftkv-envs \
  --dataset-root /external/orbia \
  --index /external/orbia/installed-files.json \
  --index-sha256 REPLACE_WITH_ACTUAL_INDEX_SHA256 \
  --case-id YOUR_ORBIA_CASE_ID \
  --output /external/prepared/lingbot13
```

必须提供**获授权的原始 ORBIA 数据集及可验证索引**。如果源码中没有原始数据分发锁，还需传入 `--lock /external/orbia/orbia480.lock.json`。LongLive 使用单独的 WorldAttention 原始清单（`--manifest`），详见 [Quickstart](docs/QUICKSTART.md)。不能以普通照片、任意 Prompt 或虚构相机轨迹替代原始输入案例。

**4. 检查原生扩展编译和导入（不生成视频）**

```bash
./scripts/draftkv.sh run \
  --backbone lingbot1.3b \
  --env-root /external/draftkv-envs \
  --asset-root /external/draftkv-assets \
  --input /external/prepared/lingbot13/cases/YOUR_ORBIA_CASE_ID/request.json \
  --output /external/runs/lingbot13-startup/native.mp4 \
  --gpu 0 --profile off --startup-only
```

检查通过时返回 `PORTABLE_NATIVE_STARTUP_READY`，**不会生成视频**。GPU 即使有其他任务，只要 CUDA 资源充足仍可进行功能检查；但共享 GPU 不适合测量正式延迟。每次运行都要使用**新的输出目录**。

**5. 正式执行视频生成**

```bash
./scripts/draftkv.sh run \
  --backbone lingbot1.3b \
  --env-root /external/draftkv-envs \
  --asset-root /external/draftkv-assets \
  --input /external/prepared/lingbot13/cases/YOUR_ORBIA_CASE_ID/request.json \
  --output /external/runs/lingbot13-generation/native.mp4 \
  --gpu 0 --profile off
```

环境、权重、输入 Plan、编译缓存和生成的视频应全部位于 **Git 仓库目录之外**。更完整的前置条件见[安装说明](docs/INSTALL.md)，Profiling 选项见 [Profiling](docs/PROFILING.md)。

## 主要命令与参数说明

| 命令 / 参数 | 含义 |
|---|---|
| `list` | 列出支持的 Backbone。 |
| `install` / `prepare` / `run` | 安装环境与依赖 / 准备已验证输入 / 执行原生检查或视频生成。 |
| `--backbone` | `longlive1.3b`、`lingbot1.3b`、`lingbot14b` 或 `solarwm33b`。 |
| `--env-root` / `--asset-root` | 外部虚拟环境目录与模型资产目录。 |
| `--python` / `--cuda-home` / `--python-dev-root` | Python 3.10、CUDA Toolkit 根目录、匹配的 CPython 开发头文件（安装阶段）。 |
| `--dependency-root` | 获授权的原始 native 依赖，安装时按仓库 SHA-256 校验。 |
| `--fa2-wheel` / `--fa2-sha256` | ABI 兼容的 FlashAttention-2 wheel 及实际小写 SHA-256（安装阶段）。 |
| `--authorized-gated` / `--with-dataset` | 确认已获得上游受限资产访问权限 / 选择同时安装数据集资产（安装阶段）。 |
| `--dataset-root` / `--case-id` | 原始数据集根目录和指定案例 ID（准备输入阶段）。 |
| `--index` / `--index-sha256` / `--lock` | ORBIA 原始索引、索引哈希，以及必要时外部提供的数据分发锁（准备输入阶段）。 |
| `--manifest` / `--native-outputs` | 获授权的 LongLive WorldAttention 原始清单与输出策略（仅 LongLive 的输入准备）。 |
| `--input` / `--output` | 已准备好的 Request 或原始 Plan；**全新**视频输出路径（运行阶段）。 |
| `--native-source` | 必要时提供获授权的 SolarWM 原始 native 源码目录（仅 SolarWM 运行阶段）。 |
| `--gpu` | `nvidia-smi` 物理 GPU 编号，默认为 `0`（运行阶段）。 |
| `--profile off|summary` | `off`（默认）：功能运行，不新增外层 Profile；`summary`：部署诊断，**不等于论文正式 E2E 测量**。 |
| `--startup-only` | 编译/导入原始 native 扩展后退出，不生成视频（仅用于 `run --profile off`）。 |
| `--dry-run` | 在 `install` 或 `run` 时预览执行选择，不实际执行。 |
| `--jobs` / `--build-fa3` | 原生编译并行任务数和必需的固定版本 FA3 构建（`install`，默认启用构建）。 |

查看完整参数：`./scripts/draftkv.sh <command> --help`。原始底层执行入口：`./run.sh --help`。

## 文档与发布范围

[详细 Quickstart](docs/QUICKSTART.md) · [安装说明](docs/INSTALL.md) · [Profiling 参数](docs/PROFILING.md) · [第三方许可声明](docs/THIRD_PARTY_NOTICES.md)

仓库保留四份必要 JSON 清单，分别用于[冻结源码哈希](RETAINED_SOURCE_SHA256.json)、[外部依赖哈希](EXTERNAL_DEPENDENCY_SHA256.json)、[原始命令映射](ORIGINAL_COMMAND_MAPPING.json)和[环境/资产安装](scripts/install_manifest.json)；文件内容和路径均未变动。

项目自有代码采用 [Apache-2.0](LICENSE) 许可证，第三方代码、模型权重及受限数据集遵循各自的许可和访问要求。本仓库为仅含源码的发布版本，不包含模型权重、受限数据集或私有 native 源码包。
