# Profiling 选项

[English](PROFILING.md) | **简体中文**

DraftKV 的可移植命令行入口支持以下两种模式：

| 模式 | 用途 |
| --- | --- |
| `--profile off`（默认） | 运行原生 Worker，不额外请求外层 Profiling 汇总；适用于功能检查与视频生成。 |
| `--profile summary` | 在外部运行目录生成 `portable_profile_summary.json`，记录可移植执行诊断信息。**它不是标准化 Benchmark，也不是论文中的延迟测量。** |

## Native 扩展启动检查

如果只想验证原始 Native 扩展能够编译并成功导入，而不生成视频，可执行：

```bash
./scripts/draftkv.sh run \
  --backbone lingbot1.3b \
  --env-root /external/draftkv-envs \
  --asset-root /external/draftkv-assets \
  --input /external/prepared/lingbot13/cases/YOUR_ORBIA_CASE_ID/request.json \
  --output /external/runs/native-startup/native.mp4 \
  --gpu 0 --profile off --startup-only
```

输出 `PORTABLE_NATIVE_STARTUP_READY` 仅表示 Native 扩展已成功导入，**不代表**模型完成 Forward，也不代表已生成视频。每次运行应使用新的外部输出目录。

## 视频生成与可选诊断

```bash
./scripts/draftkv.sh run \
  --backbone lingbot1.3b \
  --env-root /external/draftkv-envs \
  --asset-root /external/draftkv-assets \
  --input /external/prepared/lingbot13/cases/YOUR_ORBIA_CASE_ID/request.json \
  --output /external/runs/generation/native.mp4 \
  --gpu 0 --profile summary
```

当所需 Native Instrumentation 可用时，`--profile summary` 会记录部署诊断信息。普通视频生成建议使用 `--profile off`。如果共享 GPU 仍有足够资源，可以用于功能检查；但可靠的延迟 Benchmark 需要适当隔离 GPU，并保持工作负载一致。**不要将诊断汇总视为 Benchmark 结果。**

命令与输入准备见[快速开始](QUICKSTART_CN.md)；前置条件和依赖见[安装指南](INSTALL_CN.md)。原始底层命令入口可通过 `./run.sh --help` 查看，该入口要求另行准备符合原始 Schema 的运行计划及外部依赖。
