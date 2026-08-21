# DoL-OCR

DoL-OCR 是面向传统蒙古文的 OCR 训练、评测与推理源码仓库。仓库只维护两条相互衔接的生产血统：

1. RDT 语言模型与 OMVT 视觉塔的预训练、对齐和检查点契约；
2. 从官方 DoL-1.2-OCR 初始化的 AnyRes Visual SFT、Joint SFT、GRPO 和一次性锁定评测。

仓库不保存语料、图片、扫描件、标注、模型权重、检查点或运行输出。正式数据和权重必须位于 Git 工作树之外，并由不可变 receipt、SHA-256 和显式路径绑定。

## 唯一认可的初始化模型

AnyRes 后训练只接受以下不可变发布：

```text
Montlok/DoL-1.2-OCR@bee908ab2a9376f6224dff514564ebb0ae99a643
```

仓库内的 [`Model/posttrain/release_locks/dol_1_2_ocr.json`](Model/posttrain/release_locks/dol_1_2_ocr.json) 固定了仓库 ID、revision、模型、metadata、Tokenizer 和训练契约哈希。不得用文件名、`latest`、步骤别名、目录顺序或其他模型替代该来源。

## 架构

- `RDTForCausalLM`：two-stage recurrent-depth 语言核心，正式 CUDA 训练使用官方 Mamba；
- `OMVT`：保留全局锚点并编码纵排、横排、局部及版面视觉信息；
- AnyRes native detail：按原始宽高规划多尺度窗口，不要求用户预先裁成固定尺寸；
- Visual SFT：冻结语言模型和全局锚点，训练 native detail tower、bridge 与 projector；
- Joint SFT：OCR 与独立 text replay 联合训练；
- GRPO：先做三个隔离 KL pilot，再从同一 eligible joint checkpoint 启动 formal run；
- locked evaluation：正式选择完成后才允许一次性打开密封评测集。

详细实现与契约见 [`Model/README.md`](Model/README.md)，操作顺序见 [`RUNBOOK.md`](RUNBOOK.md)。

## 目录

| 路径 | 内容 |
| --- | --- |
| `Model/` | RDT、OMVT、训练循环、AnyRes SFT/GRPO、检查点与评测契约 |
| `Tokenizer/` | 传统蒙古文 MorphBPE、多语种 byte-level BPE、多模态 token 契约 |
| `scripts/` | 经仓库卫生门禁明确允许的数据构建、训练、评测和监控入口 |
| `.github/workflows/ci.yml` | Python 仓库边界、静态检查、测试和构建门禁 |

## 安装

最低 Python 版本为 3.10。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,image,model]"
```

正式 CUDA 训练还需要：

```bash
python -m pip install -e ".[train,vis,image,log]"
```

生产配置必须证明官方 `mamba-ssm` 可用；CPU 测试中的 NaiveSSM 只用于测试，不构成生产后端验收。

## 生产流程

```text
外部语料 + Tokenizer bundle
  -> receipt-bound RDT/OMVT/VLM 预训练
  -> 官方 DoL-1.2-OCR release lock
  -> AnyRes 人工审核与 READY admission
  -> Visual SFT
  -> Joint SFT + text replay
  -> KL pilot x 3
  -> Formal GRPO
  -> one-shot locked evaluation
```

AnyRes 数据必须保留原始宽高和完整像素。内部窗口可以有重叠上下文，但 cut-QA 必须证明每个连通文本组件至少在一个视图中完整可见。固定方形图仅可作为冻结的 v1 global anchor，不能取代 native detail 输入。

## 质量门禁

提交前运行与 CI 相同的命令：

```bash
python scripts/check_repository_hygiene.py
ruff check Model Tokenizer scripts
python -m pytest -q Model/tests Tokenizer/tests
python -m build
```

仓库卫生门禁会拒绝：

- JSONL、CSV、TSV、图片、音视频、PDF 和其他数据媒体；
- 权重、检查点、数据库、数组、日志、归档包和训练输出；
- 未列入生产白名单的 `scripts/` 或 `Tokenizer/tools/` 入口；
- 本地语料目录名、已退役编码目录和超过大小上限的文件。

## 数据与评测边界

- train、`sft_validation`、`kl_selection`、`formal_monitor` 必须按 source、hash、document group 和 resolved path 隔离；
- locked benchmark 必须物理隔离，不得出现在公共 manifest 中；
- 训练、调参和 pilot 不得读取 locked labels 或 pixels；
- 报告必须绑定代码 SHA、checkpoint hash、dataset receipt、生成参数和原始输出；
- 测试图像不得来自训练范围，不能把训练数据重新命名为评测集。

## 贡献

PR 必须单一职责，说明来源、兼容性、验证结果和回滚方式。新增训练入口必须先证明它属于当前预训练或 AnyRes 后训练血统，再加入显式白名单。详见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

许可证为 [Apache License 2.0](LICENSE)。安全、数据泄露或权利问题请按 [`SECURITY.md`](SECURITY.md) 报告。
