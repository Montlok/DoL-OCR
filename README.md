# DoL-OCR

DoL-OCR 是面向传统蒙古文的 OCR 训练、评测与推理工程仓库。当前模型目标是
传统蒙古文识别；Tokenizer 对中、英、日及西里尔字符的覆盖用于混排页面处理，
不代表这些语言已经完成同等规模训练。

本仓库只保存可审查的源码、配置、测试和小型示例。训练语料、真实照片、模型权重、
检查点及运行日志不得提交到 Git。

## 架构

OCR 路径由三个主要部分组成：

1. `OMVT` 视觉塔提取纵排、横排、局部和版面特征；
2. 视觉投影层将压缩后的视觉 token 映射到 RDT 隐空间；
3. RDT 语言侧执行生成式识别，训练时可按阶段冻结语言侧或视觉侧。

仓库同时保留 byte-level CTC、生成式 OCR、DPO/GRPO 等训练与评估组件。具体实验
必须在 PR 或运行记录中声明所用数据契约、冻结范围、检查点来源和停止条件。

## 目录

| 路径 | 作用 |
| --- | --- |
| `Model/` | RDT、OMVT、训练循环、OCR 与后训练实现 |
| `Tokenizer/` | 传统蒙古文 MorphBPE、多语种 BPE 与多模态处理 |
| `Encoding Mapping/` | 蒙古文编码映射 Rust crate |
| `scripts/` | 数据准备、训练、评测与烟雾测试入口 |

详细说明见 [Model/README.md](Model/README.md)、
[Tokenizer/README.md](Tokenizer/README.md) 和
[Encoding Mapping/README.md](Encoding%20Mapping/README.md)。

## 本地环境

最低支持 Python 3.10。CPU 开发与测试环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,image,model]"
```

CUDA 正式训练还需要训练与视觉依赖：

```bash
python -m pip install -e ".[train,vis,image,log]"
```

生产配置要求官方 `mamba-ssm`；CPU 测试使用仓库内的回退实现。

## 质量门禁

提交前运行：

```bash
python scripts/check_repository_hygiene.py
ruff check Model Tokenizer scripts smoke_two_stage.py
python -m pytest -q Model/tests Tokenizer/tests
cargo test --locked --manifest-path "Encoding Mapping/Cargo.toml"
cargo fmt --check --manifest-path "Encoding Mapping/Cargo.toml"
python -m build
```

GitHub Actions 对 Pull Request 和 `main` 推送执行同一组 CPU 门禁。

## 数据与权重边界

- 不提交语料、扫描件、真实照片、完整转写或 NAS 路径清单；
- 不提交 `.pt`、`.pth`、`.ckpt`、`.safetensors`、日志或训练输出；
- 示例只能使用小型、可公开且来源明确的数据；
- 真实拍照平行语料与锁定评测集必须保存在 Git 之外；
- 发现误提交时立即停止传播，并按 [SECURITY.md](SECURITY.md) 报告。

## 协作

每个 PR 应保持单一职责，说明动机、兼容性、验证结果和回滚方式。完整要求见
[CONTRIBUTING.md](CONTRIBUTING.md)。仓库采用
[Apache License 2.0](LICENSE)。
