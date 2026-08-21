# Contributing to DoL-OCR

DoL-OCR 接受能够维护或验证当前预训练、AnyRes SFT、GRPO、推理和锁定评测血统的贡献。仓库不是语料库、模型制品仓库或通用实验代码收集区。

## 1. 范围

可以接受的改动包括：

- RDT two-stage 预训练、checkpoint/resume 和官方 Mamba 集成；
- OMVT、CTC、VLM alignment 和 AnyRes native-detail 视觉路径；
- Visual SFT、Joint SFT、text replay、GRPO 和 locked evaluation；
- 与上述路径直接相关的 Tokenizer、数据契约、评测、测试、文档和 CI；
- 能阻止错误数据、错误权重、泄漏或不可重放训练的 fail-closed 检查。

以下内容不属于本仓库：

- 与当前正式 checkpoint 无关的实验架构、退役训练线或兼容壳；
- 仅由自己的测试、文档或 CI 引用，实际不被生产路径调用的工具岛；
- 通用聊天 SFT/DPO、旧固定尺寸 OCR-GRPO 或未实现的模型能力；
- 爬虫、批量采集器、语料下载器及外部平台抓取脚本；
- 原始或处理后的语料、转写、图片、扫描件、数据 manifest、训练 receipt；
- 权重、checkpoint、优化器状态、日志、分析 dump、缓存或压缩包。

任何需要联网获取数据的流程都应在受控的外部数据系统运行。本仓库只接收数据契约和不包含数据内容的验证逻辑。

## 2. 不可破坏的模型血统

AnyRes 后训练的唯一认可来源是：

```text
Montlok/DoL-1.2-OCR@bee908ab2a9376f6224dff514564ebb0ae99a643
```

[`Model/posttrain/release_locks/dol_1_2_ocr.json`](Model/posttrain/release_locks/dol_1_2_ocr.json) 中的哈希和契约是代码审查依据。修改以下任一内容时，PR 必须说明旧 checkpoint 是否仍可 strict-load，并提供对应测试：

- vocabulary capacity、token IDs 或 Tokenizer bundle identity；
- `boundary_v1` position contract；
- native OCR target encoding；
- RDT、OMVT、projector、bridge 参数名称或形状；
- checkpoint、resume、selection、journal、receipt 或 locked-eval 格式。

不得为了让测试通过而弱化来源检查、跳过 SHA 验证或给未知 checkpoint 增加静默 fallback。

## 3. 数据和评测边界

贡献者必须确认：

- 仓库 diff 不包含 `.jsonl`、`.csv`、`.tsv`、图片、音视频、PDF 或数据导出；
- 示例使用内存中构造的最小 fixture，不提交假数据文件；
- train、validation、KL selection、formal monitor 和 locked benchmark 没有 source/hash/group/path 重叠；
- locked benchmark 不会被训练进程、pilot 或超参数选择读取；
- 任何外部语料均具有明确来源、授权和删除机制，且不随 Git 分发；
- 测试集不是训练集的复制、别名或重新切分结果。

如果误提交数据、凭据或受限制内容，应立即停止传播并按 [`SECURITY.md`](SECURITY.md) 报告；普通删除提交不等于从历史和托管端引用中清除。

## 4. 新脚本和工具

`scripts/` 与 `Tokenizer/tools/` 使用精确白名单。新增入口的 PR 必须同时回答：

1. 它服务哪一个当前生产阶段；
2. 为什么现有入口不能承担该职责；
3. 谁消费它的输出，输出由什么 contract/receipt 约束；
4. 它是否访问网络、下载数据或写入工作树；
5. 如何测试、回滚和从白名单移除。

没有上述证据的入口不应加入白名单。不要通过在测试、README 或 CI 中添加自引用来证明一个脚本“有用”。

## 5. 开发环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,image,model]"
```

需要 CUDA 训练依赖时：

```bash
python -m pip install -e ".[train,vis,image,log]"
```

不要把本机地址、挂载点、运行 ID、访问令牌或数据路径写进仓库。机器专用操作记录放入未跟踪的 `RUNBOOK.local.md` 或外部运行系统。

## 6. 必跑检查

所有 PR 至少运行：

```bash
python scripts/check_repository_hygiene.py
ruff check Model Tokenizer scripts
python -m pytest -q Model/tests Tokenizer/tests
python -m build
```

如果环境无法执行某项检查，PR 必须准确写明“未运行”及原因，不能把静态阅读描述成测试通过。

涉及 CUDA、official Mamba、BF16、FSDP、视觉塔或真实 checkpoint 的改动，还必须在目标 Linux/CUDA 环境补充与风险相称的验证。CPU tiny test 不能替代生产后端验收。

## 7. 测试要求

- 修复 bug 时先添加能复现旧错误的测试；
- 数学恒等优化必须比较新旧输出，并覆盖 backward/gradient；
- checkpoint 改动必须覆盖保存、resume、hash drift 和不完整写入；
- 数据 contract 改动必须覆盖路径穿越、symlink、重复 hash、split overlap 和损坏输入；
- GRPO 改动必须覆盖行为 logprob 对齐、reference identity、no-update journal 和 deterministic resume；
- locked evaluation 改动必须保留 one-shot claim 和失败后仍消费 claim 的语义。

测试不能通过跳过真实生产分支来“证明”等价。对于依赖 GPU 的行为，应保留可在 CPU 上验证的结构契约，并另外报告 GPU 验收。

## 8. PR 结构

PR 描述应包含：

- **Summary**：改变了什么；
- **Motivation**：为何属于当前血统；
- **Compatibility**：对官方 release lock、数据、checkpoint 和 resume 的影响；
- **Validation**：实际运行的完整命令和结果；
- **Risk and rollback**：失败模式及可恢复方案；
- **Files intentionally removed**：删除项及为什么不影响预训练或强化学习。

一个 PR 只承担一个主要职责。不要把格式化全仓、历史清理和算法改动混在同一普通功能 PR 中；仓库重建或历史重写必须显式标注为治理操作。

## 9. 提交前清单

- [ ] 改动只服务当前预训练或 AnyRes 后训练血统；
- [ ] 官方 `Montlok/DoL-1.2-OCR` release lock 未被绕过；
- [ ] 没有语料、媒体、权重、checkpoint、日志、凭据或本机路径；
- [ ] 没有新增爬虫、下载器或未批准工具入口；
- [ ] 新增脚本已加入精确白名单并提供生产消费者证据；
- [ ] 数据 split 和 locked benchmark 边界未被削弱；
- [ ] 实际运行的检查及未运行项均已如实记录；
- [ ] 文档只描述当前存在且被支持的入口。

## 10. License and conduct

贡献须由提交者合法授权，并按 [Apache License 2.0](LICENSE) 提供。不得提交无权分发的第三方表达、个人信息、保密信息、恶意代码或隐藏 payload。参与者应围绕技术证据讨论，尊重语言文化、标注者和数据权利人。
