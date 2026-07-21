# 贡献指南 / Contributing Guide

> 中文版本与英文版本如有理解差异，在面向中文用户、项目治理和文化宗旨说明时，以中文版本为主要解释文本。  
> In case of interpretive differences between the Chinese and English versions, the Chinese version shall serve as the primary interpretive text for Chinese-facing project governance and cultural-purpose statements.

---

## 目录 / Table of Contents

- [1. 项目宗旨 / Project Purpose](#1-项目宗旨--project-purpose)
- [2. 仓库性质 / Repository Scope](#2-仓库性质--repository-scope)
- [3. 我们欢迎的贡献 / Contributions We Welcome](#3-我们欢迎的贡献--contributions-we-welcome)
- [4. 不接受的贡献 / Contributions We Do Not Accept](#4-不接受的贡献--contributions-we-do-not-accept)
- [5. 开源许可与贡献授权 / License and Contribution Grant](#5-开源许可与贡献授权--license-and-contribution-grant)
- [6. 版权、语料与模型边界 / Copyright, Corpus, and Model Boundary](#6-版权语料与模型边界--copyright-corpus-and-model-boundary)
- [7. 贡献流程 / Contribution Workflow](#7-贡献流程--contribution-workflow)
- [8. 测试、格式化与质量要求 / Testing, Formatting, and Quality Requirements](#8-测试格式化与质量要求--testing-formatting-and-quality-requirements)
- [9. 安全问题 / Security Issues](#9-安全问题--security-issues)
- [10. 代码风格、文档风格与引用规范 / Code Style, Documentation Style, and Citation Standards](#10-代码风格文档风格与引用规范--code-style-documentation-style-and-citation-standards)
- [11. 贡献者行为准则 / Contributor Conduct](#11-贡献者行为准则--contributor-conduct)
- [12. 权利主张、移除请求与争议内容 / Rights Claims, Takedown Requests, and Disputed Content](#12-权利主张移除请求与争议内容--rights-claims-takedown-requests-and-disputed-content)
- [13. 致谢 / Acknowledgement](#13-致谢--acknowledgement)
- [14. Pull Request Checklist](#14-pull-request-checklist)
- [15. 简明原则 / Short Principle](#15-简明原则--short-principle)

---

## 1. 项目宗旨 / Project Purpose

本项目以保护、传承和弘扬中华民族优秀传统文化为首要宗旨。

本项目所涉蒙古语文本、民族语言文字材料、历史文献、地方知识及相关文化资料，系中华民族优秀传统文化博大精深的重要组成部分。传统蒙古语资料的数字化整理、文本标注、模型辅助学习和工具链开放，对于保护民族语言文字资源、促进民族团结、推动学术繁荣和实现文化传承具有重要意义。

我们欢迎研究者、开发者、母语者、标注者、整理者、教育工作者和文化传承参与者，以依法、审慎、开放、可追溯的方式参与本项目。

This project is established with the primary purpose of protecting, preserving, and promoting the outstanding traditional culture of the Chinese nation.

The Mongolian language materials, ethnic language resources, historical texts, local knowledge, and related cultural materials involved in this project form an important part of the profound and extensive outstanding traditional culture of the Chinese nation. The digital preservation, text annotation, model-assisted learning, and open tooling of traditional Mongolian materials are important for preserving ethnic language resources, promoting ethnic unity, advancing academic prosperity, and supporting cultural continuity.

We welcome researchers, developers, native speakers, annotators, compilers, educators, and cultural contributors to participate in this project in a lawful, careful, open, and traceable manner.

---

## 2. 仓库性质 / Repository Scope

本仓库用于发布、维护和协作改进本项目相关的：

- 模型权重；
- Tokenizer；
- 模型配置文件；
- 推理配置；
- 校验信息；
- 模型卡；
- 评估、加载、验证和使用说明；
- 与模型、Tokenizer、配置和工具链相关的辅助文件。

本仓库不作为原始语料库、扫描件库、完整转写文本库、同行评议材料库或第三方作品分发库使用。

This repository is used to publish, maintain, and collaboratively improve:

- model weights;
- tokenizers;
- model configuration files;
- inference configuration files;
- checksums;
- model cards;
- evaluation, loading, validation, and usage documentation;
- auxiliary files related to models, tokenizers, configurations, and tooling.

This repository is not intended to serve as a raw corpus repository, scan archive, full transcription repository, peer-review material repository, or third-party works distribution repository.

---

## 3. 我们欢迎的贡献 / Contributions We Welcome

### 3.1 模型与权重 / Models and Weights

我们欢迎：

- 模型权重发布、校验和补充说明；
- 模型配置修正；
- 推理配置改进；
- 模型卡完善；
- 模型版本兼容性说明；
- 权重加载、转换、验证和评估流程改进；
- 已知限制、适用场景和风险提示补充。

We welcome:

- model weight releases, checksums, and documentation;
- model configuration fixes;
- inference configuration improvements;
- model card improvements;
- model version compatibility notes;
- improvements to weight loading, conversion, validation, and evaluation workflows;
- additions to known limitations, intended use cases, and risk notes.

### 3.2 Tokenizer

我们欢迎：

- Tokenizer 文件修正；
- 特殊 token 说明；
- 词表边界说明；
- 蒙古文、中文、英文及混合文本处理问题修复；
- Tokenizer 行为测试；
- 与下游模型兼容性相关的说明；
- Unicode、空格、标点、传统蒙古文方向性及编码边界问题说明。

We welcome:

- tokenizer file fixes;
- special-token documentation;
- vocabulary boundary documentation;
- fixes for Mongolian, Chinese, English, and mixed-text handling;
- tokenizer behavior tests;
- downstream model compatibility notes;
- notes on Unicode, whitespace, punctuation, traditional Mongolian directionality, and encoding boundaries.

### 3.3 工具链与脚本 / Tooling and Scripts

我们欢迎：

- 加载脚本；
- 推理示例；
- 评估脚本；
- 格式转换工具；
- 校验工具；
- 安全检查工具；
- CI、测试和格式化流程；
- 权重、Tokenizer、配置文件的完整性验证工具。

We welcome:

- loading scripts;
- inference examples;
- evaluation scripts;
- format-conversion tools;
- verification tools;
- security-checking tools;
- CI, tests, and formatting workflows;
- integrity-verification tools for weights, tokenizers, and configuration files.

### 3.4 文档 / Documentation

我们欢迎：

- README、模型卡和使用说明；
- 安全说明；
- API 或 CLI 文档；
- 贡献流程说明；
- 许可证、引用方式和致谢说明；
- 中英双语文档改进；
- 文化宗旨、模型边界、语料边界和安全边界说明。

We welcome:

- README, model cards, and usage guides;
- security documentation;
- API or CLI documentation;
- contribution workflow documentation;
- license, citation, and acknowledgement notes;
- Chinese-English bilingual documentation improvements;
- documentation on cultural purpose, model boundaries, corpus boundaries, and security boundaries.

### 3.5 问题报告 / Issue Reports

我们欢迎报告：

- 模型加载失败；
- Tokenizer 行为异常；
- 推理配置错误；
- 权重校验失败；
- 文档不清或不一致；
- 兼容性问题；
- 安全、版权或数据泄露风险；
- 可能影响传统蒙古语处理质量的问题。

We welcome reports of:

- model loading failures;
- tokenizer behavior issues;
- inference configuration errors;
- weight checksum failures;
- unclear or inconsistent documentation;
- compatibility issues;
- security, copyright, or data-leakage risks;
- issues that may affect the quality of traditional Mongolian processing.

---

## 4. 不接受的贡献 / Contributions We Do Not Accept

除非维护者另行明确说明，本仓库不接受以下内容：

- 完整原始语料库；
- 完整转写文本；
- 扫描件、图片、影印件；
- 连续长段原文；
- 未公开同行评议材料；
- 个人信息、敏感信息或保密信息；
- 来源不明、授权不明或权利状态不清的第三方受保护作品表达；
- 足以替代原始作品正常阅读、购买或授权使用的内容集合；
- 恶意代码、后门、隐藏 payload、混淆代码或异常二进制文件；
- 违反法律法规、学术伦理或本项目宗旨的内容；
- 可能诱导模型绕过访问控制、版权控制、平台安全或数据保护措施的内容。

Unless otherwise expressly approved by maintainers, this repository does not accept:

- complete raw corpora;
- full transcriptions;
- scans, images, or facsimiles;
- long continuous source-text passages;
- unpublished peer-review materials;
- personal information, sensitive information, or confidential information;
- third-party protected expressions with unclear source, authorization, or rights status;
- content collections that may substitute for normal reading, purchase, or licensed use of original works;
- malicious code, backdoors, hidden payloads, obfuscated code, or abnormal binary files;
- content that violates laws, regulations, academic ethics, or the project purpose;
- content that may induce the model to bypass access controls, copyright controls, platform security, or data-protection measures.

如误提交上述内容，维护者可以隐藏、移除、撤回、替换、重写历史记录、关闭 Pull Request 或采取其他必要措施。

If such content is submitted by mistake, maintainers may hide, remove, withdraw, replace, rewrite history, close the pull request, or take other necessary measures.

---

## 5. 开源许可与贡献授权 / License and Contribution Grant

除非具体文件、目录、模型卡或发布说明另有明确声明，本仓库内容基于 **Apache License 2.0** 发布。

向本仓库提交贡献，即表示贡献者确认：

1. 其有权提交该贡献；
2. 该贡献可以按照 Apache License 2.0 被项目接收、使用、复制、修改、合并、发布、分发、再许可和用于衍生版本；
3. 该贡献不包含贡献者无权处分的第三方受保护作品表达、个人信息、保密信息、同行评议材料或其他受限制内容；
4. 如贡献者希望某项提交不作为贡献，应在提交时以醒目方式标明 `Not a Contribution`；
5. 该贡献不会使本仓库当然成为原始语料库、完整转写文本库、扫描件库或第三方作品分发库。

Unless otherwise expressly stated in a specific file, directory, model card, or release note, the contents of this repository are released under the **Apache License 2.0**.

By submitting a contribution to this repository, the contributor represents that:

1. they have the right to submit the contribution;
2. the contribution may be accepted, used, copied, modified, merged, published, distributed, sublicensed, and used in derivative versions under the Apache License 2.0;
3. the contribution does not contain third-party protected expressions, personal information, confidential information, peer-review materials, or other restricted content that the contributor has no right to submit;
4. if the contributor intends a submission not to be treated as a contribution, it must be conspicuously marked as `Not a Contribution` at the time of submission;
5. the contribution does not make this repository a raw corpus repository, full transcription repository, scan archive, or third-party works distribution repository by default.

Apache License 2.0 不当然授权使用项目名称、机构名称、模型名称、商标、服务标志、徽标或其他品牌标识。贡献者不得使第三方误认为其修改版本、分发版本或衍生版本获得项目方官方背书。

The Apache License 2.0 does not automatically grant permission to use project names, institutional names, model names, trademarks, service marks, logos, or other branding identifiers. Contributors must not imply that their modified, distributed, or derivative versions are officially endorsed by the project maintainers.

---

## 6. 版权、语料与模型边界 / Copyright, Corpus, and Model Boundary

我们尊重作者、译者、整理者、出版机构、馆藏机构、数据库权利人、资料提供方及其他潜在第三方权利人依法享有的合法权益。

同时，本项目坚持以下边界：著作权保护的是依法构成作品的具体表达。著作权不应被扩张解释为对语言文字本身、思想、观点、事实、方法、语言规律、统计结果、索引结构、标注体系、模型参数、向量表示、质量评估指标或不可还原原文的研究结论享有当然控制权。

本仓库发布的是模型权重、Tokenizer、配置、工具链和必要说明文件，不等同于发布训练语料、原始文本或第三方作品本身。

We respect the lawful rights and interests of authors, translators, compilers, publishers, archival institutions, database right holders, material providers, and other potential third-party right holders.

At the same time, this project maintains the following boundary: copyright protects specific protectable expressions. Copyright should not be expansively interpreted as granting automatic control over language itself, ideas, opinions, facts, methods, linguistic patterns, statistical results, index structures, annotation systems, model parameters, vector representations, quality-evaluation metrics, or research conclusions that cannot reconstruct original texts.

This repository publishes model weights, tokenizers, configurations, tooling, and necessary documentation. It is not equivalent to publishing training corpora, source texts, or third-party works themselves.

### 6.1 可开放内容 / Openable Content

本仓库鼓励开放：

- 模型权重；
- Tokenizer；
- 模型配置；
- 工具代码；
- 推理示例；
- 标注规范；
- 转写规则；
- 清理规则；
- 索引结构；
- 元数据模板；
- 错误类型；
- 质量评估方法；
- 不可还原原文的统计结果和加工成果。

This repository encourages openness for:

- model weights;
- tokenizers;
- model configurations;
- tooling code;
- inference examples;
- annotation guidelines;
- transcription rules;
- cleaning rules;
- index structures;
- metadata templates;
- error categories;
- quality-evaluation methods;
- statistical results and processed outputs that cannot reconstruct original texts.

### 6.2 受控内容 / Controlled Content

本仓库不将以下内容作为默认开源对象：

- 完整原始语料；
- 完整转写文本；
- 扫描件、图片、影印件；
- 连续长段原文；
- 同行评议材料；
- 个人信息；
- 第三方受保护作品表达；
- 其他受保密义务、访问控制或法律限制的内容。

This repository does not treat the following as open-source by default:

- complete raw corpora;
- full transcriptions;
- scans, images, or facsimiles;
- long continuous source-text passages;
- peer-review materials;
- personal information;
- third-party protected expressions;
- other materials subject to confidentiality, access control, or legal restrictions.

---

## 7. 贡献流程 / Contribution Workflow

### 7.1 提交 Issue / Opening an Issue

提交 Issue 前，请先检查是否已有相同或类似问题。

Issue 建议包含：

- 问题类型；
- 受影响版本或文件；
- 复现步骤；
- 期望行为；
- 实际行为；
- 环境信息；
- 日志、截图、哈希值或最小复现示例；
- 是否涉及安全、版权、个人信息或受限制材料。

Before opening an issue, please check whether a similar issue already exists.

An issue should include:

- issue type;
- affected version or files;
- reproduction steps;
- expected behavior;
- actual behavior;
- environment information;
- logs, screenshots, hashes, or minimal reproducible examples;
- whether security, copyright, personal information, or restricted materials are involved.

### 7.2 提交 Pull Request / Opening a Pull Request

Pull Request 建议包含：

- 清晰标题；
- 修改摘要；
- 修改动机；
- 影响范围；
- 测试方式；
- 是否涉及模型权重、Tokenizer、许可证、安全或第三方材料；
- 是否存在向后兼容性影响；
- 是否需要更新文档、模型卡或安全说明。

A pull request should include:

- a clear title;
- summary of changes;
- motivation;
- impact scope;
- test method;
- whether model weights, tokenizers, licensing, security, or third-party materials are involved;
- whether there are backward-compatibility impacts;
- whether documentation, model cards, or security notes need updates.

### 7.3 推荐分支命名 / Suggested Branch Names

- `docs/...` 文档修改 / documentation changes
- `fix/...` 问题修复 / bug fixes
- `tokenizer/...` Tokenizer 相关 / tokenizer-related changes
- `model/...` 模型相关 / model-related changes
- `security/...` 安全相关 / security-related changes
- `release/...` 发布相关 / release-related changes
- `tests/...` 测试相关 / test-related changes
- `ci/...` CI 相关 / CI-related changes

### 7.4 Commit Message 建议 / Commit Message Suggestions

建议使用简洁、可追溯的 commit message：

```text
docs: update citation standards
fix: handle tokenizer padding edge case
model: add config validation for recurrent steps
tests: add tokenizer boundary tests
security: clarify restricted content policy
release: update model card and checksums
```

Suggested commit messages should be concise and traceable:

```text
docs: update citation standards
fix: handle tokenizer padding edge case
model: add config validation for recurrent steps
tests: add tokenizer boundary tests
security: clarify restricted content policy
release: update model card and checksums
```

---

## 8. 测试、格式化与质量要求 / Testing, Formatting, and Quality Requirements

本仓库同时包含 Python 代码、Tokenizer 相关实现、模型相关实现，以及 Rust 编码映射组件。贡献者提交代码、配置、Tokenizer、模型相关文件或文档前，应尽量保证修改可复现、可测试、可审查。

This repository contains Python code, tokenizer-related implementations, model-related implementations, and Rust encoding-mapping components. Before submitting code, configuration, tokenizer files, model-related files, or documentation, contributors should make changes reproducible, testable, and reviewable where possible.

### 8.1 Python 要求 / Python Requirements

本项目要求 Python `>=3.10`。Python 代码应尽量遵循以下要求：

- 使用清晰、稳定、可读的类型标注；
- 避免无必要的全局状态；
- 避免隐藏副作用；
- 保持函数、类和模块职责清晰；
- 对模型配置、Tokenizer 行为、边界条件和异常路径补充测试；
- 不引入未经说明的重型依赖；
- 不提交本地缓存、临时文件、日志文件或环境目录。

This project requires Python `>=3.10`. Python code should follow these requirements where possible:

- use clear, stable, and readable type hints;
- avoid unnecessary global state;
- avoid hidden side effects;
- keep functions, classes, and modules focused;
- add tests for model configuration, tokenizer behavior, boundary cases, and exception paths;
- do not introduce heavy dependencies without explanation;
- do not commit local caches, temporary files, log files, or environment directories.

### 8.2 Rust 要求 / Rust Requirements

涉及 Rust 组件的贡献，应保持格式统一，并通过格式化检查。

Rust-related contributions should keep formatting consistent and pass formatting checks.

建议运行：

```bash
cd "Encoding Mapping"
cargo test --quiet
cargo fmt --check
```

Recommended commands:

```bash
cd "Encoding Mapping"
cargo test --quiet
cargo fmt --check
```

### 8.3 推荐测试命令 / Recommended Test Commands

如仓库环境支持，提交前建议运行：

```bash
python3 -m unittest discover Tokenizer
python3 -m unittest discover Model
./scripts/test_all.sh
```

If supported by the repository environment, contributors are encouraged to run:

```bash
python3 -m unittest discover Tokenizer
python3 -m unittest discover Model
./scripts/test_all.sh
```

### 8.4 模型权重、Tokenizer 与二进制文件 / Model Weights, Tokenizers, and Binary Files

对于模型权重、Tokenizer 文件、配置文件或二进制文件，贡献者应尽量提供：

- 文件来源说明；
- 版本说明；
- 校验和；
- 加载方式；
- 兼容性说明；
- 已知限制；
- 是否涉及第三方模型、第三方权重、第三方代码或第三方许可。

For model weights, tokenizer files, configuration files, or binary files, contributors should provide where possible:

- file source information;
- version notes;
- checksums;
- loading instructions;
- compatibility notes;
- known limitations;
- whether third-party models, weights, code, or licenses are involved.

### 8.5 Pull Request 质量门槛 / Pull Request Quality Bar

Pull Request 应尽量满足以下要求：

- 修改范围清晰；
- 不混合无关修改；
- 不提交原始语料、完整转写文本、扫描件、图片或连续长段原文；
- 不提交个人信息、敏感信息、保密信息或同行评议材料；
- 涉及模型、Tokenizer、配置或权重的修改，应说明兼容性影响；
- 涉及许可证、版权、安全或权利边界的修改，应说明原因；
- 涉及文档的修改，应保持中英文内容尽量一致；
- 涉及代码的修改，应尽量附带测试或验证说明。

A pull request should meet the following quality bar where possible:

- the scope of changes is clear;
- unrelated changes are not mixed together;
- raw corpus materials, full transcriptions, scans, images, or long continuous source-text passages are not submitted;
- personal information, sensitive information, confidential information, or peer-review materials are not submitted;
- changes to models, tokenizers, configurations, or weights explain compatibility impact;
- changes involving licenses, copyright, security, or rights boundaries explain the rationale;
- documentation changes keep Chinese and English content reasonably aligned;
- code changes include tests or validation notes where possible.

---

## 9. 安全问题 / Security Issues

安全问题请按照 `SECURITY.md` 报告，不要在公开 Issue、Pull Request、Discussion、社交平台或公开聊天群中披露尚未修复的安全问题、数据泄露细节、可复现攻击步骤或敏感样例。

安全问题包括但不限于：

- 权重或 Tokenizer 被篡改；
- 模型加载触发非预期网络访问、文件写入或命令执行；
- 文件格式存在可利用的反序列化风险；
- 仓库误包含原始语料、完整转写文本、个人信息或第三方受保护表达；
- 模型输出可稳定还原训练材料中的长段原文；
- 发布包误包含内部文件或受限制数据。

Security issues should be reported according to `SECURITY.md`. Do not disclose unresolved security issues, data-leakage details, reproducible attack steps, or sensitive examples in public issues, pull requests, discussions, social platforms, or public chat groups.

Security issues include but are not limited to:

- tampered model weights or tokenizers;
- model loading causing unexpected network access, file writes, or command execution;
- exploitable deserialization risks in file formats;
- accidental inclusion of raw corpus materials, full transcriptions, personal information, or third-party protected expressions;
- model outputs that can reliably reconstruct long passages from training materials;
- release packages that mistakenly include internal files or restricted data.

---

## 10. 代码风格、文档风格与引用规范 / Code Style, Documentation Style, and Citation Standards

### 10.1 总体风格 / General Style

贡献者应保持仓库风格统一、表达克制、结构清晰、便于审查。

Contributors should keep the repository style consistent, restrained, well-structured, and easy to review.

原则包括：

- 文件命名清晰；
- 目录结构稳定；
- 注释服务于理解代码，不堆砌无关解释；
- 测试覆盖边界条件；
- 文档说明可执行、可复现、可维护；
- 中英文双语内容应尽量对应；
- 涉及文化宗旨时，优先体现保护、传承和弘扬中华民族优秀传统文化；
- 涉及版权、语料和模型边界时，使用审慎、可核查、可维护的表述。

Principles include:

- clear file names;
- stable directory structure;
- comments should help explain the code, not add unrelated exposition;
- tests should cover boundary cases;
- documentation should be actionable, reproducible, and maintainable;
- bilingual Chinese-English content should be reasonably aligned;
- when discussing cultural purpose, prioritize the protection, preservation, and promotion of the outstanding traditional culture of the Chinese nation;
- when discussing copyright, corpus, and model boundaries, use careful, verifiable, and maintainable wording.

### 10.2 Python 风格 / Python Style

Python 贡献应尽量遵循：

- Python `>=3.10` 语法；
- 明确的函数和类职责；
- 可读的类型标注；
- 稳定的导入顺序；
- 对异常路径作出清晰处理；
- 避免无说明的隐式依赖；
- 避免将测试逻辑混入核心实现；
- 对 Tokenizer、模型配置、张量形状、特殊 token、padding、masking 等边界行为补充测试。

Python contributions should follow where possible:

- Python `>=3.10` syntax;
- clear function and class responsibilities;
- readable type hints;
- stable import ordering;
- clear handling of exceptional paths;
- no unexplained implicit dependencies;
- no mixing of test logic into core implementation;
- tests for tokenizer behavior, model configuration, tensor shapes, special tokens, padding, masking, and other boundary behavior.

### 10.3 Rust 风格 / Rust Style

Rust 贡献应尽量遵循：

- 通过 `cargo fmt --check`；
- 通过 `cargo test`；
- 避免不必要的 `unsafe`；
- 对编码映射、边界输入、非法输入和错误处理补充测试；
- 对公共接口和关键数据结构提供简明说明。

Rust contributions should follow where possible:

- pass `cargo fmt --check`;
- pass `cargo test`;
- avoid unnecessary `unsafe`;
- add tests for encoding mappings, boundary inputs, invalid inputs, and error handling;
- provide concise documentation for public interfaces and key data structures.

### 10.4 文档风格 / Documentation Style

文档贡献应尽量遵循：

- 标题层级清晰；
- 术语前后一致；
- 示例简短、必要、可运行；
- 不使用来源不明的长文本样例；
- 不使用未经授权的完整原文、完整转写文本、扫描件或连续长段表达；
- 涉及模型权重、Tokenizer、配置、测试命令时，应给出版本或路径；
- 涉及风险、限制、许可证、权利主张时，应避免绝对化表述；
- 涉及公共文化价值时，应保持庄重、规范、体面。

Documentation contributions should follow where possible:

- clear heading hierarchy;
- consistent terminology;
- short, necessary, and runnable examples;
- no long text examples of unclear source;
- no unauthorized full source texts, full transcriptions, scans, or long continuous passages;
- provide versions or paths when discussing model weights, tokenizers, configurations, or test commands;
- avoid absolute statements when discussing risks, limitations, licenses, or rights claims;
- maintain a respectful, formal, and proper tone when discussing public cultural value.

### 10.5 引用规范 / Citation Standards

为保证学术严谨性、工程可追溯性和权利边界清晰，贡献者在文档、模型卡、Issue、Pull Request 或说明文件中引用外部资料时，应尽量遵循以下规范：

- 引用论文时，提供作者、标题、年份、会议/期刊/预印本平台和链接；
- 引用代码时，提供项目名称、仓库链接、许可证和具体版本或 commit；
- 引用模型时，提供模型名称、发布主体、版本、许可证和模型卡链接；
- 引用数据集时，提供数据集名称、发布主体、许可证、版本和访问链接；
- 引用书籍、文献或历史资料时，尽量提供作者、译者、整理者、出版信息、版本信息和页码范围；
- 引用网页时，提供标题、发布主体、链接和访问日期；
- 引用本仓库内容时，尽量使用文件路径、版本号、commit 或 release tag；
- 不引用来源不明、授权不明、权利状态不清的材料作为正式依据；
- 不将未公开同行评议材料、保密材料或平台内受控材料作为公开引用内容。

To ensure academic rigor, engineering traceability, and clear rights boundaries, contributors should follow these standards when citing external materials in documentation, model cards, issues, pull requests, or explanatory files:

- for papers, provide authors, title, year, venue/preprint platform, and link;
- for code, provide project name, repository link, license, and specific version or commit;
- for models, provide model name, publisher, version, license, and model card link;
- for datasets, provide dataset name, publisher, license, version, and access link;
- for books, texts, or historical materials, provide author, translator, compiler, publication information, version information, and page range where possible;
- for webpages, provide title, publisher, link, and access date;
- for repository content, use file paths, versions, commits, or release tags where possible;
- do not cite materials of unclear source, unclear authorization, or unclear rights status as formal authority;
- do not publicly cite unpublished peer-review materials, confidential materials, or platform-controlled materials.

### 10.6 推荐引用格式 / Recommended Citation Formats

论文 / Paper:

```text
Author(s). "Title." Venue or Publisher, Year. URL or DOI.
```

代码 / Code:

```text
Project Name, version or commit, license, repository URL.
```

模型 / Model:

```text
Model Name, publisher, version, license, model card URL.
```

数据集 / Dataset:

```text
Dataset Name, publisher, version, license, access URL, access date.
```

历史文献或书籍 / Historical Text or Book:

```text
Author / Translator / Compiler. Title. Publisher or Collection, Edition, Year, page range.
```

本仓库 / This Repository:

```text
Montlok/MoRoN-1.2, file path, commit or release tag, Apache License 2.0.
```

### 10.7 示例文本规范 / Example Text Rules

贡献者如需添加示例文本，应优先使用：

- 自行创作的短文本；
- 公共领域材料；
- 已明确授权材料；
- 合成样例；
- 抽象化、截断化、不可替代原作品的极短示例；
- 不可还原原文的标签、元数据、统计结果或结构化字段。

Contributors who need to add example text should prioritize:

- self-created short text;
- public-domain materials;
- expressly authorized materials;
- synthetic examples;
- abstracted, truncated, non-substitutive short examples;
- labels, metadata, statistical results, or structured fields that cannot reconstruct original texts.

贡献者不得将完整原文、完整转写文本、扫描件、图片、连续长段表达或足以替代原作品实质性内容的材料作为示例提交。

Contributors must not submit full source texts, full transcriptions, scans, images, long continuous passages, or materials that may substitute for the substantive content of original works as examples.

---

## 11. 贡献者行为准则 / Contributor Conduct

贡献者应尊重：

- 母语者、研究者、开发者、标注者和文化传承参与者；
- 不同技术路线和学术观点；
- 第三方依法享有的合法权益；
- 项目的文化传承宗旨；
- 仓库的安全、开源和公共协作属性；
- 我国政府以及广大社会人士对中华民族优秀传统文化保护、民族语言文字资料整理、传统蒙古语资料传承、学术研究和科技创新事业的鼎力支持。

不得通过本仓库进行：

- 人身攻击；
- 恶意骚扰；
- 泄露个人信息；
- 泄露保密材料；
- 公开传播受限制原始语料；
- 提交恶意代码或误导性文件；
- 破坏项目协作秩序的行为。

Contributors should respect:

- native speakers, researchers, developers, annotators, and cultural contributors;
- different technical approaches and academic views;
- lawful third party rights and interests;
- the project’s cultural preservation purpose;
- the repository’s security, open source, and public collaboration nature;
- We especially express our gratitude to the government of the People's Republic of China (PRC) and all sectors of society for their strong support of the protection of the outstanding traditional culture of the Chinese nation, the organization of ethnic language materials, the transmission of traditional Mongolian resources, academic research, and technological innovation.
The following are not allowed in this repository:

- personal attacks;
- harassment;
- disclosure of personal information;
- disclosure of confidential materials;
- public distribution of restricted raw corpus materials;
- submission of malicious code or misleading files;
- conduct that disrupts project collaboration.

---

## 12. 权利主张、移除请求与争议内容 / Rights Claims, Takedown Requests, and Disputed Content

如作者、译者、整理者、出版机构、馆藏机构、数据库权利人、资料提供方或其他主体认为本仓库内容涉及其合法权益，可通过 `SECURITY.md` 中的联系渠道提交说明。

请尽量提供：

- 权利主体身份说明；
- 涉及文件或链接；
- 权利主张依据；
- 希望采取的处理方式；
- 联系方式。

维护者将根据具体情况进行核查，并可采取隐藏、移除、替换、限制访问、补充来源说明、调整示例或其他合理措施。

对相关内容采取临时处理措施，不当然表示项目方承认相关权利主张成立。

If an author, translator, compiler, publisher, archival institution, database right holder, material provider, or other party believes that repository content concerns their lawful rights or interests, they may submit a notice through the contact channels in `SECURITY.md`.

Please provide, where possible:

- identification of the rights holder;
- relevant files or links;
- basis for the rights claim;
- requested action;
- contact information.

Maintainers will review the matter based on the circumstances and may hide, remove, replace, restrict access, supplement source information, adjust examples, or take other reasonable measures.

Temporary measures do not necessarily constitute an admission of the validity of any rights claim.

---

## 13. 致谢 / Acknowledgement

我们感谢所有为本项目作出贡献的人。

本项目尤其感谢我国政府以及广大社会人士对中华民族优秀传统文化保护、民族语言文字资料整理、传统蒙古语资料传承、学术研究和科技创新事业的鼎力支持。

参与模型、Tokenizer、工具链、文档、测试、评估、语言知识、文化语境说明、错误报告、权利线索和学术建议的贡献者，均是本项目长期发展的重要支持力量。

本项目尤其感谢为传统蒙古语资料数字化保护、系统整理、传承发展和学术传播作出贡献的母语者、教师、研究者、标注者、开发者和文化传承参与者。

We thank everyone who contributes to this project.

We especially express our gratitude to the government of PRC and to members of society for their strong support of the protection of the outstanding traditional culture of the Chinese nation, the organization of ethnic-language materials, the transmission of traditional Mongolian resources, academic research, and technological innovation.

Contributors to models, tokenizers, tooling, documentation, tests, evaluation, linguistic knowledge, cultural-context notes, bug reports, rights leads, and academic suggestions are important supporters of the project’s long-term development.

We especially thank native speakers, teachers, researchers, annotators, developers, and cultural contributors who support the digital preservation, systematic organization, transmission, and academic dissemination of traditional Mongolian materials.

---

## 14. Pull Request Checklist

提交 Pull Request 前，请确认：

- [ ] 我已阅读 `SECURITY.md` 和 `CONTRIBUTING.md`；
- [ ] 本贡献遵守 Apache License 2.0；
- [ ] 本贡献不包含完整原始语料、完整转写文本、扫描件、图片或连续长段原文；
- [ ] 本贡献不包含个人信息、敏感信息、保密信息或同行评议材料；
- [ ] 本贡献不包含来源不明、授权不明或权利状态不清的第三方受保护表达；
- [ ] 如修改模型、Tokenizer、配置或权重，我已说明兼容性影响；
- [ ] 如修改代码，我已尽量运行相关测试；
- [ ] 如修改 Rust 组件，我已尽量运行 `cargo test` 和 `cargo fmt --check`；
- [ ] 如引用外部资料，我已按引用规范注明来源；
- [ ] 如涉及安全、版权、许可证或权利主张，我已在 PR 中明确说明；
- [ ] 如修改文档，我已尽量保持中英文内容对应；
- [ ] 如新增示例文本，我已确认其来源、授权或公共领域状态。

Before submitting a pull request, please confirm:

- [ ] I have read `SECURITY.md` and `CONTRIBUTING.md`;
- [ ] this contribution complies with the Apache License 2.0;
- [ ] this contribution does not contain complete raw corpora, full transcriptions, scans, images, or long continuous source-text passages;
- [ ] this contribution does not contain personal information, sensitive information, confidential information, or peer-review materials;
- [ ] this contribution does not contain third-party protected expressions with unclear source, authorization, or rights status;
- [ ] if models, tokenizers, configurations, or weights are changed, I have explained compatibility impact;
- [ ] if code is changed, I have run relevant tests where possible;
- [ ] if Rust components are changed, I have run `cargo test` and `cargo fmt --check` where possible;
- [ ] if external materials are cited, I have provided citations according to the citation standards;
- [ ] if security, copyright, licensing, or rights claims are involved, I have explained them in the PR;
- [ ] if documentation is changed, I have kept Chinese and English content reasonably aligned where possible;
- [ ] if example text is added, I have confirmed its source, authorization, or public-domain status.

---

## 15. 简明原则 / Short Principle

> 仓库开源，原文受控；工具自由，贡献开放；加工成果最大开放，第三方表达依法处理；文化传承置于首位。

> The repository is open; source texts are controlled. Tools are free; contributions are open. Processed outputs are maximally open; third-party expressions are handled according to law. Cultural transmission comes first.
