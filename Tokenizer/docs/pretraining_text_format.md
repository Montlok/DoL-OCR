# 预训练文本数据格式（JSONL 规范）

本文档定义**正式预训练所需的文本 JSON 格式**。数据流分两个阶段：

```
原始语料 ──(build_pretraining_data)──▶ 编码后分片 ──(train_rdt --data)──▶ 训练
 .txt / .jsonl                          *.jsonl（含 input_ids…）
```

- **阶段一（原始输入）**：你手写或从公开数据集映射出来的语料，喂给
  `Tokenizer/tools/build_pretraining_data.py`。
- **阶段二（编码后分片）**：builder 产出的、`scripts/train_rdt.py --data`
  直接消费的行格式。**正式训练读取的是阶段二的分片**，阶段一的字段不会
  直接进模型。builder 同时生成不可变 receipt，将这些分片的字节绑定到
  tokenizer bundle、共享 tokenizer 算法，以及明确命名的行生产器源码指纹；
  正式训练只接受已注册且指纹完全一致的生产器，并必须一并传入 receipt。

多模态字段的语义见
[`multimodal_data_format.md`](./multimodal_data_format.md)；本文聚焦**各种文本格式**。

---

## 阶段一：原始输入（喂给 `build_pretraining_data`）

`--input` 接受 `.txt` 或 `.jsonl`（按扩展名区分）。

### 1.1 纯文本 `.txt`

每行一条文档，空行自动跳过；每行等价于 `{"type": "text", "text": "<该行>"}`。

```text
ᠮᠣᠩᠭᠣᠯ ᠬᠡᠯᠡ ᠪᠢᠴᠢᠭ ᠦᠨ ᠰᠤᠷᠭᠠᠯ ᠤᠨ ᠡᠬᠢ ᠰᠤᠷᠪᠤᠯᠵᠢ ...
第二条文档（与上一行各自独立成样本）...
```

### 1.2 JSONL（每行一个 JSON 对象）

| 字段 | 类型 | 必需性 | 说明 |
|------|------|--------|------|
| `text` | `str` | **必填**（缺失按 `""` 处理，会被判空跳过） | 文本内容。多模态时含 `<image>` / `<video>` 占位符 |
| `type` | `str` | 可选，默认 `"text"` | `text` / `image_text` / `ocr` / `video_text`；取后三者或带媒体字段时走多模态编码 |
| `source` | `str` | 可选 | 数据来源标识，透传进 `metadata` |
| `id` | `str` | 可选 | 样本 ID，透传进 `metadata` |
| `images` | `list[str\|bytes\|dict]` | 可选（多模态） | 见多模态文档 |
| `image_sizes` | `list[[H, W]]` | 可选 | 与 `images` 对齐 |
| `videos` / `video_sizes` | `list[...]` | 可选 | 视频对应字段 |
| `ocr_labels` | `list[list[int]]` | 可选 | 每张图的 token id 序列（OCR 头监督） |
| `reading_order` | `list[list[int]]` | 可选 | 每张图的阅读顺序（layout 头监督） |

> 透传进 `metadata` 的字段白名单：`type, ocr, images, image_sizes, videos,
> video_sizes, source, id`；其余自定义字段会被忽略。

**各种文本格式示例：**

```jsonc
// (a) 最简纯文本
{"text": "ᠮᠣᠩᠭᠣᠯ ᠬᠡᠯᠡ ᠶᠢᠨ ᠵᠢᠱᠢᠶ᠎ᠡ ᠥᠭᠦᠯᠡᠪᠦᠷᠢ ..."}

// (b) 带来源 / ID 元数据的纯文本
{"text": "...", "type": "text", "source": "cleaned_web", "id": "doc-00012"}

// (c) 图文（caption）
{"text": "<image> 图中是一段竖排蒙古文标题。", "images": ["/data/page0001.png"], "image_sizes": [[768, 1024]]}

// (d) OCR（仅占位符 + token 监督）
{"type": "ocr", "text": "<image>", "images": ["/data/scan.png"], "ocr_labels": [[12, 47, 33, 99]]}

// (e) 视频文本
{"type": "video_text", "text": "<video> 视频解说文本 ...", "videos": ["/data/clip.mp4"]}
```

### 1.3 builder 的编码行为（你不需要手填 `input_ids`）

`PretrainingDataBuilder`（`Tokenizer/pretraining/builder.py`）：

- 默认 `add_bos=True, add_eos=True`，`max_length=4096`（CLI `--max-length` 覆盖，
  默认 2048）。
- **`labels` 自动生成**：先复制 `input_ids`，再把结构/特殊 token 置为 `-100`
  （`<pad> <unk> <bos> <img> <image> <image_start> <image_patch> <image_end>
  <video*> <audio*>` 等，见 `DEFAULT_LABEL_IGNORE_TOKENS`），这些位置不计入
  语言建模 loss。
- 超过 `max_length` 时截断（多模态会同时裁剪对应 `<image_patch>` 跨度）。
- `--pack` 可把多条纯文本样本打包到一条（仅纯文本；以 `<eos>` 分隔）。

---

## 阶段二：编码后分片（`train_rdt --data` 直接消费）

每行是 `EncodedSample` 的 JSON（`encoded_sample_to_dict` = `dataclasses.asdict`）。

| 字段 | 类型 | 必需性 | 说明 |
|------|------|--------|------|
| `input_ids` | `list[int]` | **必填** | token id 序列 |
| `attention_mask` | `list[int]` | **必填** | 与 `input_ids` 等长，1=有效 0=padding |
| `labels` | `list[int]` | **必填** | 与 `input_ids` 等长；`-100` 表示该位不计 loss |
| `word_pos` | `list[int]` | **正式 receipt-backed 必填**；legacy/smoke 可选 | 词位置索引；legacy/smoke 缺失时可由 `token_offsets` 推导，再缺失则 `range(n)` |
| `morph_depth` | `list[int]` | **正式 receipt-backed 必填**；legacy/smoke 可选 | 形态深度；legacy/smoke 缺失时同上推导，再缺失则全 `0` |
| `token_offsets` | `list[[start, end]]` | 可选 | 字符跨度；仅供 legacy/smoke 在缺 `word_pos/morph_depth` 时推导 |
| `modality_spans` | `dict` | 可选 | `{"image_token_spans": [...], "video_token_spans": [...]}` |
| `metadata` | `dict` | 可选 | 透传元数据 |
| `images` / `image_sizes` | `list` | 可选（多模态） | 原样透传给 collator |
| `videos` / `video_sizes` | `list` | 可选 | 同上 |
| `ocr_labels` / `reading_order` | `list[list[int]]` | 可选 | SSL 监督标签 |

**`train_rdt` 侧的校验契约（`Model/training/data.py::_normalize_row`）：**

- `input_ids` / `attention_mask` / `labels` **必须存在且三者等长**，否则报错。
- 所有 production/non-smoke receipt-backed `--data` / `--eval-data` /
  `--mix-data` JSONL 必须持久化等长的 `word_pos` 与 `morph_depth`。
- 仅 legacy/smoke 行允许缺失这两个字段：优先用 `token_offsets` 推导；仍
  缺失则填 `range(n)` 与全 `0`。该 model-side fallback 不得用于正式训练。
- 行长 > `--seq-len` 时：
  - 纯文本行 → 截断；
  - **多模态行（含 `images` 或 `videos`）→ 直接报错**（截断会让
    `<image_patch>` 与图像负载错位）。因此务必让
    `TrainingConfig.seq_len ≥ builder 的 max_length`。

---

## 同批一致性约束

- 一个 batch 内 **要么全部带图、要么全部不带图**（保持张量形状统一）。
- 当前 pixel collator **要求每行恰好 1 张图**；多图/混合需用分桶 dataloader 拆成单图。
- `input_ids` / `attention_mask` / `labels` / `word_pos` / `morph_depth` 必须等长
  （collator 会按 batch 内最大长度右侧 padding：`input_ids→pad_id`、
  `attention_mask→0`、`labels→-100`、`word_pos/morph_depth→0`）。

---

## 端到端最小示例

```bash
# 1) 原始纯文本（.txt 每行一条，或 .jsonl 每行 {"text": ...}）
#    → 编码成分片
PYTHONPATH=. python3 -m Tokenizer.tools.build_pretraining_data \
    --tokenizer-bundle artifacts/bundle/ \
    --input  data/corpus.jsonl \
    --output data/text_shards/shard_00.jsonl \
    --receipt data/text_shards/train.receipt.json \
    --max-length 2048

# 2) 直接用编码后分片开训
PYTHONPATH=. python3 scripts/train_rdt.py --config small \
    --tokenizer-bundle artifacts/bundle/ \
    --data "data/text_shards/*.jsonl" \
    --data-receipt data/text_shards/train.receipt.json \
    --output runs/exp1
```

若配置 `--eval-data`，必须用同一 builder 为验证分片单独生成 receipt，并
同时传 `--eval-data-receipt`；不能复用训练 receipt。

编码后分片单行（纯文本，已可直接喂给 `train_rdt`）：

```json
{"input_ids": [2, 274, 256, 17, 3], "attention_mask": [1, 1, 1, 1, 1], "labels": [-100, 274, 256, 17, 3], "word_pos": [0, 1, 2, 3, 4], "morph_depth": [0, 0, 0, 0, 0], "modality_spans": {"image_token_spans": [], "video_token_spans": []}, "metadata": {"type": "text"}}
```
