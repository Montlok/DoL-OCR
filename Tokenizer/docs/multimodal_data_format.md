# 多模态预训练数据格式 (JSONL Schema)

本仓库的预训练 dataloader 通过统一 JSONL 行格式同时驱动**纯文本**与**视觉-语言**训练。本文档定义字段的语义、必需性，以及如何从公开 OCR / VLM 数据集映射进来。

## 行级 Schema

```jsonc
{
  "text":          "<image> 第一段说明文字 ...",        // 必填: 含 <image> 占位符的提示
  "images":        ["/abs/path/page0001.png"],          // 可选: 一行的图片列表
  "image_sizes":   [[1024, 768]],                       // 可选: 与 images 对齐, [[H, W], ...]
  "videos":        [],                                  // 可选: 预留, 暂未消费
  "video_sizes":   [],                                  // 可选
  "ocr_labels":    [[12, 47, 33, 99]],                  // 可选: 每张图的 token id 序列
  "reading_order": [[0, 1, 2, 3]]                       // 可选: 每张图的 patch 阅读序
}
```

### 必填字段
- `input_ids` / `attention_mask` / `labels` 由 `PretrainingDataBuilder` 在 `build_pretraining_data` 阶段从 `text` 生成；下游不需要手填。
- `text`：自然语言提示。`<image>` 占位符会被 `MultimodalProcessor` 展开成若干个 `<image_patch>` 槽位；数量由 `image_patch_count(width, height, patch_size=14, merge_size=2)` 决定，并且必须等于训练时的 `OMVTConfig.compress_to`。

### 多模态可选字段
| 字段 | 形状 | 说明 |
| --- | --- | --- |
| `images` | `list[str \| bytes \| dict]` | PIL 可读的图片来源；`PILImageProcessor` 在 collator 内即时加载 |
| `image_sizes` | `list[[int, int]]` | 原图 `[H, W]`，仅作为元数据；OMVT 自身会 resize 到 `OMVTConfig.image_size` |
| `ocr_labels` | `list[list[int]]` | 每张图对应的 token id 序列；不在的图保留 `null`；缺失时 OCR 头的 loss 自动跳过 |
| `reading_order` | `list[list[int]]` | 每张图的 patch 阅读序 (长度 ≤ `compress_to`)；不足部分自动 `arange` 补齐 |

### 同批一致性约束
`PretrainingCollator._build_pixel_batch` **当前实现严格要求每行恰好 1 张图**——这是 `VisionInjector` 的约束：它把 OMVT 的 `[B, compress_to, D]` 特征按行一一替换 `<image_patch>` 槽位。混批 0/1 图或多图都会立刻抛 `ValueError`，错误信息会指出如何分桶。

未来扩展到 N>1 / row 需要同时改 collator（reshape 到 `[B, N*compress_to, D]`）+ VisionInjector（按行段切片注入）+ 文本侧（`<image>` 展开到 `N*compress_to` 个槽位）。当前不在路线图上；多图数据请通过 bucketed dataloader 把每条记录拆成多个单图行。

## 三个训练入口的开关

| 脚本 | 启用多模态的方式 |
| --- | --- |
| `scripts/train_rdt.py` | `--multimodal --image-size 56 --n-image-tokens 4 --tokenizer-bundle <bundle> --data <jsonl> --data-receipt <receipt>` |
| `scripts/train_vlm_align.py` | `--data <jsonl>`（同时锁住视觉塔可加 `--frozen-vision`） |
| `scripts/train_omvt_ssl.py` | `--data <jsonl>`（orientation 自监督自动生效；OCR/layout 头按字段降级） |

这里的 RDT receipt 必须来自生成这些预分词 JSONL 的
`Tokenizer.tools.build_pretraining_data --receipt ...`，它同时绑定分片
字节、tokenizer bundle 和 tokenizer 算法。若另设 `--eval-data`，验证集需
独立 receipt，并通过 `--eval-data-receipt` 传入。
所有 production/non-smoke receipt-backed train/eval/mix JSONL 行还必须
持久化 `word_pos` 与 `morph_depth`；model-side fallback 仅限 legacy/smoke，
不得作为 production 路径。

## 外部数据 → 本 Schema 的迁移骨架

外部数据必须在 Git 工作树之外完成来源、授权、split 和删除机制审核。本仓库不提供下载器或爬虫。

若 OCR 标注里含分词后的 token 序列：
```python
{
  "text": "<image>",
  "images": [page_path],
  "ocr_labels": [token_ids],          # 单图简写；builder 会规范成 [[...]]
  "reading_order": [layout_order],    # 单图简写；builder 会规范成 [[...]]
}
```
预渲染的 WebDataset 图文配对由正式脚本转换：

```bash
python -m scripts.build_ocr_data_from_pairs \
  --shards-dir "$PAIR_SHARDS_DIR" \
  --shard-indices "$SHARD_INDICES" \
  --out "$OCR_DATA_ROOT" \
  --tokenizer-bundle "$TOKENIZER_BUNDLE"
```

需要显式合成预训练行时使用 `scripts.build_ocr_data`。两类输出都必须留在仓库外，并携带对应的不可变数据 contract；不能把生成结果作为测试 fixture 提交。

## 与 OMVT 的耦合
- `images` 在 collator 里被 `PILImageProcessor` 加载为 `[B, 3, H, W]` 张量，随后送入 `Model/omvt/patcher.collate_omvt_batch` 切成 4 个尺度（vertical/horizontal/square/layout）。
- `pixel_values` 是一个 `dict[str, Tensor]`；`train_one_step` 已经会把它递归搬到模型设备并透传给 `RDTForCausalLM(pixel_values=...)`。
- 占位符 `<image_patch>` 已在 `DEFAULT_LABEL_IGNORE_TOKENS` 里，模型不会在这些位置计算 LM 损失。

## 依赖
- 启用多模态需要 `Pillow>=9`；可通过 `pip install -e .[image]` 安装。
- 不强依赖 `torchvision`：`PILImageProcessor` 用纯 PIL + `torch.frombuffer` 实现，无 numpy 依赖。
