# DoL-OCR 运行手册 / Operations Runbook

蒙文 RDT 预训练 + OCR 对齐的日常操作、故障恢复与结果判读。所有命令在训练机(box)上执行,除非注明。

Day-to-day operations for the Mongolian RDT pretraining + OCR alignment line. All commands run on the training box unless noted.

## 当前状态(2026-07-07)/ Current state

- **预训练**:`~/dolocr/runs/mn_pretrain_v1`,0.972B `two_stage_pretrain`,纯蒙文 v3 语料(2.14B token/epoch,15 shards + 1 held-out),吞吐 ~1.9K tok/s,一个 epoch ≈ 12-13 天,WSD 调度(stable 段任意 checkpoint 可用),每 2000 步存档、保留最近 4 个。
- **质量基线**:step 4000 时 eval_loss 1.2315 ≈ train loss(无背诵)。
- **数据**:packed shards `~/dolocr/pretrain_data/mn_part_*.jsonl`(训练)+ `eval/mn_part_15.jsonl`(held-out,勿动);OCR 对齐数据 `~/dolocr/data_v1/`(勿动);tokenizer `~/dolocr/bundle_v3b`。
- **环境**:python = `~/jupyterlab/.venv/bin/python3`(torch cu130 + 官方 mamba);仓库 `~/DoL-OCR`;跑任何脚本前 `cd ~/DoL-OCR` 且 `PYTHONPATH=.`。

## 日常监控 / Monitoring

```bash
bash ~/panel.sh                      # 实时面板:GPU/内存/磁盘/最近 step/epoch 进度(5s 刷新)
grep "step="      ~/dolocr/mn_pretrain_v1.log | tail -5   # loss/吞吐(每 ~10 分钟一行)
grep "eval_loss"  ~/dolocr/mn_pretrain_v1.log | tail -3   # 验证 loss(每 4000 步)
cd ~/DoL-OCR && PYTHONPATH=. ~/jupyterlab/.venv/bin/python3 -m scripts.rdt_monitor tui --run ~/dolocr/runs/mn_pretrain_v1
```

健康判据:GPU ~96%;step 每 ~30 秒 +1;grad_norm < 1;train 与 eval loss 差距 < 0.1。
eval_loss 明显低于 train(差 >0.2)或 grad_norm 持续 >5:停下来查,别硬跑。

## 停止与恢复 / Stop & resume

永远用控制面优雅停,不要 kill:

```bash
cd ~/DoL-OCR
PYTHONPATH=. ~/jupyterlab/.venv/bin/python3 -m scripts.rdt_monitor control save --run ~/dolocr/runs/mn_pretrain_v1
PYTHONPATH=. ~/jupyterlab/.venv/bin/python3 -m scripts.rdt_monitor control stop --run ~/dolocr/runs/mn_pretrain_v1
# trainer 在一步内(约 30s)存档退出
```

恢复(训练死机/重启后同样适用;数据流按步数精确重放):

```bash
RESUME_ONLY=1 bash scripts/swap_to_align.sh
```

## 对齐轮(借 GPU)/ Alignment round

一条命令:优雅停预训练 → 清缓存 → val 泄漏检查 → 生成验证 → 冻结对齐 6000 步(约 7h)→ 哨兵评测 → 自动恢复预训练。总计约 8 小时,预训练无损。

```bash
cd ~/DoL-OCR
nohup bash scripts/swap_to_align.sh > ~/dolocr/swap.log 2>&1 &   # 全链(必须 nohup:ssh 断开不孤儿化)
tail -f ~/dolocr/swap.log                                        # 看进度
# FROZEN_STEPS=3000 / SKIP_GEN=1 可加在 nohup env 前
```

任何一步失败,链的退出钩子会自动恢复预训练(幂等,绝不双开);恢复后验证进程存活并等待推进证据(fast-forward 进度或新 step 行),30 分钟内未确认会提示人工看 panel.sh,进程死亡则以非零退出。

结果判读(`~/dolocr/runs/align_frozen_v2/sentinel.log` 最后一行):

- `contribution`(空白图 CER − 真图 CER,单位为 CER 点):**> +20 = 视觉通路在工作**,底座可用,值得跑 3b 解冻或加深底座重跑;+5~+20 = 弱信号,先加深底座(多训预训练)再重跑对齐;**≈0 或负 = 塌了**(v1 崩溃签名:-1.9),检查 SSL 塔与数据,不要解冻。
- `real_cer`:字素错误率,1.0=全错。第一轮预期很高(底座浅),看趋势不看绝对值;对照:旧 CRNN 合成 test 1.10%。
- 生成验证段(日志里 `PREFIX:`/`MODEL :`):模型续写应当是连贯蒙文;乱码/重复循环 = 底座有问题。

对齐用更深底座重跑:预训练多训几天后再执行同一条命令即可(幂等;先 `rm -rf ~/dolocr/runs/align_frozen_v2` 清上一轮)。

## 故障处理 / Failures

- **训练进程消失**:`tail -50 ~/dolocr/mn_pretrain_v1.log` 找 OOM/报错 → `RESUME_ONLY=1 bash scripts/swap_to_align.sh` 续跑。
- **磁盘满**(`df -h /`):可删 `~/dolocr/runs/` 下旧 run(align_unfreeze_v1 34G、rescue_refreeze 8G、verify_tower_restore 4G、ctc_* 5G——都是 v1 失败线产物);绝不删 `pretrain_data/`、`data_v1/`、`bundle_v3b`、`ckpt_keep/`。
- **page cache 挤 CUDA(启动新 GPU 进程 OOM)**:训练运行期间不要跑任何旁路 GPU 程序(已实测必炸);需要 GPU 就走 swap 链。
- **训练中不要跑 evict**:evict 的大分配会 OOM 训练进程(历史事故)。
- Mac 上不能加载官方 mamba 权重(双后端 in_proj 形状不同),验证一律在 box 的 swap 窗口做。

## 后续路线 / Roadmap

1. **预训练**:跑到 2-3 epoch(数据受限上限 3-4 epoch);要快 = 8×A100 云上 2-3 天(shards 直接搬,DDP 现成:`--dist ddp` + torchrun;先跑 100 步双卡校验)。
2. **对齐**:底座够深后 frozen → 小 lr(6e-5)解冻(`run_dol_ocr_phase3.sh unfreeze`,哨兵挂 STOP_BELOW=5 自动停塌掉的 run)。
3. **真实数据**(用户定调:真实标注很珍贵,只用于 RL 与评测,不做 SFT 燃料):~30% 锁死做 golden set(分域:印刷/报纸/档案/拍照),~70% 做 GRPO reward(全参,reward=−grapheme_CER);domain gap 靠合成拟真增强 + 无标注 SSL + 自动伪标签,不消耗人工标注。
4. **数据引擎**:字体扩到 20-40 种、退化/版式增强、GLM-OCR 蒸馏混排非蒙文部分;蒙文语料扩充只能回 RAW 挖或新采集(cleaned 蒙文全量就是 12G/2.14B token)。
