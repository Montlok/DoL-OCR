# DoL 蒙文 RDT 训练线 — 交接指引(2026-07-07)

工作区 `~/projects/LLM`(非 git);代码仓 `~/projects/LLM/DoL-OCR`(origin=Montlok/DoL-OCR,**必须保持 private**;另有 Montlok/DoL-1.2 同样 private)。语料与权重绝不提交 git。

## 现场

- 训练机 box:`ssh spark1`(LAN 192.168.1.108 优先;隧道备选 spark/spark2,会间歇断)。
- 正在长跑:`~/dolocr/runs/mn_pretrain_v1` — 0.972B RDT two_stage,纯蒙文语料(2.14B token/epoch,~1.9K tok/s,一个 epoch 12-13 天),WSD 调度随停随续。step 4000 实测 eval_loss 1.2315 ≈ train(无背诵)。模型用途:**只做传统蒙文 OCR**。
- 一切操作命令、故障恢复、对齐轮(swap_to_align.sh)、结果判读标准:看 **`DoL-OCR/RUNBOOK.md`**(box 上同样有一份)。
- 完整状态与决策记录:`~/.claude/projects/-Users-gabiri-projects-LLM/memory/mn-pretrain-status.md` 及同目录其他 memory 文件(可直接读)。

## 硬禁忌(全部实测过)

1. 训练运行时,box 上不要启动任何旁路 GPU 程序或 page-cache evict——统一内存下必 OOM。
2. 不要 kill 训练进程;用 `scripts.rdt_monitor control save/stop`(优雅存档退出),恢复用 `RESUME_ONLY=1 bash scripts/swap_to_align.sh`。
3. `pkill -f`/`pgrep -f` 的模式会匹配到自身所在命令行——远端执行时用 `[t]rain_rdt` 这类字符类写法。
4. Mac 上加载不了官方 mamba 训的权重(in_proj 8672 vs CPU fallback 8192),验证在 box 的 swap 窗口做。
5. 真实标注数据很珍贵:只用于 RL(约 70%)和永不训练的 golden set 评测(约 30%),不做 SFT 燃料。

## 标注工具

真实扫描行的转写标注用独立仓库 Montlok/mn-annotator(private,本地 ~/projects/LLM/mn-annotator):`python3 server.py --repo ~/projects/LLM/DoL-OCR --bundle <bundle>` 启动全可视化标注台(内嵌 nominal-Unicode 归一化校验);产出 TSV 经 validate.py 终检后,~70% 进 GRPO reward、~30% 锁 golden set。

## 用户工作纪律

全中文回复;不用 emoji;不用形容词与戏剧化词汇(禁:铁律/冷数字/尸检/点火/收官等);只报实测数字+差距,不吹;每次有意义改动先过一个零上下文 harsh-critic 审查(APPROVE/REJECT 循环)再推进;先小批验证(含 index 0)再放量,绝不用精选样本声称"能用"。
