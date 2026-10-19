# SeaDronesSee 补充实验队列

这是放在 `D:\工贸\sivp\work` 下的执行工具包。它提供实验队列调度、已有 run 产物一致性审计和服务器执行顺序；不包含模型权重，也不会自动改服务器环境或启动云端服务。

## 对照结论

实验方案 PDF 是设计与验收要求，不代表其中列出的 A0–B6 都还没做。SeaDronesSee 执行报告把新版 1,547 帧主链路、主配置和多个消融记为已完成，但报告同时注明以下事项仍要处理：

1. **必须新增推理：新版链路的 resize＋JPEG 对照。** 首先匹配预测引导低载荷工作点约 54 KB；其次匹配区域模式约 360 KB。约 314 KB 是用户当前清单中的第三工作点，可排在其后。
2. **先做选择器代码审计，再决定是否重跑。** 检查 support 与 support＋exploration 候选集、legacy/spatial uniform 的算法、同帧同 ROI 保留数、去重与重叠覆盖顺序。只有实现不公平或和报告不一致时才重跑 A6/A7/A8。
3. **包大小以每帧 `packet.bin` 实际长度为准。** 报告称此口径已在新版运行采用；需由逐帧表、包索引或可重生成的包独立核验，补齐组件大小与完整包大小的对应关系。
4. **补现有结果统计和溯源：** AP@[.50:.95]、类别 AP、边缘 only/0 KB、模型与类别映射、权重和配置哈希、run_manifest、逐帧大小、预测和失败记录。原 200 帧清单恢复不了时不得虚构 V_rest；只把视频级 dev/test 作为替代敏感性分析。
5. **条件项：** 随机大网格比较只在论文要主张“探索优于随机”时做；视频簇 bootstrap 仅在需要强调视频相关性稳健性时做。

稿件目录里的 v28 状态文件较早，仍把全量 packet 路径和 JPEG 对照列为待办；后续 SeaDronesSee 报告更新了 packet 主链路状态，但并未关闭 resize 对照。以原始 run 文件完成验收后再更新稿件状态。

### 当前 GitHub 源码与执行报告存在可见差异

我检视的公开仓库最新提交为 `f503931`。在 [main 的 run_a.py](https://github.com/AntuoYou/Prediction-Conditioned-Evidence/blob/main/protocol_v2/run_a.py) 与 [npu 的 run_a.py](https://github.com/AntuoYou/Prediction-Conditioned-Evidence/blob/npu/protocol_v2/run_a.py) 中：A3 当前是 ROI Q80；A5 当前 `explore: 4`；A6/A7/A8 名称写 25%，实际值为 `ratio: 1.00`；packed 分支只把 `sup` 传给 `select_patches`，没有把 exploration ROI 并进候选集。SeaDronesSee 报告记录的运行配置/候选修复与这些文件不一致。**这不能直接证明报告里的运行错了**，但表示 run_id 必须关联到服务器当时的未提交 diff 或配置快照，才可以验收和作为 54 KB 对照目标。

执行顺序因此是：冻结服务器 commit 与 diff → 对照 run_manifest/实际 config 确认 A3/A5/A6/A7/A8 运行语义和真实字节 → 审计候选及重复/边界 → 再做 resize 字节校准与推理。若找不到运行配置证据，先在新分支把配置修正并重跑受影响比较；不要直接用公开当前 A7 数字计算匹配预算。

## 服务器执行次序

### 0. 获取并核验服务器代码

在服务器现有项目目录先保存当前 commit 与 `git diff`，新建实验分支，然后从本目录复制工具包。不要把权重、数据或运行输出提交到 Git。

```bash
git status --short
git rev-parse HEAD
git diff > /tmp/pre_supplement_worktree.diff
git switch -c codex/seadronesse-supplement-queue
```

在本地仓库 checkout 中可用本包的 `install_into_repo.ps1` 复制工具文件并查看 Git 变更：

```powershell
& 'D:\工贸\sivp\work\seadronesse_supplement_queue\install_into_repo.ps1' `
  -RepoPath 'D:\path\to\Prediction-Conditioned-Evidence'
```

确认 `git status --short` 只包含本工具包后，再自行提交并推送分支：

```bash
git add seadronesse_supplement_queue
git commit -m "Add SeaDronesSee supplement experiment queue"
git push -u origin codex/seadronesse-supplement-queue
```

服务器拉取：

```bash
git fetch origin codex/seadronesse-supplement-queue
git switch --track origin/codex/seadronesse-supplement-queue
```

本轮本机 `git clone` 连接重置，未能取得服务器工作树；但 GitHub API 已确认 `main`/`npu` 当前包含 `protocol_v2`。运行前仍须以服务器工作树为准，并先保存未提交差异。

### 1. 对齐 run_id 与实际代码，不占推理卡

检查每个结果目录的 `run_manifest.json`、配置快照、Git commit 和 diff 备份。重点核对 A3 的 ROI JPEG 质量、A5 的 K 值、A6/A7/A8 的 `ratio`、packed 候选范围是否含 exploration。检查 ROI 非 patch 对齐时的边界补齐、重叠 ROI 产生的几何重复 patch，以及最终实际发送数量。缺少配置快照时，把该结果标为待核验；不得用当前 GitHub 文件代替历史执行文件。

### 2. 审计已有结果，不占推理卡

把此文件夹放到服务器仓库根目录的 `seadronesse_supplement_queue/`，然后执行：

```bash
python3 seadronesse_supplement_queue/audit_run_outputs.py \
  --root experiments/runs \
  --output experiments/supplement_audit.json \
  --check-packet-files
```

它检查逐帧行、重复 image_id、失败状态、逐帧 packet_bytes 与 summary 均值、保存的数据包长度和 SHA256、manifest 与预测文件是否存在。它不验证检测器实现，也不重新计算 COCO AP。丢弃了数据包时，先按包索引和固定编码配置重生成核查样本；不能只拿均值证明每帧记录完整。

### 3. 选择器审计（不占推理卡）

`audit_selectors.py` 在同一批图像上分别计算 support-only 与 support+exploration 候选集，并记录 uniform、spatial、evidence 三种选择的候选数、预算、去重数和越界数。需要先把 `--repo-root`、GT、图片目录和 edge cache 指向服务器实际文件：

```bash
python3 seadronesse_supplement_queue/audit_selectors.py \
  --repo-root /app/Salience-DETR \
  --gt /app/data/seadronessee/compressed_version/annotations/instances_val2017.json \
  --img-dir /app/data/seadronessee/compressed_version/val2017 \
  --edge-cache /app/Salience-DETR/experiments/edge_cache/edge_best_conf001.jsonl \
  --out-dir experiments/supplement_audit/selectors \
  --ratio 0.25 --expand 0.1 --explore-k 8
```

这份报告是基于当前 checkout 的 selector 代码做计数诊断，不会自动证明历史 run 使用了同一实现，也不检查最终图像覆盖顺序。要验收历史结果，仍需对照其 commit、diff、配置快照和 `build_packet`；如有候选差异，再决定是否重跑受影响比较。

### 4. 新版 resize＋JPEG 对照

先从已完成的新版主结果计算逐帧及平均完整 packet 大小；参数在历史 200 帧无法恢复时不可声称是在原 C_dev 上校准。用不查看标签的字节校准找到接近目标的 resize/JPEG 参数，随后冻结参数与哈希，再以同一云端权重、预处理、后处理和 COCO evaluator 推理：

| 顺序 | 比较目标 | 建议 run_id |
| --- | ---: | --- |
| 1 | 约 54 KB（与新版预测引导 25% 同链路匹配） | `resize_match_54kb_v2` |
| 2 | 约 360 KB（与摘要主推区域工作点匹配） | `resize_match_360kb_v2` |
| 3 | 约 314 KB（可选/第三工作点） | `resize_match_314kb_v2` |

按“实际平均包长差不超过 5%”标为匹配；超过则保留结果并如实标注。记录每帧 scale、JPEG quality、编码后真实大小、解码输入形状、预测 JSON、失败日志与评估参数。切勿拿旧输出融合路径的 JPEG 数字当新版对照。

`calibrate_resize_bytes.py` 会在均匀抽样图像上测所有 scale/quality 的实际 ZIP packet 字节，不读 GT、不占推理卡。选定参数后，把 scale/quality 写入 `queue.json` 对应任务的 `params`，配置服务槽，再启用该任务。`resize_jpeg_runner.py` 临时向现有 `run_a.CONFIGS` 注册 `ctx` 配置并调用同一 packet runner，不改上游文件；其解码、检测和评估路径因此复用当前 runner。若有可用校准开发集，应在冻结参数前按预先声明的 AP_S→mAP→字节次序选择；历史 200 帧不可恢复时，要明确说明只能按 bytes 选参数，不能宣称复现原校准协议。

校准命令示例：

```bash
python3 seadronesse_supplement_queue/calibrate_resize_bytes.py \
  --repo-root /app/Salience-DETR \
  --image-dir /app/data/seadronessee/compressed_version/val2017 \
  --output experiments/supplement_audit/resize_candidates.csv \
  --targets-kb 54,314,360
```

该校准只给出按字节接近目标的 scale/quality 候选，不挑 AP 最优点。正式推理前必须核实对应 run config、服务 URL、图像清单和评估配置，并把选定参数写回队列任务。

### 6. 从已保存预测补评估和 edge-only

`evaluate_predictions.py` 可对已有 COCO 格式预测重算总体及类别 AP，也可按 image ID 子集评估；需确保 GT、类别 ID、score floor、maxDets 等和论文所用评估协议一致。它不生成新预测：

```bash
python3 seadronesse_supplement_queue/evaluate_predictions.py \
  --gt /app/data/seadronessee/compressed_version/annotations/instances_val2017.json \
  --predictions experiments/runs/main_v2/predictions_coco.json \
  --out-dir experiments/supplement_audit/main_v2_eval \
  --score-floor 0.05
```

`edge_only_predictions.py` 将 edge cache 的 YOLO 类别映射到 COCO 类别，生成可交给上述 evaluator 的预测文件。先根据训练/数据配置核验映射，不能默认类别 ID 一致：

```bash
printf '{"0": 1, "1": 2}' > experiments/supplement_audit/class_map.json
python3 seadronesse_supplement_queue/edge_only_predictions.py \
  --edge-cache experiments/edge_cache/edge_best_conf001.jsonl \
  --class-map experiments/supplement_audit/class_map.json \
  --output experiments/supplement_audit/edge_only_predictions.json \
  --score-floor 0.05
```

示例 class map 仅展示格式，类别 ID 必须按实际 COCO `categories` 和模型 label 映射填写。此步骤只转换已保存的边缘预测，不会模拟 0 KB packet 下的云端输出。

### 7. 8 卡排队方式

`run_queue.py` 支持多个已验证的服务槽并行调度。实验进程需要各自唯一 run_id、输出目录和独立 HTTP 服务端口。每个槽在 `queue.json` 里填写 `name/device/url/health_url`；按现场 `npu-smi info` 和服务健康检查确认可用后，再把 `--max-parallel` 调到空闲槽数量。没有验证过的卡或端口不要填入。报告交接文档曾标记部分卡和端口被其他任务占用，这只是交接时状态，开跑前要重新确认。

队列调用示例：

```bash
python3 seadronesse_supplement_queue/run_queue.py \
  --queue seadronesse_supplement_queue/queue.json \
  --repo-root /app/Salience-DETR \
  --output experiments/queue_logs \
  --max-parallel 1
```

在 `queue.json` 中将已核实任务设为 `enabled: true` 并添加服务槽后再运行。`--max-parallel 8` 只表示最多同时发起八个独立任务，不会替你创建八个检测服务，也不意味着八张卡当前都空闲。

## 建议最终汇总

完成必要推理后，用同一预测集重新计算 AP@[.50:.95]、AP50、AP_S/AP_M/AP_L、类别 AP、载荷均值/中位数/P90/P95，并对同帧的本文与 resize 预测做配对区间。补齐 `run_manifest.json`、`per_image.csv`、`packet_index.csv`、预测 JSON、COCO 评估日志、失败记录和论文表格行溯源。无需为了这些统计再推理一次。

## 文件

- `run_queue.py`：JSON 命令队列，支持独立服务槽、并发上限、逐任务日志。
- `audit_run_outputs.py`：对现有运行目录做结构与包字节核查。
- `audit_selectors.py`：对当前代码做候选范围、选择数量、重复与边界诊断。
- `calibrate_resize_bytes.py`：在抽样图像上做实际包字节校准，保留候选表。
- `resize_jpeg_runner.py`：通过当前 protocol-v2 runner 执行 resize+JPEG 配置。
- `evaluate_predictions.py`：从保存的 COCO 预测重算总体与逐类别 AP。
- `edge_only_predictions.py`：将 edge cache 按显式类别映射转换成 COCO 预测。
- `queue.json`：剩余工作队列模板，推理项默认关闭。
- `install_into_repo.ps1`：将工具包复制到本地仓库 checkout，不会自动提交或推送。
- `calibrate_roi_bytes.py`：只按完整 packet 字节校准 B3/B6 简单 ROI 参数，不读取目标标注、不调用云端。
- `queue_next.json.template`：A4 网格随机三种 seed 与 B3/B6 的八卡队列模板；校准和服务映射核对前所有任务均关闭。

### 本轮新增任务

旧 A4 是随机 16×16 patch，不能代替方案定义的 6×10 网格随机选 8 格。`protocol_v2.run_a --config A4_grid --seed N` 会为每帧按 `seed + image_id * 1000003` 派生随机流，并将 seed 与 row-major cell ID 写入 `per_image.csv`。建议使用 11、23、37 三个预先固定的 run seed，不拼接或挑选最高分。

简单 ROI 使用 `roi_scale` 对 ROI crop 做等比例降采样，包内 manifest 记录参数，服务端仅凭 packet 解码并恢复 ROI 原尺寸。先运行字节校准：

```bash
python3 -u seadronesse_supplement_queue/calibrate_roi_bytes.py --repo-root /app/Salience-DETR --gt /app/data/seadronessee/compressed_version/annotations/instances_val2017.json --img-dir /app/data/seadronessee/compressed_version/val2017 --edge /app/Salience-DETR/experiments/edge_cache/edge_best_conf001.jsonl --output /app/Salience-DETR/experiments/supplement_audit/roi/roi_candidates.csv --targets-kb 87,311
```

候选表只用于选字节可行配置；若原开发 200 帧清单不可恢复，不根据验证集 AP 选参数。把 `queue_next.json.template` 复制到服务器为 `queue_next.json`，审阅候选并填入 B3/B6 参数，再重新核验端口绑定的物理 NPU、启用任务。
