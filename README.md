# ChunkingRobustRAG

仓库包含阶段 0 的 API/环境契约和阶段 1 的数据准备、证据对齐流程。研究主线使用 OpenAI-compatible Chat Completions 接口生成答案和执行 judge；embedding 保持本地 E5，避免同时改变多个实验变量。

## 环境

- Python 3.12.14
- 直接依赖写在 `requirements.in`
- 传递依赖和哈希写在 `requirements.lock`
- API key 只通过本地 `.env` 或进程环境变量提供，不提交到 Git
- 创建虚拟环境前需要安装 Python 3.12，并确保 `py -3.12 --version` 可用

创建环境并安装锁定依赖：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --require-hashes -r requirements.lock
```

不要在已经激活的 `.venv` 中再次运行 `python -m venv .venv`。如果环境已存在，先退出旧终端，在新终端直接激活并检查：

```powershell
.\.venv\Scripts\Activate.ps1
python --version
python -c "import openai; print(openai.__version__)"
```

## 配置

复制 `.env.example` 为 `.env`，填写 `LLM_API_BASE`、`LLM_API_KEY` 和 `LLM_MODEL`。`JUDGE_API_BASE`、`JUDGE_API_KEY`、`JUDGE_MODEL` 未填写时分别回退到 LLM 配置。

YAML 配置可放在 `configs/api.yaml`；环境变量优先于 YAML，显式传给 smoke test 的 `--config` 优先选择该文件。API base URL 原样传给 OpenAI-compatible 客户端，不自动修改路径。

## 检查

离线检查不访问网络，也不需要 API key：

```powershell
python -m scripts.smoke_api --dry-run
python -m pytest
```

真实 API smoke test：

```powershell
python -m scripts.smoke_api --live
```

live 模式会输出两次调用的状态，并验证第二次相同请求命中本地缓存。日志只记录 request id、模型、状态、延迟、token usage、重试次数和错误码，不记录 key、请求头或完整 prompt。

## 阶段 1：准备干净数据

本阶段不需要 `.env` 或 API key。默认研究配置固定 `seed=42`、`data_version=clean-v4-hardneg`、`schema_version=data-v2`；NQ 和 HotpotQA 分别准备 100 条目标问题和 2,000 篇干净文档，两个知识库独立使用。严格 hard-negative 筛选需要本地 `intfloat/e5-base-v2`，数据构建不会自动联网下载模型。

先把完整的 E5 模型目录放到 `models/intfloat-e5-base-v2/`，目录内应包含 Sentence Transformers 的配置、tokenizer 和权重文件。配置中的 `hard_negative_e5_model_path` 是相对仓库根目录解析的；模型目录缺失或不能离线加载时，命令会直接失败。构建阶段会把模型文件的 SHA-256 汇总写入 manifest，模型变化必须使用新的 `data_version`。

如果当前机器允许单独下载模型，可以先执行下面的准备命令，再使用 `--offline` 构建数据；数据构建本身不会下载模型：

```powershell
hf download intfloat/e5-base-v2 --local-dir models/intfloat/e5-base-v2
```

```powershell
python -m scripts.prepare_data --dataset all
python -m scripts.prepare_data --dataset all --offline
python -m scripts.prepare_data --dataset all --validate-only
```

可以用 `--dataset nq` 或 `--dataset hotpotqa` 单独运行。首次默认运行会下载缺失原始文件；`--offline` 只使用本地文件，缺少文件即失败；`--validate-only` 只读取准备好的版本，不需要原始文件，也不访问网络。两种数据集分别原子发布：其中一个失败时，另一个已经通过校验的版本仍然保留。失败返回非零退出码，错误写入 stderr，成功摘要写入 stdout。

### 原始地址返回 403 或无法连接时

如果 Google Storage 返回 `HTTP 403: Forbidden`，或者 CMU 下载域名不可达，选择单独的 Hugging Face 来源配置：

```powershell
python -m scripts.prepare_data --config configs/data.huggingface.yaml --dataset all
python -m scripts.prepare_data --config configs/data.huggingface.yaml --dataset all --validate-only
```

该配置使用发布方命名空间下的 [NQ](https://huggingface.co/datasets/google-research-datasets/natural_questions) 和 [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa) Parquet 数据，固定完整 commit revision，仅下载 NQ `dev/validation` 的 7 个分片（约 1.34 GB）及 HotpotQA `distractor/validation` 的 1 个分片。NQ 保留 HTML、token/byte 范围和人工标注；HotpotQA 保留全部上下文和支持句。读取时还原为原始标注结构，再复用相同的清洗、筛选和证据对齐流程，并强制检查源记录数分别为 7,830 和 7,405。

来源变化显式使用 `data_version=clean-hf-v4-hardneg`，输出位于 `data/processed/clean-hf-v4-hardneg/nq/` 和 `data/processed/clean-hf-v4-hardneg/hotpotqa/`；配置、固定 revision、实际下载 URL 和文件哈希均写入 manifest。原 Google/CMU 配置保留，不会静默切换数据来源。已有本地 Parquet 文件可以放在该配置指定的路径，再加 `--offline` 处理。

403 等确定性的请求拒绝不重复重试；429 限流、408 超时和服务端错误仍按原规则重试。下载路径需要能够访问所选数据站点；离线测试通过不代表当前网络已能访问远程站点。

不下载真实数据的端到端检查使用仓库内的**合成测试样本**，输出版本与研究数据分开：

```powershell
python -m scripts.prepare_data --config configs/data.smoke.yaml --dataset all --offline
python -m scripts.prepare_data --config configs/data.smoke.yaml --dataset all --validate-only
```

合成样本仅用于验证程序行为，不是 NQ/HotpotQA 的研究样本或实验结果。

### 来源、配置和清洗规则

`configs/data.yaml` 锁定官方 NQ v1.0 开发集的 5 个原始分片，以及 HotpotQA `hotpot_dev_distractor_v1.json`。来源与格式参见 [NQ 官方说明](https://github.com/google-research-datasets/natural-questions)和 [HotpotQA 官方说明](https://github.com/hotpotqa/hotpot)。NQ 数据许可证为 CC BY-SA 3.0，HotpotQA 为 CC BY-SA 4.0；许可证与来源 URL 随 manifest 保存。

所有相对数据路径以仓库根目录解析。通过 `--config` 选择独立 YAML；修改 `sources.<dataset>.files[].path` 可指定已下载的本地原始文件，`url` 可以省略；可选 `sha256` 用于强制校验已知校验和。即使没有预设校验和，也会将实际 SHA-256 写入 manifest 并用于版本冻结。

- 统一 UTF-8、Unicode NFC 和换行；保留大小写、标点、数字及段落结构，不截断正文。NQ 按行流式读取，候选文档写入临时 SQLite，不在内存中堆积原始 HTML 页面。
- NQ 从 `document_tokens` 去除 HTML，同时构建原始 token 到清洗后字符位置的映射。HotpotQA 按句序拼接正文，空句仍占据原始编号位置。
- 文档按“数据集＋规范化标题＋清洗正文”生成 SHA-256 ID；相同身份文档合并来源，同名异文保持独立，避免按标题错误绑定证据。
- NQ 排除无答案、yes/no、多跨度及冲突答案，要求至少两位独立标注者对同一短答案达成一致，并具有有效 long answer。答案一致性仅处理 NFC、空白和大小写，不删除标点或改变数字。
- HotpotQA 排除空答案和 yes/no，保留官方答案字符串、bridge/comparison 类型及完整支持句链。支持句不要求包含最终答案；缺失标题、重复歧义标题、越界或空支持句会排除该问题。
- 两个数据集均限定答案最多 16 个空白分隔词。先按归一化问题去重，再按 `SHA-256(seed, dataset, 原始ID)` 排序抽样。HotpotQA 按合格候选题型比例，使用最大余数法分配名额。
- 纳入目标问题的全部非空上下文，再按稳定哈希顺序补充同一开发集背景文档至 2,000 篇。问题、文档不足或必需上下文超过配置容量均明确失败。

### 产物和加载接口

严格版本输出到 `data/processed/clean-v4-hardneg/nq/` 和 `data/processed/clean-v4-hardneg/hotpotqa/`（Hugging Face 配置对应 `clean-hf-v4-hardneg`），每个目录包含：

| 文件 | 主要字段与用途 |
| --- | --- |
| `documents.jsonl` | `doc_id`、`dataset`、`title`、`text`、`source_refs`、`text_sha256`，兼容规划中的干净文档 schema |
| `queries.jsonl` | `query_id`、`source_query_id`、`question`、`ground_truth`、`evidence_doc_ids`、`context_doc_ids`、`hard_negative_doc_ids`、`related_doc_ids`、`hard_negative_quality`、题型和版本 |
| `evidence.jsonl` | `evidence_id`、`query_id`、`doc_id`、`kind`、`start_char`、`end_char`、`text`、`source_ref` 和版本 |
| `manifest.json` | 来源/输出哈希、完整配置、抽样 ID、代码哈希及 Git revision、清洗/排除计数、标题冲突、长度分布和自动验收结果 |
| `audit_samples.jsonl` | 固定随机抽取的 20 条问题，附完整证据与来源，`review_status=pending`，供人工核查 |

所有原始 ID 都以字符串保存。`ground_truth` 是非空字符串数组，多个元素只表示来自同一答案标注的等价写法，不是多个联合答案。干净文档的 `source_doc_id=doc_id`、`is_poison=false`、`poison_version` 等于当前数据版本，污染字段为 null。证据文档、hard negative 和背景文档分别使用 `document_role=evidence|hard_negative|background`，`target_query_ids` 记录该文档服务的问题。

严格 hard negative 的候选先在完整开发集上用标题和正文前 1,024 个 token 做 BM25 Top-50 初筛，再使用本地 E5 按 `0.4 × query_e5_cosine + 0.4 × evidence_e5_cosine + 0.2 × normalized_bm25` 排序。候选必须通过实体/主题锚点、问题类型、答案泄漏、证据近重复和 `query >= 0.55`、`evidence >= 0.60` 两个 E5 阈值，并且每题至少 2 篇进入 BM25 Top-20；不满足条件的问题会被淘汰并重新抽样，不能降低阈值或复制文档。manifest 的 `hard_negative.candidate_audit` 保存每个候选的排名、分数、筛选原因和最终状态。

字符位置使用 Python Unicode 字符索引和左闭右开区间，必须满足 `document.text[start_char:end_char] == evidence.text`。NQ 额外保留原始 token/UTF-8 byte 跨度、annotation ID 和原始答案；HotpotQA 保留原始问题 ID、上下文位置、标题及零起始 `sentence_id`。这些位置不等同于后续 chunk tokenizer 的 token 位置。

```python
from src.data import load_documents, load_queries, load_evidence, validate_dataset

directory = "data/processed/clean-v4-hardneg/nq"
documents = load_documents(f"{directory}/documents.jsonl")
queries = load_queries(f"{directory}/queries.jsonl")
evidence = load_evidence(f"{directory}/evidence.jsonl")
report = validate_dataset(directory)
```

单文件加载器会校验字段和记录身份，`validate_dataset` 进一步校验数量、文件哈希、跨文件引用、版本、审计样本和证据跨度。后续检索只索引文档正文及正常标题/来源，问题、真实答案和证据标签单独加载，不拼接进知识库正文。

### 版本冻结与人工验收

同一版本重跑会检查原始文件哈希、配置和准备模块代码哈希；一致时复用已通过校验的输出。发生变化时修改 `data_version`（例如 `clean-v2`），不会覆盖旧版本。`--validate-only` 校验冻结产物及当前配置，不要求当前代码或原始文件与历史版本一同保留。下载与预处理失败不会留下可被当作完整版本的半成品。

自动验收要求数量精确、文档/问题/证据 ID 唯一、无空正文和悬空引用、干净标记正确，证据引用完整率和跨度回读通过率均为 100%。manifest 的 `status=validated` **只表示自动检查通过**；`human_review_status=pending` 不代表人工复核完成。

人工分别核查两个数据集的 20 条审计样本，检查答案来源、证据位置及 HotpotQA 多跳链。将审核人、结论、备注和对应 manifest 哈希记入独立的 `data/reviews/` 文件，不直接修改冻结的审计样本；发现清洗/对齐问题时修正规则，增加版本后重跑。

验收统计使用字符数和空白分词数，不提前固定阶段 3 tokenizer。实验结论限于短答案、受控规模开发集语料库；NQ 文档是页面、HotpotQA 文档是段落，两者长度分布在各自 manifest 中单独报告。原始数据、生成产物和审核记录被 Git 忽略，合成测试样本随代码保存。

