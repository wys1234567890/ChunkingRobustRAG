# ChunkingRobustRAG

仓库包含阶段 0 的 API/环境契约、阶段 1 的数据准备与证据对齐、阶段 2 的隐式事实污染数据集、阶段 3 的分块与检索模块，以及阶段 4 的 API RAG 推理。研究主线使用 OpenAI-compatible Chat Completions 接口生成答案和执行 judge；embedding 保持本地 E5，避免同时改变多个实验变量。阶段 5（自动评估）及以后依赖批量 LLM 调用，在阶段 4 跑通后再实现。

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

## 阶段 2：隐式事实污染数据集

本阶段读取阶段 1 的冻结干净库，按目标 query 注入**隐式事实污染**：把真实答案替换为一个结构相同但错误的陈述，写成普通说明句，不含“忽略上文”“请回答”等显式指令。完整规格见 `stage2.md`，配置固定在 `configs/corruption.yaml`。

```powershell
python -m scripts.build_corruption --dataset all
python -m scripts.build_corruption --dataset all --validate-only
```

- 污染等级固定为 `0/5/10/20%`，主线按 query 级预算控制：每个 query 分别绑定 `0/1/2/4` 篇专属污染文档。两个数据集各有 100 条 query 时，四个档位的污染文档数分别为 0、100、200、400 条。
- `query_allocation=fixed_per_query` 要求每个 query 的污染文档数量严格相同；`poison_docs_per_query` 写入配置、query 记录和 manifest。每条污染文档只服务一个 query，不能通过共享文档或不均匀分配改变攻击剂量。
- 输出目录为 `data/poisoned/fact-v2-entity-append-query-budget/<dataset>/ratio-0p00|0p05|0p10|0p20/`，每个目录包含 `documents.jsonl`、`queries.jsonl`、`evidence.jsonl`、`poison_records.jsonl`、`audit_samples.jsonl` 和 `manifest.json`。
- `poison_records.jsonl` 逐条记录真实事实、错误事实、来源文档、来源证据、替换次数、构造方式、审核状态和文本哈希，可完整回溯；`queries.jsonl` 增加 `poison_doc_ids`、`poison_docs_per_query`、`poison_ratio_target`、`poison_ratio_actual`、`poison_ratio_clean` 和 `poison_version`。
- 生成过程先写 staging 目录，再用独立的 `validate_poison_dataset` 校验 ID、哈希、真实事实是否彻底消失、错误事实是否出现、显式指令 blocklist、query 覆盖率和固定 query 预算；全部通过后才原子重命名为正式目录，失败会删除 staging，不会留下半成品。
- `status=validated` 只表示自动验收通过，`human_review_status=pending` 仍待人工抽查；发现问题时修改规则并提升 `poison_version` 重跑，不覆盖旧版本。
- 程序行为可用合成样本验证：`python -m scripts.build_corruption --synthetic --dataset nq`，产物与真实研究数据隔离。

## 阶段 3：分块与检索

本阶段读取阶段 1/2 的 `documents.jsonl`，按文档生成 chunk JSONL；输入可以是干净库，也可以是带 `is_poison=true` 的事实污染库。分块只索引文档正文，不把问题、答案或证据标签拼进知识库。

### 统一 chunk schema

每个 chunk 输出以下字段：

| 字段 | 说明 |
| --- | --- |
| `chunk_id` | 由文档、策略、参数、span 和文本哈希生成的稳定 ID |
| `doc_id` | 来源文档 ID，不会跨文档拼接 |
| `text` | 恰好等于 `document.text[start:end]` |
| `start` / `end` | Python Unicode 字符索引，左闭右开 |
| `is_poison` | 从源文档复制；缺失或非布尔值直接报错，避免污染文档被误当干净文档 |
| `strategy` | `fixed`、`semantic` 或 `hierarchical` |
| `chunk_size` | 以 tokenizer 计算的 token 预算 |
| `overlap` | 重叠比例（0–1 之间） |
| `tokenizer` | 实际使用的 tokenizer 名称，随 chunk 冻结 |
| `token_count` | 该 chunk 文本重新分词后的实际 token 数，保证不超过 `chunk_size` |
| `dataset` / `source_doc_id` / `poison_version` | 数据集与污染溯源字段，便于回连阶段 2 记录 |
| `section_path` | hierarchical 的层级路径，如 `["Root", "Child"]`；无标题文档记为 `["body"]`，其余策略为空 |

同一文档内不会出现空块、完全相同的 span 或跨文档串接；重叠块具有不同的 `start/end`，因此仍可区分。

按 token 窗口切片后，片段重新分词可能比全文分词多出边界 token。生成 chunk 时会按片段自身分词结果收缩末尾，保证 `token_count <= chunk_size`；因此固定长度块的实际长度可能略小于预算。

### 分块策略

- `fixed`：按 token 窗口切分，`step = chunk_size - int(chunk_size * overlap)`。
- `semantic`：先按句末标点和换行得到句子边界，再把完整句子装入 token 预算；超长单句回退到窗口切分。
- `hierarchical`：解析 Markdown 标题层级，按节切分并记录父级到当前标题的 `section_path`，节内再做窗口切分；无标题时按空行分段，单段落时按句子分节。因此即使文档没有标题，hierarchical 也不会退化成 fixed。

### 构建 chunk 库

研究主线使用本地 E5 对应 tokenizer，保证 chunk-size 与 embedding 模型口径一致：

```powershell
python -m scripts.chunk_documents `
  --input data/processed/clean-v1/nq/documents.jsonl `
  --output-dir data/processed/chunks/nq `
  --strategies fixed,semantic,hierarchical `
  --chunk-sizes 256,512,1024,2048 `
  --overlaps 0,0.1,0.2 `
  --tokenizer hf:intfloat/multilingual-e5-small
```

离线冒烟测试可显式使用空白分词器；它只用于检查程序行为，不能作为研究结果：

```powershell
python -m scripts.chunk_documents `
  --input data/processed/clean-v1/nq/documents.jsonl `
  --output-dir data/processed/chunks/nq-smoke `
  --strategies fixed,semantic,hierarchical `
  --chunk-sizes 512 `
  --overlaps 0 `
  --tokenizer whitespace `
  --limit 10
```

每个配置生成一个 JSONL，并写入 `manifest.json`，记录源文件哈希、tokenizer、chunk 数、输出哈希和 `chunk_config_version=stage3-v1`。更换 tokenizer、chunk-size 或 overlap 后必须重新生成并保留新版本，不能覆盖旧实验配置。

### 检索

BM25 与本地 E5 使用同一接口，`Top-k` 默认固定为 5：

```powershell
python -m scripts.retrieve --chunks data/processed/chunks/nq/fixed-cs512-ov00.jsonl --query "who wrote the declaration of independence" --retriever bm25
python -m scripts.retrieve --chunks data/processed/chunks/nq/fixed-cs512-ov00.jsonl --query "who wrote the declaration of independence" --retriever e5
python -m scripts.retrieve --chunks data/processed/chunks/nq/fixed-cs512-ov00.jsonl --queries data/processed/clean-v1/nq/queries.jsonl --retriever both --top-k 5 --output results/stage3/retrieval.jsonl
```

- BM25：`rank-bm25` Okapi，Unicode 词级切分，按分数降序、`chunk_id` 升序稳定打破平局。
- E5：`sentence-transformers` 加载 `intfloat/multilingual-e5-small`，query 加 `query: `、chunk 加 `passage: ` 前缀，归一化后做余弦相似度。该模型最大输入为 512 token；`chunk_size` 为 1024/2048 时 embedding 会被截断，实验记录中必须注明，或改用支持长上下文的 embedding 模型。离线环境可把完整模型目录放到 `models/intfloat-multilingual-e5-small/`，`scripts/retrieve.py` 用 `--model-name models/intfloat-multilingual-e5-small`，并配合 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。
- 每条结果包含 `score`、`rank`、`retriever`、`retriever_version`，并保留完整 chunk 字段；版本串记录包版本和 BM25 参数或 E5 模型名。

### 验收

```powershell
python -m pytest tests/test_chunking.py tests/test_retrieval.py
```

测试覆盖三种策略的 span 回读、重叠、句子边界、层级边界与父级路径、无标题文档的层级退化、污染标记、重复 ID、非法参数，以及 BM25/E5 的 Top-5 返回与稳定排序。真实 E5 首次运行会下载模型；没有本地模型时先完成 BM25 和注入式假模型测试，不要因此跳过接口验收。

## 阶段 4：API RAG 推理

本阶段把阶段 3 的检索结果接入 LLM，生成每条 query 的最终答案。核心模块是 `src/rag.py`，脚本是 `scripts/run_experiment.py`。

### RAG pipeline

`RagPipeline` 串联三个步骤：

1. **检索**：调用传入的 `Retriever`（BM25 或本地 E5）取 Top-k chunk。
2. **上下文构造**：`build_context` 把命中的 chunk 按排名编号拼成 `[1] ... [2] ...` 文本块。
3. **生成**：`build_messages` 组装 system prompt + user（含 context 和 question），交给 `ApiClient.generate`。

system prompt 只要求模型基于给定上下文回答、证据不足时说不知道，不包含任何与投毒相关的指令；污染文档本身也不携带控制模型行为的文本。缓存键由 `ApiClient` 基于 messages + 模型 + 参数 + `prompt_version` + `config_version` 生成，因此相同的检索上下文会命中缓存，不同 chunk 配置互不干扰。

### 结果 schema

每条结果写入一行 JSONL，字段对应规划 10.2 并补充阶段 4 可计算的信息：

| 字段 | 说明 |
| --- | --- |
| `query_id` / `dataset` / `question` | 问题身份 |
| `chunk_type` / `chunk_size` / `overlap` | 分块配置（从 chunk 文件首条推断） |
| `retriever` / `retriever_version` / `top_k` | 检索配置 |
| `poison_ratio` / `poison_version` | 污染元数据 |
| `retrieved_poison` | Top-k 中是否存在 `is_poison=true` 的 chunk |
| `retrieved_chunk_ids` / `poison_chunk_ids` | 召回的 chunk ID 列表 |
| `ground_truth` / `false_facts` | 真实答案和该 query 对应的错误事实（来自 `poison_records.jsonl`） |
| `model_output` | 模型生成文本；失败时为空字符串 |
| `api_status` / `error_code` / `cache_hit` / `retry_count` | API 调用状态 |
| `api_model` / `prompt_version` / `config_version` | 版本溯源 |
| `latency_ms` / `input_tokens` / `output_tokens` / `estimated_cost` / `request_id` | 工程指标 |

`judge_label`、`clean_correct` 等评估字段留到阶段 5 填充。

### 运行实验

```powershell
python -m scripts.run_experiment `
  --chunks data/processed/chunks/fact-v1/nq/ratio-0p05/fixed-cs512-ov00.jsonl `
  --queries data/poisoned/fact-v2-entity-append-query-budget/nq/ratio-0p05/queries.jsonl `
  --poison-records data/poisoned/fact-v2-entity-append-query-budget/nq/ratio-0p05/poison_records.jsonl `
  --retriever bm25 `
  --top-k 5 `
  --output results/stage4/nq-0p05-fixed-bm25.jsonl `
  --poison-ratio 0.05 `
  --dataset nq
```

- `--limit N` 只处理前 N 条 query，用于 smoke test。
- `--poison-records` 可选；提供后会把 `target_query_id` 对应的 `false_fact.value` 挂到结果的 `false_facts` 字段。
- 输出以**追加模式**写入，已存在的 `query_id` 会被跳过，因此中断后重跑可续跑，不会重复调用 API。
- API key 从 `.env` 或环境变量读取，不写入结果或日志。

### 验收

```powershell
python -m pytest tests/test_rag.py
```

测试覆盖上下文编号、Top-k 截断、prompt 结构、`retrieved_poison` 判定、`ground_truth`/`false_facts` 透传、API 错误透传（`api_status=error`、`error_code` 保留）、空 `query_id` 拒绝，以及缓存命中时不重复调用 API。

## 阶段 2 与阶段 3 的边界

阶段 4 的 RAG 推理依赖真实 LLM API，但阶段 2 的污染数据、阶段 3 的分块与本地 BM25/E5 检索都不依赖外部 API，可以离线复现。阶段 5（规则评估 + LLM-as-judge）和阶段 6（统计与可视化）在阶段 4 结果稳定后实现。

