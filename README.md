# ChunkingRobustRAG

当前仓库先冻结阶段 0 的 API 和环境契约。研究主线使用 OpenAI-compatible Chat Completions 接口生成答案和执行 judge；embedding 保持本地 E5，避免同时改变多个实验变量。

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

