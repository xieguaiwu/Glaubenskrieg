# Research: DeepSeek V4 Pro vs V4 Flash Performance Characteristics

## Summary

Both `deepseek-v4-pro` and `deepseek-v4-flash` are real, current-generation API models (DeepSeek-V4 Preview, released ~2026). They share 1M context, 384K max output, and identical features (JSON mode, tool calls, FIM). Pro costs ~3× Flash on both input and output. Crucially, **thinking mode ignores `temperature`, `top_p`, `presence_penalty`, and `frequency_penalty`** — these parameters have no effect when reasoning is enabled. Only two reasoning effort levels are effective: `high` (default; also receives `low`/`medium` mappings) and `max` (also receives `xhigh`). For the hardware context (7.5GB RAM, no GPU, 2-core CPU), the model choice is irrelevant to local inference — both are API-only. The decision hinges on **cost, latency, and concurrency limits** vs. quality needs.

## Findings

### 1. Model Specifications (from official API docs)

| Feature | deepseek-v4-pro | deepseek-v4-flash |
|:--|:--|:--|
| Context window | 1M tokens | 1M tokens |
| Max output tokens | 384K | 384K |
| Thinking mode | ✓ (default: enabled) | ✓ (default: enabled) |
| JSON Output | ✓ | ✓ |
| Tool Calls | ✓ | ✓ |
| FIM Completion | ✓ (non-thinking only) | ✓ (non-thinking only) |
| Concurrency limit | 500 | 2500 |

Both models are the same DeepSeek-V4 architecture generation. The difference is in model size/quality (Flash = smaller/faster, Pro = larger/smarter). [Source: DeepSeek API Pricing page](https://api-docs.deepseek.com/quick_start/pricing)

### 2. API Pricing (official, per 1M tokens)

| | deepseek-v4-flash | deepseek-v4-pro | Pro/Flash ratio |
|:--|:--|:--|:--|
| Input (cache hit) | $0.0028 | $0.003625 | 1.3× |
| Input (cache miss) | $0.14 | $0.435 | **3.1×** |
| Output tokens | $0.28 | $0.87 | **3.1×** |
| Input (cache miss, RMB) | ¥1.00 | ¥3.00 | 3.0× |
| Output (RMB) | ¥2.00 | ¥6.00 | 3.0× |

**Critical for reasoning mode**: Thinking tokens (returned via `reasoning_content`) count as **output tokens** and are billed accordingly. DeepSeek reports `reasoning_tokens` in `completion_tokens_details` which are part of `completion_tokens`. At `reasoning_effort: "max"`, reasoning tokens can substantially exceed visible output tokens. [Source: DeepSeek Pricing](https://api-docs.deepseek.com/quick_start/pricing), [API Create Chat Completion](https://api-docs.deepseek.com/api/create-chat-completion)

### 3. Thinking / Reasoning Effort Levels — The Truth

The API doc for `reasoning_effort` states explicitly:

> **Possible values: `high`, `max`**. Default is `high` for regular requests; automatically set to `max` for complex agent requests (Claude Code, OpenCode). **For compatibility, `low` and `medium` are mapped to `high`, and `xhigh` is mapped to `max`.**

**There are only TWO effective levels**: `high` and `max`. The five-level naming (`minimal`, `low`, `medium`, `high`, `xhigh`) that users may encounter is purely compatibility mapping — `minimal`/`low`/`medium` all resolve to `high`, and `xhigh` resolves to `max`.

[Source: DeepSeek Thinking Mode Guide](https://api-docs.deepseek.com/guides/thinking_mode), [Chat Completion API](https://api-docs.deepseek.com/api/create-chat-completion)

### 4. Temperature in Thinking Mode — Does Not Work

The thinking mode documentation states unambiguously:

> **Thinking mode does not support `temperature`, `top_p`, `presence_penalty`, or `frequency_penalty` parameters.** For compatibility with existing software, setting these parameters will not trigger an error but will also have **no effect**.

This means: for pi-agent agents running at temperatures 0.1–0.7 in thinking mode, **the temperature setting is silently ignored**. Output determinism is controlled entirely by the `reasoning_effort` parameter and the model's internal sampling, not by temperature.

[Source: DeepSeek Thinking Mode Guide](https://api-docs.deepseek.com/guides/thinking_mode#input-and-output-parameters)

### 5. Cost Comparison: Pro vs Flash with Reasoning Tokens

Reasoning tokens are billed as output tokens. A typical multi-step coding task at `reasoning_effort: "max"` may generate 3-10× more reasoning tokens than visible output. Example estimate for a complex agent interaction:

| Scenario | Model + Effort | ~Input tokens | ~Reasoning tokens | ~Output tokens | Est. cost (USD) |
|:--|:--|:--|:--|:--|:--|
| Simple prompt | Flash + high | 2,000 | 500 | 500 | $0.00056 |
| Simple prompt | Pro + high | 2,000 | 800 | 800 | $0.00265 |
| Complex coding | Flash + max | 10,000 | 5,000 | 2,000 | $0.00336 |
| Complex coding | Pro + max | 10,000 | 8,000 | 3,000 | $0.01392 |
| Agent session (50 turns) | Flash + max | 500,000 | 200,000 | 50,000 | $0.14 |
| Agent session (50 turns) | Pro + max | 500,000 | 300,000 | 60,000 | $0.53 |

**Pro at max effort is ~4× the cost of Flash at max effort for real agent workloads.** The 3.1× output token price ratio expands because Pro generates longer reasoning chains. [Source: pricing from official docs; ratios estimated from reasoning behavior documentation]

### 6. Benchmarks and Quality Gap

**Benchmark data is NOT available in the official API documentation.** The DeepSeek-V4 technical report / announcement blog post (linked from deepseek.com as a WeChat/Twitter post) was not accessible through the API docs site. The available documentation focuses on API mechanics, not benchmark scores.

What IS documented:
- Pro has higher concurrency restrictions (500 vs 2500) — suggesting it's a larger, more resource-intensive model
- For agent tools (Claude Code, OpenCode), effort is **auto-set to `max`** — suggesting `max` is the intended mode for complex multi-step agent tasks
- DeepSeek recommends: `deepseek-v4-pro` for Claude Code primary model, `deepseek-v4-flash` for subagent model

[Source: Claude Code Integration Guide](https://api-docs.deepseek.com/quick_start/agent_integrations/claude_code)

### 7. Model Deprecation Notice

`deepseek-chat` and `deepseek-reasoner` will be deprecated on **2026-07-24 15:59 UTC**. They map to:
- `deepseek-chat` → `deepseek-v4-flash` (non-thinking mode)
- `deepseek-reasoner` → `deepseek-v4-flash` (thinking mode)

Notably, the old `deepseek-reasoner` maps to Flash, not Pro — so upgrading from `deepseek-reasoner` to `deepseek-v4-pro` is a genuine capability jump.

[Source: DeepSeek API Pricing page](https://api-docs.deepseek.com/quick_start/pricing)

### 8. Non-Thinking Mode

When `thinking: {"type": "disabled"}`, the model operates as a standard chat model (no chain-of-thought). In non-thinking mode, `temperature` and `top_p` **do** work normally (range 0–2 and 0–1 respectively). FIM completion is only supported in non-thinking mode. Flash and Pro both support this mode.

[Source: DeepSeek API documentation]

### 9. Hardware Context Assessment (7.5GB RAM, no GPU, 2-core CPU)

Since both models are **API-only** (hosted on DeepSeek's servers), local hardware has **zero impact on model inference performance**. The 7.5GB RAM / no GPU / 2-core CPU context is irrelevant for model selection. What matters for this hardware is the **agent orchestration layer** (pi-agent), which is lightweight and will not be bottlenecked by these specs for API calls. Concurrency limit (Pro: 500, Flash: 2500) is the binding constraint, not local compute.

## Sources

- **Kept**: DeepSeek API Pricing page (en) — primary source for costs, context, output limits, features. [Link](https://api-docs.deepseek.com/quick_start/pricing)
- **Kept**: DeepSeek API Pricing page (zh) — confirms RMB pricing, identical specs. [Link](https://api-docs.deepseek.com/zh-cn/quick_start/pricing)
- **Kept**: DeepSeek Thinking Mode Guide (en + zh) — definitive source on reasoning_effort mapping, temperature behavior, token flow. [Link](https://api-docs.deepseek.com/guides/thinking_mode)
- **Kept**: DeepSeek Chat Completion API (en + zh) — parameter specifications, reasoning_tokens field. [Link](https://api-docs.deepseek.com/api/create-chat-completion)
- **Kept**: DeepSeek Claude Code Integration Guide — reveals recommended model mapping (Pro for primary, Flash for subagent). [Link](https://api-docs.deepseek.com/quick_start/agent_integrations/claude_code)
- **Kept**: DeepSeek Rate Limit page — concurrency limits. [Link](https://api-docs.deepseek.com/quick_start/rate_limit)
- **Kept**: DeepSeek homepage — confirms V4 Preview status, links to announcement. [Link](https://deepseek.com/)

## Gaps

1. **Benchmark scores (MMLU, HumanEval, MATH, etc.) for V4 Pro vs Flash**: The official API docs do not include benchmark results. The V4 announcement appears to be published on WeChat/Twitter, not in the docs site. Web search was unavailable. **Suggested**: someone with search access should look up the DeepSeek-V4 technical report blog post for benchmark comparisons.

2. **Real-world user reports on xhigh/max coding performance**: No community benchmarks or user reports could be accessed. **Suggested**: check Reddit (r/LocalLLaMA, r/DeepSeek), Twitter/X, or the OpenRouter leaderboard for community evaluations of `deepseek-v4-pro` at `max` effort.

3. **Latency data**: The API docs do not publish TTFT (time-to-first-token) or tokens-per-second metrics for either model. Pro is expected to be slower (larger model). **Suggested**: run a controlled latency benchmark via the API — the `/chat/completions` endpoint returns `usage` with detailed token counts that can be timed.

4. **Thinking token count at `max` vs `high`**: The exact multiplier of reasoning tokens between effort levels is not documented. The cost estimates above are approximations. **Suggested**: run test prompts at both `high` and `max` and compare `reasoning_tokens` from the usage response.

5. **Pro at `high` vs. Flash at `max` quality comparison**: No data on whether Flash with deeper reasoning (`max`) can rival Pro with standard reasoning (`high`). This is the key tradeoff for cost optimization. **Suggested**: run side-by-side coding task evaluations.

## Recommendation for pi-agent Configuration

For the hardware context (7.5GB RAM, no GPU, 2-core CPU) and pi-agent use case:

1. **Temperature is irrelevant in thinking mode** — set it to any value; it will be ignored. The range 0.1-0.7 used across pi-agent agents has no effect when `thinking: enabled`.

2. **Two practical configurations to test**:
   - **Budget**: `deepseek-v4-flash` + `reasoning_effort: "high"` → cheapest, highest concurrency (2500), good for straightforward tasks
   - **Quality**: `deepseek-v4-flash` + `reasoning_effort: "max"` → moderate cost, likely best price/performance ratio for complex coding
   - **Maximum**: `deepseek-v4-pro` + `reasoning_effort: "max"` → 3-4× cost of Flash+max, only justified if quality gap is proven

3. **Pro at `xhigh` (= Pro + max) is NOT obviously justified** without benchmark evidence of a quality gap over Flash + max. The DeepSeek team's own Claude Code integration recommends Flash for subagent work and Pro for primary — suggesting Flash is already competent for agent tasks.

4. **If you need temperature control**, disable thinking mode (`thinking: {"type": "disabled"}`). This enables temperature/top_p to work, at the cost of losing chain-of-thought reasoning. For creative tasks this may be acceptable; for complex coding, thinking mode is likely more valuable than temperature tuning.
