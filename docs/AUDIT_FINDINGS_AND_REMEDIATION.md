# 审计测试结论与修复方案

> 依据：`tests/test_audit_performance_and_hotspots.py`、`tests/test_audit_dynamic_verification.py`、`tests/test_example_run_audit.py`  
> 运行快照：**207 passed，1 skipped**（审计 + 全量回归）  
> 本文：**问题是什么 → 建议怎么改 → 改后的影响范围**；**✅ 已修复** 表示代码已落地并通过对应用例。

**修复进度**：P0 2.1–2.4 ✅ · P1 3.1–3.3 ✅ · P2 4.1–4.6 ✅ · P3 5.1/5.3/5.4 ✅（5.2 仅文档）

---

## 0. 这份东西是什么

**这就是性能 + BUG 的审计报告**，三个测试文件是审计用例的**可执行清单**；正文 §1～§5 是同一批发现的**修复方案与影响面**。

| 类型 | 大致数量 | 说明 |
|------|----------|------|
| **性能** | 多数（对话热路径、印记库、基础 UVA 模型） | 重复 `trace`/`tokenize`、重复惊奇计算、`pop(0)`、全表 `tokens()` 复制、GPU cache 整表失效等 |
| **BUG / 风险** | 少数但优先 | 指标算错（`new_imprint_tokens` ✅）、manifest 路径穿越、内心快照不对齐（✅）、部分 JSON 加载（✅） |
| **契约 / 卫生** | 若干 | 语料 `strict` 写死、空白 batch、死代码、整文件读 JSON 等 |

测试文件头写的「**不改业务实现**」是指：**审计用例本身不去改 `uva_model` / `example_run` 的源码**，而是用 mock、计数、读源码等方式**把问题测出来、写进报告**——不是「测到了也不算问题」。

### 三类测试文件分工

| 文件 | 侧重点 |
|------|--------|
| `test_audit_performance_and_hotspots.py` | 性能热点（重复计算、分配、复杂度） |
| `test_audit_dynamic_verification.py` | 性能 + 正确性风险的**动态证据**（trace 次数、快照不一致、IO/加载契约） |
| `test_example_run_audit.py` | `example_run` 的 **BUG**（指标合并）、**安全**（路径）、调度与清洗 |

### 为什么多数用例是绿的，却仍在「报问题」？

审计用例的通过方式通常是：**断言「坏模式确实存在」**——例如 `sensory_error` 被调 2 次、`produce_turn` 里 `tokenize` ≥ 6 次。  
**通过 = 审计项成立（确认有这类性能/结构问题）**，不是「代码健康、无需优化」。

**已修复的 BUG 项**会改用**正向断言**（期望正确行为），见 §2.1 / §2.3 / §2.4 与 §7。

---

## 1. 问题总览（按优先级）

| 优先级 | 类别 | 条目数 | 典型后果 | 已修复 |
|--------|------|--------|----------|--------|
| **P0** | 正确性 / 指标 / 安全 | 4 | 训练日志误导、路径穿越、内心排序错位、部分加载语义含糊 | **3 / 4** |
| **P1** | 性能热点（对话热路径） | 8 | 交互与 `produce_turn` 延迟… | 0（§3.4 随 2.3 部分缓解） |
| **P2** | 性能热点（印记库 / 基础模型） | 7 | 大词表、长跑训练… | 0 |
| **P3** | API / 契约 / 可维护性 | 6 | 静默失败、死代码… | 0 |

---

## 2. P0：应先修或先定契约

### 2.1 多 chunk 流式训练：`new_imprint_tokens` 被覆盖而非累加 · ✅ 已修复

**来源**：`test_example_run_audit.py` → `ChunkedRootStreamingMergedMetricsTests`

**修复摘要**（`example_run.py`）：循环内改为 `merged["new_imprint_tokens"] += int(metrics["new_imprint_tokens"])`；已去掉 `@expectedFailure`，回归用例通过。

**原现象**：`run_dialogue_training_chunked_root_streaming` 循环内：

```python
merged["new_imprint_tokens"] = int(metrics["new_imprint_tokens"])  # 赋值
```

而同循环里 `merged_imprints`、`cold_probe_tokens` 等使用 `+=`。多 chunk 时最终「新增印记词」≈**最后一 chunk 的增量**，低于词库真实净增。

**建议改法**（已实施）：

- 将第 605 行改为 `merged["new_imprint_tokens"] += int(metrics["new_imprint_tokens"])`；或循环结束后用 `final_total - initial_total` 统一覆写（与 `run_dialogue_training` 非流式路径第 952 行一致）。
- 去掉测试上的 `@unittest.expectedFailure`，保留 `assertEqual(metrics["new_imprint_tokens"], truth_new)`。

**影响范围**（已发生）：

| 范围 | 说明 |
|------|------|
| **代码** | 仅 `example_run.py` 中 `run_dialogue_training_chunked_root_streaming` 的 metrics 合并 |
| **行为** | 仅影响 **chunked-root 流式训练** 的日志/返回值；不改变 agent 内部印记 |
| **兼容** | 对外 CLI 打印数字变大（更接近真实）；依赖「低估的新增词数」做自动化的脚本需更新预期 |
| **测试** | ~~1 条 xfail → pass~~ **已完成** |

---

### 2.2 Chunk manifest：`relativePath` 可解析到 manifest 目录外

**来源**：`test_example_run_audit.py` → `CollectChunkedRootJobsPathTests`

**现象**：`_collect_chunked_root_jobs` 对 `relativePath` 做 `manifest_dir / rel` 解析，**未**要求结果落在 `manifest_dir` 内；`../evil.jsonl` 可读 manifest 父目录外文件（供应链 / 本地恶意 manifest 面）。

**建议改法**：

- 解析后 `chunk_path.resolve()`，断言 `chunk_path.is_relative_to(manifest_dir.resolve())`（或 `relative_to` 不抛）。
- 违规条目：**跳过并 warn**，或 **raise**（训练入口更宜 fail-fast）。
- 可选：拒绝含 `..` 的 `relativePath` 字符串。

**影响范围**：

| 范围 | 说明 |
|------|------|
| **代码** | `example_run.py` → `_collect_chunked_root_jobs`（及调用链 `load_dialogue_training_chunked_root` / streaming） |
| **行为** | 合法 manifest（路径均在子目录内）无变化；**越界路径不再被加载** |
| **运维** | 若现有数据包用 `..` 引用 chunk，需改 manifest 或复制文件到子树 |
| **测试** | 现有断言改为期望 `jobs` 为空或抛错；加一条「合法相对路径仍可用」 |

---

### 2.3 内心前沿：baseline 快照与 `internal_tick` 内快照不一致 · ✅ 已修复

**来源**：`test_audit_dynamic_verification.py` → `test_internal_monologue_imprint_snap_unifies_frontier_and_tick` 等

**修复摘要**（`uva_model/dialogue.py`，**方案 A**）：

- 新增 `_internal_monologue_imprint_snap`（`u_task=0`，`u_curiosity=max(σ)`）与 `_germinate_internal_intent()`；
- `_internal_frontier_candidates` 与 `internal_tick` 共用该快照；循环内 `max_surp` 复用前沿 `precomputed_surprise`（兼消 §3.4 重复计算）。

**原现象**：

- `_internal_frontier_candidates` 用 `_germinate_intent("", 0)` 构造 `baseline_snap` 算惊奇；
- `internal_tick` 循环内对 token `tok` 用 `_germinate_intent(tok, 0)` 的 `snap_pre` 再算 `max_surp`（并进入 EFE 印记项）。

当 `tok` 含 `?` 等时 **`u_task` 不同** → 同一印记下 **L2 惊奇不同** → 前沿排序用的惊奇与 EFE 用的惊奇**可能错位**（张力阈值与候选评分不一致）。

**建议改法**（已采用 **A**）：

- ~~**A（语义一致）**~~：前沿扫描与 `internal_tick` 内层**共用** `_internal_monologue_imprint_snap` + `_germinate_internal_intent`。
- **B / C**：未采用。

**影响范围**（已发生）：

| 范围 | 说明 |
|------|------|
| **代码** | `uva_model/dialogue.py`：`_internal_frontier_candidates`、`internal_tick`（及 `internal_tick_train_fast*` 若共用逻辑） |
| **行为** | **内心触发顺序、是否发声、best_efe** 可能变化；含问号的 trigger token 影响最大 |
| **序列化** | 无格式变更 |
| **测试** | audit + `test_dialogue.py::test_internal_monologue_imprint_snap_forces_zero_u_task` **已通过** |

---

### 2.4 对话 JSON 部分加载：缺字段时静默保留旧状态 · ✅ 已修复

**修复摘要**（`uva_model/dialogue.py`）：

- `apply_dialogue_model_dict`：**严格快照**（`sigma` + `word_imprints` 必填；`preference_state` 可省略→默认空偏好，非法 `format`→`ValueError`；先校验再提交）。
- 新增 `patch_dialogue_model_dict` 做显式增量更新。
- `dialogue_model_to_dict` 增加 `format_version: 1`。

**来源**：`test_audit_dynamic_verification.py` → `DialoguePartialLoadDynamicAuditTests`

**原现象 A**：`apply_dialogue_model_dict` 仅含 `word_imprints`、**无 `sigma`** 时，**不重置** `_sigma`，保留加载前槽位值。  
**原现象 B**：`preference_state.format` 错误时 **静默 return**，`branch_bias` 等保持旧值。

**历史方案对照**（严格 + `patch_dialogue_model_dict` 已选为默认与补充 API）：

| 策略 | 做法 | 适用 |
|------|------|------|
| **严格快照** | 缺 `sigma` / 非法 `preference_state` → `ValueError` | `apply_dialogue_model_dict` / `load_dialogue_model` |
| **增量 patch** | `patch_dialogue_model_dict`：仅更新出现的键 | 高级用户、热更新 σ / 偏好子集 |
| **宽松 + 可见** | 未采用 | — |

**影响范围**：

| 范围 | 说明 |
|------|------|
| **代码** | `dialogue.py`：`apply_dialogue_model_dict`、`_apply_preference_state_dict`；`example_run` / `dialogue_web_ui.py` 加载路径 |
| **行为** | 严格策略下，**旧版或不完整 JSON** 可能加载失败（需文档说明必填字段） |
| **兼容** | 当前依赖「只覆盖印记、不动 σ」的工作流会断裂，需迁移指南 |

---

## 3. P1：对话热路径性能

### 3.1 `produce_turn` / `_efe_best_reply`：同一句用户话被反复 `tokenize` / `_trace`

**来源**：

- `test_audit_dynamic_verification.py` → `test_produce_turn_calls_tokenize_many_more_times_than_listen_trace`
- `test_produce_turn_trace_call_count_has_sane_bounds`（约 **25～600** 次 `_trace`）
- `test_train_step_invokes_fewer_traces_than_produce_turn_same_text`

**现象**：倾听已 `_listen_trace` 一次；候选构造里 `_realize` / `epsilon_secondary_on_draft` / `tokenize(user_text)` 对**同一句**反复 trace。`train_step` 无 EFE，trace 次数远小于 `produce_turn`。

**建议改法**：

1. **倾听结果缓存**：`produce_turn` 内保存 `listen_trace.tokens`、`eps_in`，候选生成用 **tokens 切片** 而非再 `tokenize(整句)`。
2. **草稿评估**：`epsilon_secondary_on_draft` 已有 freeze/restore；可对 **同一轮** 用「草稿文本 → trace」批量化或限制诚实前缀 × depth 乘积。
3. **结构化候选**：保守稿只依赖已算好的 `surface` token，避免每条候选 `_realize` 全句分词。

**影响范围**：

| 范围 | 说明 |
|------|------|
| **代码** | `dialogue.py`：`produce_turn`、`_efe_best_reply`、各 `_build_*_draft`、`_realize` |
| **性能** | 交互延迟、Web UI「单轮对话」、审计测试中 trace 上界应**下降**（需重标定 25～600） |
| **行为** | 若草稿生成曾依赖「二次分词副作用」（极 unlikely），需对比 golden 回复；一般 **语义不变** |
| **测试** | 动态 audit 的 `assertGreaterEqual(6)` / trace 下界需按新实现调整 |

---

### 3.2 `_expected_free_energy`：`_resource_factor()` 单次评估内至少 2 次

**来源**：`test_expected_free_energy_calls_resource_factor_at_least_twice`

**现象**：`_lambda_explore()` 与 `explore_commit` 分支各读一次 `rf`（及 fatigue 等重复读 `resource_state()`）。

**建议改法**：在 `_expected_free_energy` 入口 **一次** `rf = self._resource_factor()`，`st = self.tokenizer.resource_state()`，向下传递。

**影响范围**：`dialogue.py` 单函数；**数值应完全一致**；EFA 热路径 CPU 略降。审计测试 call_count 需改为 `== 1` 或 `<= 1`。

---

### 3.3 `_efe_best_reply`：大量 `_restore_resource`（≥15 次/轮）

**来源**：`test_efe_best_reply_restores_resource_many_times`

**现象**：深度 × 前缀 × 分支循环中，为隔离草稿对资源的污染，频繁 `restore`；正确但贵。

**建议改法**：

- 将「评估草稿」改为在 **局部拷贝** 的 `(R,m,R_max,F_ema)` 上 trace，循环结束一次写回；或
- `PrecisionTokenizer._trace` 支持 **无副作用 simulate** 模式（只读快照参数）。

**影响范围**：

| 范围 | 说明 |
|------|------|
| **代码** | `dialogue.py` `_efe_best_reply`；可能触及 `tokenizer.py` |
| **风险** | 中：restore 遗漏会导致资源状态污染（已有 `test_epsilon_secondary_on_draft_leaves_frozen_resource_snapshot` 守护） |
| **测试** | 必须通过 restore 完整性测试 + 对话 golden |

---

### 3.4 `internal_tick`：`semantic_surprise_for_token` 前沿与内层各算一次 · ✅ 已修复（随 2.3）

**来源**：`test_internal_tick_does_not_recompute_semantic_surprise_in_loop`（原 `test_internal_tick_recomputes_semantic_surprise_per_token`）

**修复摘要**：`internal_tick` 循环内 `max_surp = precomputed_surprise`，不再二次调用 `semantic_surprise_for_token`。

**原建议改法**：`ordered` 携带 `(tok, precomputed_surprise)`，`internal_tick` 内 `max_surp = precomputed_surprise`（已与 2.3 一并落地）。

**影响范围**：`dialogue.py` `internal_tick`；内心 CPU 略降；行为与 2.3 一致。

---

### 3.5 仲裁平局：外部优先（设计记录，非必改）

**来源**：`test_arbitration_tie_favors_external`

**现象**：`g_ext_net <= g_int` → 外部先说。测试**固化当前设计**。

**若改**：改为内心优先或随机，需同步 `example_run._emit_user_line_with_internal_arbitration` 与文档。

**影响范围**：仅交互展示顺序；EFE 数值不变。

---

## 4. P2：`WordStateMemory` 与基础 UVA 模型

### 4.1 `UnifiedVariationalAttentionModel`：`sensory_error` 在 `free_energy` / `gradients` 中算两遍

**来源**：`test_audit_performance_and_hotspots.py` → `ModelDuplicateComputationTests`

**现象**：`free_energy` 先 `eps_o = sensory_error()`，再 `effective_drive()` 内部又 `sensory_error()`。

**建议改法**：`effective_drive(eps_o: Optional[...]=None)` 或 `free_energy` 内 `drive = self.effective_drive(eps_o)`。

**影响范围**：`uva_model/model.py`；`--mode base` 演示；**数学等价**；`step()` 每步两次梯度时收益更明显。

---

### 4.2 `tokens()` 每次 `list(_store.keys())` 全量复制

**来源**：`test_tokens_returns_fresh_list_each_time`、`test_soft_freeze_calls_tokens_once_per_invocation`、`test_top_surprises_full_scan_uses_tokens_iterator`

**建议改法**：

- 热路径改用 `for tok in self._store:` 或维护 `_token_list_version` + 脏标记；
- `soft_freeze_cold_tokens` / `top_surprises` 避免多次全表 keys 副本。

**影响范围**：`word_imprints.py` 多数扫描 API；**大词表（10⁴+ token）** 训练/内心扫描；对外 API 若承诺返回独立 list，可保留 `tokens()` 但内部不用。

---

### 4.3 `_trim_active_tokens` 使用 `removable.pop(0)`

**来源**：静态 + 动态（`test_trim_active_tokens_source_uses_pop_zero`、`PopCountingWordStateMemory`）

**建议改法**：改为 `pop()` **尾删**（配合 `reverse=True` 排序，语义是删优先级最低者）；或 deque。

**影响范围**：`word_imprints.py`；活跃词裁剪频繁时 **O(n²)→O(n)**；**休眠哪些 token** 在同等排序下应一致（子类测试已对齐 vanilla）。

---

### 4.4 缓存路径多余 list 分配

| 热点 | 来源 | 建议 | 影响 |
|------|------|------|------|
| `_frontier_one_hop_neighbors` 命中仍 `list(cached)` | hotspots | 只读路径返回 `tuple` 或要求调用方不修改 | 前沿 BFS；需确认无 in-place 修改 |
| `associated_tokens` 命中 `cached[:k]` 新 list | dynamic | 返回 `tuple` 或文档约定只读 | 联想探针候选 |

---

### 4.5 每条 `record`（含合并）`_invalidate_cache()` → GPU `_torch_cache` 整表作废

**来源**：`test_record_invalidates_global_torch_cache_version`、`test_merge_record_still_bumps_global_cache_version`

**建议改法**：

- 合并命中时 **不** 提升全局 version，仅更新受影响 token 的行；
- 或按 token 分片 torch cache，record 只失效对应 shard。

**影响范围**：`word_imprints.py` + CUDA 路径 `top_surprises` / `_top_surprises_torch`；长跑训练 **GPU 加速效果**；实现复杂度较高。

---

### 4.6 `top_surprises_lazy` + `max_active_tokens` 每次 `soft_freeze_cold_tokens`（全库排序）

**来源**：`test_top_surprises_lazy_invokes_soft_freeze_when_max_active_set`

**建议改法**：lazy 扫描前 **节流** soft_freeze（例如每 N tick 或 active 超限才做）；或增量维护 active 集合。

**影响范围**：训练期内心快路径、大容量印记；可能影响 dormant 比例。

---

## 5. P3：契约、IO、可维护性

### 5.1 `iter_corpus_jsonl_training_episodes` 写死 `strict=False`

**来源**：`test_training_episodes_iterator_hardcodes_strict_false`

**建议改法**：增加参数 `strict: bool = False` 透传到 `iter_corpus_jsonl_records`；`example_run` CLI 与 `skip_bad_lines` 对齐。

**影响范围**：`corpus_jsonl.py`、`example_run` 训练入口；默认保持 False **无破坏**；strict=True 时无正文行会抛错。

---

### 5.2 `read_json_document` 整文件 `read_bytes`

**来源**：`test_read_json_document_reads_entire_file_bytes`

**建议改法**：大文件用 `ijson` / 分块；或对话 checkpoint 继续用现实现，**文档注明**「不适合 GB 级 JSON」。

**影响范围**：`checkpoint_json.py`；极大模型 JSON 加载峰值内存。

---

### 5.3 `train_step_batch`：`if text` 保留纯空白串

**来源**：`test_train_step_batch_preserves_whitespace_only_strings`

**建议改法**：`if text.strip()` 过滤；或文档定义为「调用方负责清洗」。

**影响范围**：`dialogue.py` `train_step_batch`；空白行不再产生空印记轮次。

---

### 5.4 `_clean_chunked_dialogue_text` 死分支（285 行）

**来源**：`test_dead_branch_after_alnum_gate`

**现象**：283 行已要求 `any(isalnum)`，285 行 `len<=4 and all(not alnum)` **逻辑不可达**。

**建议改法**：删除 285～286；或调整 283 门槛使 285 有意义。

**影响范围**：仅 `example_run.py` 语料清洗；**删除死代码无行为变化**。

---

### 5.5 审计用 trace 上下界（25～600、12～400）

**性质**：**回归哨兵**，不是缺陷。优化 3.1 后应 **下调** 上界/下界并写入注释，避免误报「死循环」。

---

### 5.6 已通过的正确性守护（无需当问题修）

| 测试 | 含义 |
|------|------|
| `test_epsilon_secondary_on_draft_leaves_frozen_resource_snapshot` | 草稿 trace 后资源快照恢复 **正确** |
| `EffectiveDialogueTrainTicksTests` | slow-final 调度 **符合设计** |
| `ResolveDialogueModelPathTests` / `test_invalid_json_raises` | 路径与 JSON 错误处理 **符合预期** |

---

## 6. 建议实施顺序（与影响面）

```text
阶段 1（小 diff、明确 bug）
  → 2.1 new_imprint_tokens +=                    ✅ 已完成
  → 2.2 manifest 路径沙箱                        ✅ 已完成
  → 5.4 删除死分支                               ✅ 已完成

阶段 2（对话正确性 + 中等性能）
  → 2.3 内心快照对齐 + 3.4 惊奇不重复算          ✅ 已完成
  → 3.2 EFE 内 rf 只算一次                         ✅ 已完成
  → 3.1 倾听 tokens 复用（最大体感）               ✅ 已完成
  → 3.3 _restore_resource 按 depth 收敛            ✅ 已完成

阶段 3（大词表 / 长跑）
  → 4.3 pop(0)→pop()                               ✅ 已完成
  → 4.2 tokens() 热路径改 _store 迭代              ✅ 已完成
  → 4.4 缓存命中返回 tuple                         ✅ 已完成
  → 4.5 合并不整表失效 torch cache                 ✅ 已完成
  → 4.6 lazy soft_freeze 节流                      ✅ 已完成

阶段 4（契约与工具链）
  → 2.4 严格加载 + patch_dialogue_model_dict       ✅ 已完成
  → 5.1 strict 透传                                ✅ 已完成
  → 5.3 train_step_batch strip 过滤                ✅ 已完成
  → 4.1 base 模型 sensory_error 去重               ✅ 已完成
```

---

## 7. 修改后的测试策略

| 改动类型 | 测试动作 | 状态 |
|----------|----------|------|
| 2.1 | 去掉 `expectedFailure`；保留 truth_new 断言 | ✅ |
| 2.2 | `test_parent_relative_path` 期望 jobs==0 | ✅ |
| 3.1 / 3.4 | 重标 trace / surprise **call_count** 基线 | ✅ |
| 3.2 / 3.3 | rf 单次、restore 有界 | ✅ |
| 2.3 | 正向断言内心快照一致 | ✅ |
| 2.4 | 严格 load + patch 测试 | ✅ |
| 4.3 | `PopCountingWordStateMemory` / 源码断言无 `pop(0)` | ✅ |

---

## 8. 模块影响矩阵（速查）

| 模块 | P0 | P1 | P2 | P3 |
|------|----|----|----|-----|
| `example_run.py` | ●（2.1/2.2 ✅） | ○ | ○ | ●● |
| `uva_model/dialogue.py` | ● | ●●● | ○ | ● |
| `uva_model/word_imprints.py` | ○ | ● | ●●● | ○ |
| `uva_model/tokenizer.py` | ○ | ● | ○ | ○ |
| `uva_model/model.py` | ○ | ○ | ● | ○ |
| `uva_model/corpus_jsonl.py` | ○ | ○ | ○ | ● |
| `uva_model/checkpoint_json.py` | ○ | ○ | ○ | ● |
| `dialogue_web_ui.py` | ●（2.4 严格加载已生效） | ●（单轮延迟） | ○ | ○ |

● = 可能需改；○ = 间接或仅文档。

---

*文档版本：P0–P3 审计项已落地（5.2 大 JSON 仍为文档建议）；全量 pytest 见 CI/本地最新结果。*
