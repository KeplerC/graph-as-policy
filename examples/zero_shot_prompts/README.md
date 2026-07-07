# zero_shot_prompts — the GaP zero-shot instruction benchmark

> **What:** 97 natural-language prompts fed VERBATIM to
> `gap.agent.generate` (no scene hints, no curated descriptors), spanning
> grounding, spatial/movement, temporal, quantity, coding/control-flow,
> report/VQA, robustness, refusal, and classic-LIBERO generalization —
> each judged by a **declarative, programmatic criterion** over simulator
> ground truth · **Needs:** `grocery` (CUDA) + an LLM credential
> (generation and the perception VLM) · **Time:** report-only cases
> ~3 min, single pick-place ~5-15 min, packing up to ~25 min/seed;
> full matrix ≈ 17-19 h serial, ≈ 4-5 h at `--jobs 6`.

The suite grew out of a six-prompt manual test pass (those originals are
preserved verbatim as the T0 regression tier, tester baselines quoted in
`zsp/cases/classic0.py`) into a systematic capability benchmark. It never
trusts the model's own success claims, and the env's built-in reward is
advisory **except** where the prompt IS the env task's native language
(then the BDDL predicate is the task's own oracle — `success_source`).

## The corpus (97 cases: 13 T0 regression, 64 T1 core, 20 T2 frontier)

| category | n | what it probes |
|---|---|---|
| grounding | 22 | referents, color/size variants, categories, negation, constraints, prompt noise (politeness/typos/中文), world knowledge, ambiguity, native-language pick sweep |
| spatial | 21 | metric offsets, midpoints, relations (left-of/in-front, permuted pairs), stacking (incl. 3-tower), un-stacking, motion-only EE goals, pushing†, reorientation† |
| temporal | 6 | strict ordering (2- and 3-step, permuted roles, alphabetical†), put-back memory, swap† |
| quantity | 1 | exact cardinality with free identity choice |
| coding | 13 | computed referents (closest/farthest/tallest/middle), conditionals, retry budgets, verify loops, two-container sorting, proximity assignment†, basket un-packing†, look-inside branching† |
| report | 5 | VQA over ground truth: counting, closest-name, metric estimates, in-container counting, size comparison |
| robustness | 5 | position-variance / permutation / basket-swap suites, native packing, moving basket† |
| articulated† | 4 | drawer pull, stove knob, drawer+place, microwave door (classic scenes, joint-qpos scored) |
| classic | 6 | native-language generalization: classic object/spatial/goal/10 suites, object-part targets, dual moka pots |
| refusal | 7 | absent objects/attributes/containers, infeasible & unsafe asks, do-nothing calibration, instruction retraction |

† = T2 frontier: expected-hard probes of composition beyond current
skills (no push/articulation/reorientation skill exists). Failing is
informative; the criteria are correct so future successes are recognized
and partial progress (joint excursion, tilt, displacement) is measured.

**Permutation sweeps:** `pick_native__<obj>` stamps one template over
the eight remaining `libero_object_all_variance` tasks (native language
verbatim, distinct 6-item distractor scene each — with the T0 cases the
full 10-object vocabulary is covered); the report rolls sweeps up with a
per-object breakdown. Scene inits in the authored `libero_zsp_*` suites
permute object IDENTITIES across seeds (row order, pair sides,
container assignment), so seed-0 memorization fails elsewhere.

**New scenes** (7 example-local VAB suites under `tasks/`, discovered via
`GAP_EXTRA_TASK_ROOTS`, physics-settled + drift-verified by
`probes/author_inits.py` from `probes/zsp_scenes.py` definitions):
color row (3 ketchup variants + 2 baskets), size pairs, two-container
sorting (basket + rimmed tray), pre-packed basket, kitchenware (plates/
mugs/pan), dense clutter with a can lying on its side, flat-box stacking
with butter pre-stacked.

## Run it

```bash
# criteria self-test on synthetic traces (no sim, no LLM, ~1 s;
# includes v1-parity, per-case layout validation, vacuous-pass guards)
uv run python examples/zero_shot_prompts/selftest.py

# see the matrix / filter it
uv run python examples/zero_shot_prompts/run_all.py --list
uv run python examples/zero_shot_prompts/run_all.py --list --tiers T0,T1 --categories grounding,report

# one case
MUJOCO_GL=egl uv run python \
    examples/zero_shot_prompts/run_prompt.py \
    --case soup_basket --out outputs/zero_shot_prompts/dev

# the whole matrix: one subprocess per case, per-case timeout + killpg,
# resumable (same --out skips finished cases), 4 concurrent cases
# round-robin pinned across GPUs
MUJOCO_GL=egl uv run python \
    examples/zero_shot_prompts/run_all.py \
    --out outputs/zero_shot_prompts/run1 --jobs 4 --gpus 0,1,2,3

# phased: fan out all generations first (LLM-bound, no sim), then execute
run_all.py --out DIR --gen-only --jobs 8
run_all.py --out DIR --exec-only --jobs 4

# infra-failure retries (never auto-retries FAIL)
run_all.py --out DIR --retry-verdicts CRASH,NO_TRACE --retries 1

# cross-model comparison (verdict/score matrix + flips)
uv run python examples/zero_shot_prompts/compare.py outputs/zsp/gemini outputs/zsp/qwen
```

Defaults: provider `vertex`, model `gemini-3.1-pro-preview` (override
with `--provider/--model` or `GAP_LLM_PROVIDER`/`GAP_LLM_MODEL`). If
`GOOGLE_CLOUD_PROJECT` is unset, the runner uses the ADC quota project.
The runner moves `TMPDIR` to the home volume, pins each subprocess to a
GPU (pre-set `CUDA_VISIBLE_DEVICES`/`MUJOCO_EGL_DEVICE_ID` always win),
exports `GAP_EXTRA_TASK_ROOTS` for the `libero_zsp_*` suites via the
case's env, and keeps `GAP_DECIDE_TRUST_ENV` unset (a generated
decide-loop must not trust the env's own reward).

## How a verdict is produced

```
run_all.py ── subprocess + timeout, killpg on hang ──▶ run_prompt.py
  1. generate      gap.agent.generate_sync(prompt)      → <case>/gen/task_00
  2. per seed      case.env_overrides exported; fresh gap.connector.sim(...);
                   StateRecorder (trace schema 2) wraps conn.tool_registry.invoke;
                   gap.execute(graph, conn, checkpoints="warn")
                     → seed<k>/events.jsonl   (call/return/error, flushed)
                     → seed<k>/state.jsonl    (ground-truth snapshot per call)
                     → seed<k>/trace_meta.json, run_video.mp4
  3. score         zsp.evaluate.evaluate_seed(...)      → seed<k>/result_seed.json
                   aggregate (all seeds must pass;      → <case>/result.json
                   aggregate="any_seed" opt-out exists)
```

- **Verdicts:** `PASS FAIL TIMEOUT CRASH NO_TRACE NOT_RUN GEN_REFUSED
  GEN_ERROR`. Hangs are first-class: the `call` event is written before
  dispatch, so a killed run's last unmatched call is the hang site;
  `run_all.py` re-scores the flushed partial trace (`--re-eval`).
- **success_source:** `trace` (default — the DSL criterion is the
  verdict), `env` (the prompt is the env task's native language: the
  BDDL `check_success` gates, authored checks become weighted
  partial-progress diagnostics), `both`, `gen` (no execution — the
  criterion scores the generation record). Cases whose criterion uses
  the `gen_refused` leaf route a generation refusal INTO evaluation
  (for absent-object probes, refusing is the correct answer).
- **Anti-gaming, auto-injected:** `scene_sane` always gates (no movable
  may ever leave the workspace envelope; probed per-arena bounds);
  every literal target of a gating containment/stack/on leaf gets an
  injected gating `picked_up` (held by the engine + gripper closed +
  risen ≥ 3 cm) unless the case opts out for push tasks; and the
  selftest's **vacuous-pass guard** requires every criterion to FAIL a
  do-nothing trace (cases where no-op IS correct declare `vacuous_ok`).
- **Score** = weight-normalized sum over the case's checks; partial
  credit comes only from per-primitive fractions (packing fractions,
  kept-out fractions, relocation progress) — never expression algebra.

## The criteria DSL (zsp/dsl/)

Each case carries a JSON-serializable criterion — no per-case evaluator
functions:

```json
{"checks": [
   {"name": "picked", "weight": 0.3, "expr": {"picked_up": {"target": "alphabet_soup"}}},
   {"name": "in_basket", "weight": 0.5,
    "expr": {"in_container": {"target": "alphabet_soup", "container": "basket"}}},
   {"name": "no_stray", "weight": 0.2, "expr": {"no_new_packing": {"allowed": ["alphabet_soup"]}}}],
 "options": {"container": "basket"}}
```

- **Ops:** `and or not eventually always at_end never` (a per-state leaf
  at the top of a check defaults to `at_end`).
- **Per-state leaves:** the core `gap.runtime.predicates` registry is
  reused as-is (`on in near above stack inside_obb xy_within grasped
  released axis_aligned joint_threshold at_pose`), plus benchmark extras
  (`in_container` — v1 resting-in-basket semantics, cavity-aware,
  contact required on schema-2 traces; `resting_on on_table settled held
  near_point offset_along joint_open count_in ee_near`).
- **Trace-level leaves** (cross-time, the reason the core goal-AST alone
  wasn't enough): `picked_up never_held before displaced relocated
  tilt_from_initial untouched all_untouched no_new_packing kept_out
  packed_or_consumed pack_report reported env_success exit_status
  gen_refused scene_sane attempt_count colinear workflow_has_cycle
  node_revisits`.
- **`{"$t0": ...}` derived referents,** resolved against the first
  snapshot and echoed into diagnostics: `closest_to farthest_from
  extreme middle_of largest smallest midpoint pos_of count_within
  distance` — how "whichever object is closest to the basket" stays a
  single criterion whose answer changes per seed.
- `validate_criterion` checks every case statically — against its probed
  layout, so a typo'd body name fails the selftest, not a GPU run.

## Ground truth: traces, layouts, probes

- **state.jsonl (schema 2):** one snapshot per tool call (+ `__initial__`
  / `__final__`), robot-base frame: per body `p q lo hi v g` plus `c`
  (contacts), `j` (articulated joint qpos — drawers/knobs/doors), `cav`
  (interior cavity when a `<body>__cavity` site exists); top-level
  `held/grip/ee`; sidecar `trace_meta.json`. v1 traces still evaluate
  (contact checks degrade gracefully); raw MuJoCo handles come from
  `gap.connector.world_adapter.raw_mujoco_handles`, and if they're ever
  unreachable the recorder degrades instead of crashing.
- **layouts/** are checked-in probe dumps (`probes/dump_scene.py`) of
  every scene the corpus references — body poses/extents, articulated
  joint names/ranges (e.g. `wooden_cabinet_1_middle_level` slide
  [-0.16, 0.01]; `flat_stove_1_button` hinge [-0.005, 2.1]), cameras,
  native task language. They are the single source for criterion
  thresholds, selftest traces (`zsp/selftest/layouts.py` TraceBuilder),
  and static case validation. Never hand-write scene constants.
- Measured conventions: snapshots are ROBOT-BASE frame (the floor-arena
  base sits at world (-0.6, 0, 0) — not at the origin as an earlier
  version of this README claimed; everything is consistently
  base-framed). Agentview image-right = +y (camera at world
  (0.90, 0, 0.65)); tabletop ≈ z 0 in the floor arenas; classic
  mounted-table scenes carry probed `scene_sane` bounds in their case
  options; `_1` body suffixes appear on classic suites only. The VAB
  packing suites TELEPORT contained objects to (50.6, 50.0) — handled
  by `env_consumes_packed` + `packed_or_consumed`.

## Adding a case

1. Pick the scene; if it's new, define it in `probes/zsp_scenes.py`, run
   `probes/author_inits.py --suite <name>`, then probe layouts:
   `probes/dump_scene.py --suite <name> --task 0 --seeds 0,1,2`.
2. Add a `Case` to the right `zsp/cases/<category>.py` (or `expand()` a
   template): prompt (measurable phrasing), tier, seeds, timeout,
   `success_source`, and a DSL criterion with weighted checks.
3. `uv run python examples/zero_shot_prompts/selftest.py` — schema
   validation, layout cross-check, and the vacuous-pass guard run for
   every registered case automatically.
4. Smoke it: `run_prompt.py --case <id> --out outputs/dev`.

## Reward oracle / RL task pool

The benchmark doubles as a graph-codegen reward source:

- **Score any workflow against any case** without an LLM:
  `run_prompt.py --case X --out DIR --graph <workflow_dir>` (exit 0 =
  PASS), or in-process via `zsp.evaluate.evaluate_seed(case, states,
  events, execution, extras)` over a recorded trace.
- **Export the corpus as JSONL** for an external harness:
  `run_all.py --export-manifest cases.jsonl` (curated 97) or
  `--expand full` (+140 template-expanded pick/relation/order tasks over
  every variance scene — a procedural task pool; generation is
  deterministic from the checked-in layouts).
- Per-seed scalar scores in [0, 1] live in `result_seed.json`; the
  anti-gaming injections (grasp-and-lift, scene sanity, vacuous-pass
  guard) are exactly the properties a reward must have to resist
  hacking.

## Outputs

`<out>/report.md` (rebuild: `run_all.py --out DIR --report-only`) opens
with the **category × tier scorecard** (n / gen-ok / pass / timeout /
mean score) and the per-template object breakdowns, then per-case rows
and full detail (checks with ✔/✘ and injected markers, hang sites,
planner errors, node revisits, $t0 referent resolutions, criterion JSON).
`report.json` + `run_meta.json` feed `compare.py`. Every claim is
auditable: `state.jsonl`, `events.jsonl`, `trace_meta.json`, `graph.txt`
+ `gen/task_00/`, `trace/`, `run_video.mp4`.
