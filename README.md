# VR foraging daily summary (marimo, client-side)

A rewrite of the lab's "summary of today's training" notebook
([`SummaryTodayTraining.ipynb`](https://github.com/AllenNeuralDynamics/Aind.Behavior.VrForaging.Analysis/blob/main/notebooks/BehaviorAnalysis/General/SummaryTodayTraining.ipynb))
as a single marimo notebook, [`notebooks/daily_summary.py`](notebooks/daily_summary.py), that reads the
[packaged](https://github.com/AllenNeuralDynamics/Aind.Behavior.VrForaging.Packaging) per-session tables
straight from public S3.

It shows the same outputs as the original: headline stats, water per patch type, the patch × site choice
raster, velocity aligned to odor onset, P(stop) over the session, the P(leave) heatmap and a scrollable
session timeline.

## Three ways to run it

| | Command / URL | Python runs |
| --- | --- | --- |
| 1. Local app in the browser | `uv run marimo run notebooks/daily_summary.py` (or `edit`) | on your machine |
| 2. A session from the URL | append `?session=867425_2026-11-13_21-11-07` to either URL | either |
| 3. Static site on GitHub Pages | `https://<owner>.github.io/<repo>/?session=<session_id>` | in the visitor's browser (Pyodide/WASM) |

Also `uv run python notebooks/daily_summary.py --session <id>` runs it headless as a script.

Build the static site locally exactly as CI does:

```bash
uv run marimo export html-wasm notebooks/daily_summary.py --mode run --no-sandbox -o _site
uv run python -m http.server --directory _site   # WASM needs http://, not file://
```

## How the data is found (no server, no index, no credentials)

```text
?session=<subject>_<YYYY-MM-DD>_<HH-MM-SS>
   │
   ├─ GET https://aind-open-data.s3.us-west-2.amazonaws.com/?list-type=2&prefix=<subject>_<date>_&delimiter=/
   │     → every session of that subject that day; keep the newest "<session_id>_processed_<ts>/" per session
   │
   └─ GET <asset>/sites.parquet, position_velocity.parquet, licks.parquet, software_events.parquet
```

Both buckets allow anonymous, CORS-enabled `GET`s, so the browser can read them directly. The aggregated
`vr-foraging-dataset/sites.parquet` is ~470 MB and is not used: a session's own asset is ~13 MB in total.
Newly processed sessions show up immediately, since nothing has to be rebuilt.

> The packaging docs say to look up `source_s3_location` in `session.parquet` rather than build the path.
> This notebook lists the bucket instead, because `session.parquet` is 32 MB of mostly config JSON, too much
> for a browser tab. If processed assets ever move out of `aind-open-data`, `list_processed_sessions` is the
> one function to change.

## Layout

```text
notebooks/daily_summary.py   the notebook. Logic lives in top-level @app.function's (importable, testable);
                             cells only wire UI → load → figure.
tests/test_daily_summary.py  pytest on those functions with hand-checkable synthetic tables
.github/workflows/pages.yml  ruff + marimo check + pytest on every PR; WASM export → Pages on main
```

The logic stays in the notebook file, not a `src/` package as in `analysis-harlow-learning-sets`, on purpose.
A WASM build can only import what Pyodide can install, and a local package would need a wheel shipped next
to the page. Because these are `@app.function`s, `from daily_summary import classify_outcomes` still works
for tests and other notebooks. Once the logic is shared across notebooks, move it to a package and
`micropip.install` its wheel.

## What changed relative to the original notebook

| Original | Here |
| --- | --- |
| Data from `Z:\scratch\...` share via `find_sessions_relative_to_date` + raw-file parsing | Packaged parquet tables over HTTPS; works from any machine with a browser |
| Hard-coded `mouse = '863696'`, `date_string = ...`, edit-and-rerun | `?session=` URL, subject/date pickers, URL updates when you pick a session |
| `%autoreload`, hidden state, cells that only work in a certain order | marimo DAG: changing an input re-runs exactly the dependent cells |
| `warnings.simplefilter("ignore")`, `pd.options.mode.chained_assignment = None` | No silenced warnings; no in-place mutation of shared frames |
| `try: v_thr = data[...]["velocityThreshold"] except: v_thr = 8` | Threshold read from the logged `StopVelocityThreshold` event; not drawn if absent |
| `plt.subplot()` called inside `iterrows()` for every site; `print(trial_dict)` per trial | Vectorized: one `bar` call per panel, one `np.interp` for all aligned traces |
| Outcome colors via a 30-line per-row `if/elif` with patch-name special cases (`'D'`, `'S'`) | One `np.select` over table columns; adds the "aborted wait" outcome the old code merged in |
| `color_dict_label[odor_label]` → `KeyError` for any new patch name | Colors assigned to sorted patch labels in a fixed, CVD-validated order |
| Hidden helper defined twice (`update_plot`), dead commented code, `savefig` to a personal OneDrive path | One definition per function, no file side effects |
| No tests | `pytest` on every pure function, run in CI with `ruff` and `marimo check --strict` |

## Notes

- The `Site` model documents `reward_amount` in milliliters, but the stored values are microliters (5.0 per
  reward). The notebook labels it µL.
- "Leave, depleted" from the old raster is gone. The sites table only records `reward_available` at sites
  where the animal stopped, so a skipped site's depletion is unknown from this table alone.
