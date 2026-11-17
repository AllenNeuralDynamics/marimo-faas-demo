# VR foraging daily summary (marimo, client-side)

A rewrite of the lab's "summary of today's training" notebook
([`SummaryTodayTraining.ipynb`](https://github.com/AllenNeuralDynamics/Aind.Behavior.VrForaging.Analysis/blob/main/notebooks/BehaviorAnalysis/General/SummaryTodayTraining.ipynb))
as a small marimo notebook that reads the
[packaged](https://github.com/AllenNeuralDynamics/Aind.Behavior.VrForaging.Packaging) per-session tables
straight from public S3.

It shows headline stats, water per patch type, the patch × site choice raster, velocity aligned to odor
onset, P(stop) over the session, the P(leave) heatmap and a scrollable session timeline. An **Export as
PDF** button at the top downloads a 5-page report, built in Python on click.

## Three ways to run it

| | Command / URL | Python runs |
| --- | --- | --- |
| 1. Local app in the browser | `uv run marimo run notebooks/daily_summary.py` (or `edit`) | on your machine |
| 2. A session from the URL | append `?session=<session_id>` to either URL | either |
| 3. Static site on GitHub Pages | [`allenneuraldynamics.github.io/marimo-faas-demo/?session=<session_id>`](https://allenneuraldynamics.github.io/marimo-faas-demo/) | in the visitor's browser (Pyodide/WASM) |

Also `uv run python notebooks/daily_summary.py --session <id>` runs it headless as a script.

Build the static site locally exactly as CI does:

```bash
uv run marimo export html-wasm notebooks/daily_summary.py --mode run --no-sandbox -o _site
uv run python scripts/inject_boot_overlay.py _site/index.html   # loading progress overlay
uv run python -m http.server --directory _site   # WASM needs http://, not file://
```

## How the data is found (no server, no credentials)

```text
?session=<session_id>                     (or subject + day pickers; or nothing = newest session)
   │
   ├─ DocDB  https://api.allenneuraldynamics.org/v2/metadata_index/data_assets
   │    1. the session:  {name, data_level: "raw"}            → its subject and day (record fields)
   │    2. that day:     {subject_id, acquisition_start_time in day, acquisition_type}
   │                     → raw and derived records together; a derived record's source_data
   │                       names its raw session; newest by creation_time wins → `location`
   │
   └─ S3     <location>/sites.parquet, position_velocity.parquet, licks.parquet, software_events.parquet
```

`session_id` is the raw record's `name`, and it is treated as an opaque key. It is only compared for
equality in DocDB queries, never parsed or used to build a path. Subject and date come from the record's
own fields, and the asset path is the derived record's `location`.

This is the same REST call `aind-data-access-api`'s `MetadataDbClient` makes, issued directly because the
client's `requests` session doesn't run in Pyodide. The API and both buckets send `Access-Control-Allow-Origin: *`,
so the browser reads them directly. Query 2 deliberately doesn't filter on `source_data` alone: that field
isn't indexed and takes ~6 s, against ~0.1 s for subject + day.

The day picker matches `acquisition_start_time` in UTC: a session started after 4 pm Pacific is listed
under the next day.

## Layout

```text
notebooks/daily_summary.py      the notebook: pick a session → load → show. No logic.
notebooks/vr_data.py            DocDB + S3 access and analysis (no matplotlib)
notebooks/vr_plots.py           figures and the PDF report
notebooks/boot.py               progress steps reported to the loading overlay (browser only)
scripts/boot_overlay.html       the overlay; injected into the export by scripts/inject_boot_overlay.py
tests/test_summary.py           pytest on vr_data/vr_plots with hand-checkable synthetic tables
.github/workflows/pages.yml     ruff + marimo check + pytest on every PR; WASM export → Pages on main
```

`marimo export html-wasm` packs local modules the notebook imports into wheels under `public/wheels/` and
installs them in the browser, so they need no build step. The wheels declare no dependencies, though:
Pyodide installs only packages the notebook file imports, which is why the notebook imports `numpy`,
`pandas`, `pyarrow` and `matplotlib` itself.

## Loading, speed and caching

In the static build, a progress overlay paints before Python exists. It shows: starting Python →
installing packages → loading libraries → finding the session → downloading data → processing and
plotting. The first two steps are read from marimo's own loading text; the rest are posted by
`boot.report()` over a `BroadcastChannel`. After 45 s without progress it offers to show the page anyway.

Measured in Chrome against a local server:

| | cold (empty browser cache) | warm (reload) |
| --- | --- | --- |
| Python + packages | ~18 s | ~15 s |
| loading libraries (pandas 3.8 s, numpy, pyarrow) | ~4.5 s | ~4.7 s |
| session lookup (2 DocDB queries) | ~0.5 s | ~0.4 s |
| data download, overlapped with the ~3 s matplotlib import | ~6.5 s | ~3.5 s |
| processing + plotting | ~2 s | ~2 s |
| **total** | **~33 s** | **~26 s** |

- **Overlap:** matplotlib is imported only in the load cell, after the downloads are issued. Python under
  Pyodide blocks the worker's event loop, but requests already sent keep downloading.
- **Caching:** processed assets never change in place, so their parquet files are fetched with
  `cache: "force-cache"`, and the browser reuses its HTTP-cached copy across reloads. Loaded sessions
  are also kept in memory, so switching back to one is instant. DocDB answers are never cached.
- **What's left:** almost all of it is Pyodide starting and loading packages, the same for a notebook that
  does nothing. The next win would be dropping pandas (~4 s), which would mean rewriting the analysis on
  pyarrow + numpy.

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
