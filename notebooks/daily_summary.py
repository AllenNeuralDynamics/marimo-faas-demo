# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25.2",
#     "matplotlib>=3.10",
#     "numpy>=2",
#     "pandas>=2.2",
#     "pyarrow>=18",
# ]
# ///
"""VR-foraging daily session summary. All logic lives in ``vr_summary.py``, next to this file."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="VR foraging - daily summary")

with app.setup:
    import marimo as mo

    # The static export bundles vr_summary.py (next to this file) as a wheel, but not its
    # dependencies: Pyodide installs only what this file imports, so they are named here.
    import matplotlib  # noqa: F401
    import numpy  # noqa: F401
    import pandas
    import pyarrow  # noqa: F401
    import vr_summary as vs


@app.cell(hide_code=True)
def header():
    mo.md("""
    # VR foraging: daily session summary
    Pick a subject and a day, or open this page with `?session=<session_id>`.
    """)
    return


@app.cell(hide_code=True)
def export_pdf(figures, per_patch, session_id, stats):
    mo.download(
        data=lambda: vs.pdf_report(session_id, stats, per_patch, list(figures.values())),
        filename=f"{session_id}.pdf",
        mimetype="application/pdf",
        label="Export as PDF",
    ).center().style({"transform": "scale(1.6)", "margin": "1.25rem 0"})  # big: shadow DOM ignores font-size
    return


@app.cell
async def pick_session():
    # ?session= (or --session) wins; otherwise the newest session in the metadata service.
    query = mo.query_params()
    requested = query.get("session") or mo.cli_args().get("session")
    focus = await vs.find_raw_session(requested)
    mo.stop(focus is None, mo.callout(f"No VR-foraging session named {requested!r}.", kind="danger"))
    started = pandas.Timestamp(focus["acquisition"]["acquisition_start_time"])
    subject_input = mo.ui.text(value=focus["subject"]["subject_id"], label="Subject")
    day_input = mo.ui.date(value=started.date(), label="Day (UTC)")
    mo.hstack([subject_input, day_input], justify="start")
    return day_input, query, requested, subject_input


@app.cell
async def session_choice(day_input, query, requested, subject_input):
    sessions = await vs.sessions_on_day(subject_input.value, day_input.value)
    ready = sessions.dropna(subset=["processed_location"])
    mo.stop(ready.empty, mo.callout("No processed sessions for this subject on this day.", kind="warn"))
    assets = dict(zip(ready["session_id"], ready["processed_location"], strict=True))
    session_picker = mo.ui.dropdown(
        options=list(assets),
        value=requested if requested in assets else next(iter(assets)),
        label="Session",
        on_change=lambda sid: query.set("session", sid),  # keeps the URL shareable
    )
    session_picker
    return assets, session_picker


@app.cell
async def load(assets, session_picker):
    session_id = session_picker.value
    with mo.status.spinner(title=f"Loading {session_id}"):
        sites, position, threshold = await vs.load_session(assets[session_id])
    reward_sites = vs.prepare_reward_sites(sites)
    colors = vs.patch_colors(reward_sites["patch_label"])
    stats = vs.session_summary(reward_sites, position)
    per_patch = vs.per_patch_table(reward_sites)
    figures = {
        "raster": vs.plot_outcome_raster(reward_sites),
        "velocity": vs.plot_aligned_velocity(reward_sites, position, colors, threshold),
        "stop_rate": vs.plot_stop_rate(reward_sites, colors),
    }
    return figures, per_patch, session_id, stats


@app.cell(hide_code=True)
def summary(per_patch, session_id, stats):
    mo.vstack(
        [
            mo.md(f"## {session_id}"),
            mo.hstack(
                [
                    mo.stat(stats["odor_sites"], label="Odor sites visited"),
                    mo.stat(f"{stats['stop_rate']:.0%}", label="Stopped", caption=f"{stats['stops']} stops"),
                    mo.stat(
                        f"{stats['reward_rate']:.0%}", label="Stops rewarded", caption=f"{stats['rewards']} rewards"
                    ),
                    mo.stat(f"{stats['water_ul']:.0f} µL", label="Water collected"),
                    mo.stat(f"{stats['distance_m']:.0f} m", label="Distance run"),
                    mo.stat(f"{stats['duration_min']:.0f} min", label="Duration"),
                ],
                widths="equal",
            ),
            mo.ui.table(per_patch, selection=None, label="Per patch type"),
        ]
    )
    return


@app.cell(hide_code=True)
def plots(figures):
    mo.vstack(list(figures.values()))
    return


if __name__ == "__main__":
    app.run()
