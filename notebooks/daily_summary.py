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
"""VR-foraging daily session summary. Logic lives next to this file: vr_data.py (data) and vr_plots.py (figures)."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="VR foraging - daily summary")

with app.setup:
    import asyncio

    import boot  # progress for the static site's loading overlay; no-op elsewhere

    boot.report("libraries")
    import marimo as mo

    # The static export bundles our local modules as wheels, but not their dependencies:
    # Pyodide installs only what this file imports, so they are named here.
    import numpy  # noqa: F401
    import pandas
    import pyarrow  # noqa: F401
    import vr_data as vd


@app.cell(hide_code=True)
def header():
    mo.md("""
    # VR foraging: daily session summary
    Pick a subject and a day, or open this page with `?session=<session_id>`.
    """)
    return


@app.cell(hide_code=True)
def export_pdf(figures, per_patch, session_id, stats, vp):
    mo.download(
        data=lambda: vp.pdf_report(session_id, stats, per_patch, list(figures.values())),
        filename=f"{session_id}.pdf",
        mimetype="application/pdf",
        label="Export as PDF",
    ).center().style({"transform": "scale(1.6)", "margin": "1.25rem 0"})  # big: shadow DOM ignores font-size
    return


@app.cell
async def pick_session():
    # ?session= (or --session) wins; otherwise the newest session in the metadata service.
    boot.report("session")
    query = mo.query_params()
    requested = query.get("session") or mo.cli_args().get("session")
    focus = await vd.find_raw_session(requested)
    if focus is None:
        boot.report("done")
    mo.stop(focus is None, mo.callout(f"No VR-foraging session named {requested!r}.", kind="danger"))
    started = pandas.Timestamp(focus["acquisition"]["acquisition_start_time"])
    subject_input = mo.ui.text(value=focus["subject"]["subject_id"], label="Subject")
    day_input = mo.ui.date(value=started.date(), label="Day (UTC)")
    mo.hstack([subject_input, day_input], justify="start")
    return day_input, query, requested, subject_input


@app.cell
async def session_choice(day_input, query, requested, subject_input):
    sessions = await vd.sessions_on_day(subject_input.value, day_input.value)
    ready = sessions.dropna(subset=["processed_location"])
    if ready.empty:
        boot.report("done")
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
    boot.report("download")
    with mo.status.spinner(title=f"Loading {session_id}"):
        download = asyncio.ensure_future(vd.load_session(assets[session_id]))
        await asyncio.sleep(0)  # let the requests go out before the import below blocks Python
        import matplotlib.pyplot  # noqa: F401 - ~3 s in the browser, overlapped with the download
        import vr_plots as vp

        sites, position, licks, threshold = await download
    boot.report("processing")
    reward_sites = vd.prepare_reward_sites(sites)
    colors = vp.patch_colors(reward_sites["patch_label"])
    stats = vd.session_summary(reward_sites, position)
    per_patch = vd.per_patch_table(reward_sites)
    figures = {
        "raster": vp.plot_outcome_raster(reward_sites),
        "velocity": vp.plot_aligned_velocity(reward_sites, position, colors, threshold),
        "stop_rate": vp.plot_stop_rate(reward_sites, colors),
        "leave": vp.plot_leave_heatmap(reward_sites),
    }
    boot.report("done")
    return colors, figures, licks, per_patch, position, session_id, sites, stats, vp


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


@app.cell
def timeline_start(stats):
    start = mo.ui.slider(
        0, stats["duration_min"] * 60, step=10, label="Timeline start (s)", show_value=True, full_width=True
    )
    mo.vstack([mo.md("## Session timeline"), start])
    return (start,)


@app.cell(hide_code=True)
def timeline(colors, licks, position, sites, start, vp):
    vp.plot_timeline(sites, position, licks, colors, start.value)
    return


if __name__ == "__main__":
    app.run()
