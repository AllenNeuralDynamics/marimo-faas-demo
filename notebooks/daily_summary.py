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
"""VR-foraging daily session summary.

The cells only wire things together. Layout lives in ui.py, data in vr_data.py, figures in vr_plots.py.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="VR foraging - daily summary")

with app.setup:
    import boot  # progress for the static site's loading overlay; no-op elsewhere

    boot.report("libraries")
    import marimo as mo

    # The static export bundles our local modules, but not their dependencies:
    # the browser installs only what this file imports, so they are named here.
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import pyarrow  # noqa: F401
    import ui


@app.cell(hide_code=True)
def header(session):
    ui.header(session)
    return


@app.cell(hide_code=True)
async def pick_subject_and_day():
    # ?session=<id> in the URL (or --session) picks the session; otherwise the newest one.
    query = mo.query_params()
    requested = query.get("session") or mo.cli_args().get("session")
    subject, day = await ui.session_inputs(requested)
    return day, query, requested, subject


@app.cell(hide_code=True)
async def pick_session(day, query, requested, subject):
    picker, sessions = await ui.session_picker(subject, day, requested, query)
    return picker, sessions


@app.cell(hide_code=True)
def controls(day, picker, subject):
    ui.controls(subject, day, picker)
    return


@app.cell(hide_code=True)
async def load(picker, sessions):
    session = await ui.load(picker, sessions)
    return (session,)


@app.cell(hide_code=True)
def summary(session):
    ui.summary(session)
    return


@app.cell(hide_code=True)
def plots(session):
    import matplotlib  # noqa: F401 - named so the browser build installs it; ui.plots imports it

    ui.plots(session)
    return


@app.cell(hide_code=True)
def timeline_controls(session):
    start = ui.timeline_slider(session)
    ui.timeline_heading(start)
    return (start,)


@app.cell(hide_code=True)
def timeline(session, start):
    ui.timeline(session, start)
    return


if __name__ == "__main__":
    app.run()
