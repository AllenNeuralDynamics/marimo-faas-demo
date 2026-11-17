"""Page layout for the daily summary notebook: each function is one cell's worth of UI.

Kept out of the notebook so its cells stay one line each. Progress for the static site's
loading overlay is reported from here too (see boot.py).
"""

import boot
import marimo as mo
import pandas as pd
import provenance
import vr_data as vd

BORDER = "1px solid #e1e0d9"


def _caption(text: str) -> mo.Html:
    return mo.md(f"<span style='color:#6b6a66;font-size:0.9rem'>{text}</span>")


# ---------------------------------------------------------------------------
# Picking a session
# ---------------------------------------------------------------------------


async def session_inputs(requested: str | None):
    """Subject and day inputs, defaulting to the requested session (or the newest one)."""
    boot.report("session")
    focus = await vd.find_raw_session(requested)
    if focus is None:
        boot.report("done")
    mo.stop(focus is None, mo.callout(f"No VR-foraging session named {requested!r}.", kind="danger"))
    started = pd.Timestamp(focus["acquisition"]["acquisition_start_time"])
    subject = mo.ui.text(value=focus["subject"]["subject_id"], label="Subject")
    day = mo.ui.date(value=started.date(), label="Day (UTC)")
    return subject, day


async def session_picker(subject, day, requested: str | None, query):
    """Dropdown of the processed sessions that day; picking one updates the page's URL."""
    sessions = await vd.sessions_on_day(subject.value, day.value)
    ready = sessions.dropna(subset=["processed_location"])
    if ready.empty:
        boot.report("done")
    mo.stop(ready.empty, mo.callout("No processed sessions for this subject on this day.", kind="warn"))
    labels = {f"{row.start_time:%H:%M} UTC · {row.session_id}": row.session_id for row in ready.itertuples()}
    by_id = {sid: label for label, sid in labels.items()}
    picker = mo.ui.dropdown(
        options=labels,
        value=by_id.get(requested, next(iter(labels))),
        label="Session",
        on_change=lambda sid: query.set("session", sid),
    )
    return picker, ready


def controls(subject, day, picker) -> mo.Html:
    return mo.hstack([subject, day, picker], justify="start", gap=1.5, wrap=True).style(
        {"border": BORDER, "border-radius": "10px", "padding": "0.6rem 1rem"}
    )


async def load(picker, sessions) -> vd.Session:
    session_id = picker.value
    boot.report("download")
    with mo.status.spinner(title=f"Loading {session_id}"):
        row = sessions.loc[sessions["session_id"] == session_id].iloc[0]
        boot.report("processing")
        session = await vd.open_session(row)
    boot.report("done")  # the header and stats can show now; plots follow
    return session


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


def _pdf_button(session: vd.Session) -> mo.Html:
    def build() -> bytes:  # on click only: figures are redrawn for the PDF
        import vr_plots as vp

        figures = vp.session_figures(session)
        return vp.pdf_report(
            session.session_id, session.stats, session.per_patch, list(figures.values()), provenance.label()
        )

    button = mo.download(
        data=build, filename=f"{session.session_id}.pdf", mimetype="application/pdf", label="Export as PDF"
    )
    return button.style({"transform": "scale(1.35)", "transform-origin": "right center"})


def _provenance() -> mo.Html:
    url = provenance.commit_url()
    text = provenance.label()
    return _caption(f"[{text}]({url})" if url else text)


def header(session: vd.Session) -> mo.Html:
    left = mo.vstack(
        [
            mo.md("# VR foraging daily summary"),
            mo.md(
                f"**{session.session_id}** · subject {session.subject_id} · {session.start_time:%a %d %b %Y, %H:%M} UTC"
            ),
            _provenance(),
        ],
        gap=0.2,
    )
    return mo.hstack([left, _pdf_button(session)], justify="space-between", align="center")


def summary(session: vd.Session) -> mo.Html:
    s = session.stats
    per_reward = s["water_ul"] / s["rewards"] if s["rewards"] else 0
    per_minute = s["distance_m"] / s["duration_min"] if s["duration_min"] else 0
    cards = [  # (value, label, caption): every card has a caption, so they line up at the same height
        (s["odor_sites"], "Odor sites visited", f"in {session.per_patch['patches'].sum()} patches"),
        (f"{s['stop_rate']:.0%}", "Stopped", f"{s['stops']} stops"),
        (f"{s['reward_rate']:.0%}", "Stops rewarded", f"{s['rewards']} rewards"),
        (f"{s['water_ul']:.0f} µL", "Water collected", f"{per_reward:.0f} µL per reward"),
        (f"{s['distance_m']:.0f} m", "Distance run", f"{per_minute:.0f} m/min"),
        (f"{s['duration_min']:.0f} min", "Duration", f"from {session.start_time:%H:%M} UTC"),
    ]
    cards = [mo.stat(value, label=label, caption=caption, bordered=True) for value, label, caption in cards]
    return mo.hstack(cards, widths="equal", gap=0.75)


def _patch_table(per_patch: pd.DataFrame) -> mo.Html:
    rows = [
        f"| {r.patch_label} | {r.patches} | {r.sites} | {r.stop_rate:.0%} | {r.rewards} | {r.water_ul:.0f} |"
        for r in per_patch.itertuples()
    ]
    head = "| Patch type | Patches | Odor sites | Stopped | Rewards | Water (µL) |\n|---|--:|--:|--:|--:|--:|"
    return mo.md("\n".join([head, *rows]))


def _section(title: str, how_to_read: str, body) -> mo.Html:
    return mo.vstack([mo.md(f"### {title}"), _caption(how_to_read), body], gap=0.4)


def plots(session: vd.Session) -> mo.Html:
    """All figures, in sections. Imports matplotlib here, after the stats are on screen."""
    mo.output.replace(_caption("Loading the plotting library (matplotlib)…"))
    import vr_plots as vp

    mo.output.replace(_caption("Drawing plots…"))
    figures = vp.session_figures(session)
    return mo.vstack(
        [
            _section(
                "Choices, patch by patch",
                "One column per patch, one cell per odor site, colored by what the animal did there.",
                figures["raster"],
            ),
            _section(
                "Velocity at odor onset",
                "First site of each patch. Line = median, band = interquartile range.",
                figures["velocity"],
            ),
            _section(
                "Patch types and leaving",
                "Left: totals per patch type. Right: how often the animal skipped a site, "
                "given the rewards and consecutive failures it had just had in the patch.",
                mo.hstack([_patch_table(session.per_patch), figures["leave"]], widths=[1, 1], align="center", gap=2),
            ),
            _section(
                "Engagement over the session",
                "Probability of stopping at an odor site, smoothed over 15 sites, per patch type.",
                figures["stop_rate"],
            ),
        ],
        gap=2,
    )


def timeline_slider(session: vd.Session):
    """Start of the 60 s timeline window; sends its value on release, so the plot redraws once per drag."""
    return mo.ui.slider(
        0,
        session.stats["duration_min"] * 60,
        step=10,
        label="Start (s)",
        show_value=True,
        full_width=True,
        debounce=True,
    )


def timeline_heading(slider) -> mo.Html:
    return _section("Session timeline", "60 s at a time: licks, choice cues and rewards above velocity.", slider)


def timeline(session: vd.Session, slider):
    import vr_plots as vp

    colors = vp.patch_colors(session.reward_sites["patch_label"])
    return vp.plot_timeline(session.sites, session.position, session.licks, colors, slider.value)
