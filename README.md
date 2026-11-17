# VR foraging daily summary

A one-page summary of a VR-foraging session: headline stats, water per patch type, the choice raster,
velocity at odor onset, stop probability over the session, P(leave) and a scrollable timeline.
It runs entirely in your browser; there's nothing to install.

**Open it:** https://allenneuraldynamics.github.io/marimo-faas-demo/

## Using it

- **Pick a session:** type a subject id, pick the day (UTC), then choose the session from the list.
  With no session in the link, the page opens the newest session.
- **Share a session:** the address bar updates when you pick a session. Copy it, or build a link yourself:

  ```text
  https://allenneuraldynamics.github.io/marimo-faas-demo/?session=<session_id>
  ```

  For example:
  https://allenneuraldynamics.github.io/marimo-faas-demo/?session=878046_2026-11-13_20-59-59

- **Export as PDF:** the button at the top downloads a report with the stats and every figure.
  To get it without clicking, add `&pdf=1` to a session link and the PDF downloads once the page has
  loaded:
  https://allenneuraldynamics.github.io/marimo-faas-demo/?session=878046_2026-11-13_20-59-59&pdf=1
- **Timeline:** drag the slider to move the 60 s window through the session.

The first load takes about half a minute while Python starts in your browser; a progress bar shows each
step. Stats appear first and the plots follow a few seconds later. Reloads and sessions you've already
opened are faster.

Only sessions that have been processed show up. A session started after 4 pm Pacific is listed under the
next day, since days are in UTC.

## Running it on your machine

With [uv](https://docs.astral.sh/uv/) installed:

```bash
uv run marimo run notebooks/daily_summary.py      # open the app in your browser
uv run marimo edit notebooks/daily_summary.py     # edit the notebook
```

Add `?session=<session_id>` to the local URL to open a specific session.

To write a session's PDF without opening a browser:

```bash
uv run python notebooks/daily_summary.py --session <session_id> --pdf report.pdf
```
