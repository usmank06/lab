# The website

The site is the `website/` folder: plain HTML and CSS, no build step, same look as [usmankhan.io](https://usmankhan.io). It is published with GitHub Pages at **https://usmankhan.io/lab/**.

## How it is deployed

The portfolio repo (`usmank06.github.io`) owns the `usmankhan.io` domain, so every other repo with Pages enabled is served under it at `usmankhan.io/<repo-name>/`. This repo is `lab`, so the site is at `/lab/`.

GitHub Pages can only serve a repo's root or its `/docs` folder on its own, and neither is the site here. So [`.github/workflows/pages.yml`](../.github/workflows/pages.yml) uploads `website/` as the Pages artifact on every push to `main` that touches it.

**One-time setup, or nothing deploys:** in the repo on GitHub, go to *Settings → Pages → Build and deployment* and set *Source* to **GitHub Actions**. After that, the workflow runs on push and the site is live a minute later. You can also run it by hand from the *Actions* tab (*Deploy website → Run workflow*).

Because the site lives under `/lab/`, every link and asset path in the HTML is relative: `styles.css` from the home page, `../styles.css` from an experiment page, `decision-models-vs-llms/` for a page, `../` to get back to the list. Never `/styles.css`, which would point at the portfolio. The one absolute link is `/`, which is the portfolio on purpose.

## Files

```
website/
├── index.html              the list of experiments, plus an About section
├── styles.css              all styling, shared by every page
└── <slug>/
    └── index.html          one experiment's write-up
```

The slug is the experiment's folder name without the date: `2026-10-13-decision-models-vs-llms/` in the repo is `decision-models-vs-llms/` on the site.

## How a page is put together

Every page is the same shell as the portfolio: a sticky sidebar on the left with the title, a one-line lead and a section nav; a scrolling column on the right. Under 960px it collapses to one column and the nav hides. The nav highlights the section in view with the same script as the portfolio.

An experiment page has these parts, in this order:

| Part | What goes in it |
|---|---|
| Sidebar | `Lab` kicker (links back to the list), the title, the one-line question, nav: Question · Method · Results · Next |
| Details card | Started date, status, link to the repo folder |
| The question | What you want to find out and why you care, then **What I expect** |
| Method | Numbered steps: what you build, what you compare, what you measure |
| Results | `Nothing yet.` until there are some; then paragraphs, a `<table>`, or an `<img>` with a `<p class="caption">` |
| What's next | The next step, or what you would do with more time |

[`templates/page.html`](../templates/page.html) is this layout with `{{PLACEHOLDERS}}`. To make a page, copy it to `website/<slug>/index.html` and replace every placeholder; the full list with formats and examples is in [adding-a-lab.md](adding-a-lab.md#1-start-it).

### Status

Status is shown by the `.status` element, in the Details card and in the list row. It takes one of three classes, and the text must match:

| Class | Text | Dot |
|---|---|---|
| `status-progress` | In progress | blue |
| `status-done` | Done | green |
| `status-dropped` | Dropped | grey |

The same three words are used in the README table. Keep them identical everywhere.

### Setting up a new page

1. Copy `templates/page.html` to `website/<slug>/index.html` and replace the placeholders.
2. Add a row to the top of the `<ul class="rows">` in `website/index.html`, newest first: a link to `<slug>/`, the title, the one-line question, the status and the month. The exact markup is in [adding-a-lab.md](adding-a-lab.md#2-list-it).
3. Write the Question, What I expect and Method sections. Leave Results as `Nothing yet.` until they exist.
4. Preview it locally (below), then commit and push. The workflow deploys it.

### Updating a page

- When the experiment finishes, change both `.status` elements (the page's Details card and its row on the home page) to `status-done` / `Done`, write the Results and What's next sections, and update the `Updated <month>` meta in the Details card.
- When you drop it, use `status-dropped` / `Dropped` and put a short note on why in Results.
- Update the status in the README table at the same time.

### Writing in the page

Text goes inside `<div class="prose">`. It styles `p`, `h3`, `ul`, `ol`, `a`, `code`, `pre`, `blockquote`, `table` and `img`. Keep paragraphs short. For a number-heavy result, a table beats prose. For a chart, export it as an image into the page's folder and use `<img>` with a caption.

## Previewing locally

From the repo root:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000/website/>. Serving the repo root rather than `website/` puts the site under a subpath, like `/lab/` in production, so a wrong absolute path shows up as broken locally instead of only after deploying.

## Changing the design

`styles.css` is a trimmed copy of the portfolio's stylesheet with the same tokens (`--text`, `--muted`, `--border`, `--accent`), type scale and breakpoints (960px, 640px). If the portfolio's design changes, carry the change over here so `/lab` still reads as part of the same site.
