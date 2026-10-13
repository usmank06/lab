# Adding a lab

A lab is one question, one folder in this repo, and one page on the site. This is the whole lifecycle: starting it, listing it, finishing or dropping it.

## 1. Start it

Decide the title, the one-line summary (the question the lab answers) and the slug; see [Naming](#naming). The start date is today. Then, from the repo root:

1. Create `YYYY-MM-DD-<slug>/` and copy `templates/README.md` into it as `README.md`.
2. Create `website/<slug>/` and copy `templates/page.html` into it as `index.html`.
3. In both files, replace every placeholder:

| Placeholder | Value | Example |
|---|---|---|
| `{{TITLE}}` | the title | `Decision models vs. LLMs` |
| `{{SUMMARY}}` | the one-line question | `How much better is a decision model than an LLM at ranking a few hundred options?` |
| `{{DATE}}` | start date, ISO | `2026-10-13` |
| `{{DATE_LONG}}` | start date, written out | `October 13, 2026` |
| `{{MONTH}}` | start month and year | `October 2026` |
| `{{SLUG}}` | the slug | `decision-models-vs-llms` |
| `{{FOLDER}}` | the folder name | `2026-10-13-decision-models-vs-llms` |

In the page, `{{TITLE}}` also lands in `<title>` and `{{SUMMARY}}` in a `content="…"` attribute, so escape `&` as `&amp;` and `"` as `&quot;` there. In body text, write apostrophes as `&rsquo;`, as the rest of the file does.

Nothing else in the templates needs changing to start. Check nothing was missed:

```bash
grep -rn '{{' YYYY-MM-DD-<slug> website/<slug>
```

## 2. List it

Two lists get a new row at the top, newest first. Both start as **In progress**, and both use the same title and summary as the page.

**`README.md`**, the experiments table. Columns: date, title, status, folder link, write-up link.

```
| 2026-10-13 | Decision models vs. LLMs | In progress | [2026-10-13-decision-models-vs-llms](2026-10-13-decision-models-vs-llms/) | [usmankhan.io/lab/decision-models-vs-llms](https://usmankhan.io/lab/decision-models-vs-llms/) |
```

**`website/index.html`**, the `<ul class="rows">` in the Experiments section. The month here is the short form, `Oct 2026`.

```html
<li>
    <a href="decision-models-vs-llms/" class="row">
        <span class="row-text">
            <span class="row-title">Decision models vs. LLMs</span>
            <span class="row-sub">How much better is a decision model than an LLM at ranking a few hundred options?</span>
        </span>
        <span class="row-meta"><span class="status status-progress">In progress</span><span class="meta">Oct 2026</span></span>
    </a>
</li>
```

## 3. Work in the folder

The folder is yours: scripts, notebooks, data, scratch files. Keep its `README.md` current as you go:

- **The question** and **What I expect**, written before you run anything.
- **Method**, updated if the plan changes.
- **Log**, one dated line per session: what you tried, what happened, what you decided.
- **Results**, when there are any.

Large or private data stays out of git. Put it in the folder's `.gitignore` or keep it elsewhere and note where in the README. If a lab needs a specific environment, say how to set it up at the top of its README (`requirements.txt`, `package.json`, whatever fits), so it still runs in a year.

Fill in the page's Question and Method sections early, from the folder's README. Leave Results as `Nothing yet.`

## 4. Finish it, or drop it

When it is done:

1. Write **Results** and **What's next** in the folder's README and on the page. Say what came out, including if it was weak or surprising.
2. Change the status to **Done** in all three places: the README table, the page's Details card, and the page's row in `website/index.html`. The class and the word go together: `status-done` / `Done`.
3. Update the `Updated <month>` meta in the page's Details card.

When you stop early, do the same with **Dropped** (`status-dropped`) and put a short note in Results on why.

Then commit and push. Pushing anything under `website/` deploys the site; see [website.md](website.md) for how, and for the one-time Pages setting it needs.

## Naming

- Folder: `2026-10-13-decision-models-vs-llms`. Date it started, then the slug.
- Slug: short, lowercase, hyphens, no date. It is the URL: `usmankhan.io/lab/decision-models-vs-llms/`.
- Title: sentence case, as you would say it. `Decision models vs. LLMs`, not `decision_models_vs_llms`.
- Summary: one line, phrased as the question the lab answers. It is the lead on the page and the subtitle in the list.
