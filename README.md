# Lab

Mini research projects on anything I'm curious about, both what I finish and what I don't.

Write-ups are at **[usmankhan.io/lab](https://usmankhan.io/lab/)**. The code and notes for each one are here.

## Experiments

Newest first. Status is one of **In progress**, **Done** or **Dropped**.

| Date | Experiment | Status | Folder | Write-up |
|---|---|---|---|---|
| 2026-10-13 | Decision models vs. LLMs | In progress | [2026-10-13-decision-models-vs-llms](2026-10-13-decision-models-vs-llms/) | [usmankhan.io/lab/decision-models-vs-llms](https://usmankhan.io/lab/decision-models-vs-llms/) |

## How this repo is laid out

```
lab/
├── README.md                          this file
├── docs/
│   ├── adding-a-lab.md                how to start a new experiment and list it
│   └── website.md                     how the website is set up and how a page is made
├── website/                           the static site, deployed to usmankhan.io/lab
│   ├── index.html                     list of experiments
│   ├── styles.css
│   └── <slug>/index.html              one page per experiment
├── templates/                         what new labs and pages are copied from
└── YYYY-MM-DD-<slug>/                 one folder per experiment: code, data, notes
    └── README.md
```

Every experiment is a folder at the root named by the date it started and a short slug. Its `README.md` carries the question, the method, a dated log and the results. The website page for it is the public write-up of the same thing.

## Adding a lab

Copy the two files in `templates/` into a new `YYYY-MM-DD-<slug>/` folder and `website/<slug>/`, fill in the placeholders, and add a row to the table above and to the list in `website/index.html`. The exact steps, placeholders and row formats are in [docs/adding-a-lab.md](docs/adding-a-lab.md). How the site deploys is in [docs/website.md](docs/website.md).
