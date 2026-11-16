# Stats worker

A Cloudflare Worker that serves [Umami](https://umami.is) analytics for usmankhan.io and usmankhan.io/lab from its own `workers.dev` address instead of `cloud.umami.is`, so ad blockers that target known analytics domains don't drop the visits.

It does two things:

- `/p.js` returns Umami's tracker script (cached for a day).
- `/api/*` forwards the tracker's events to Umami Cloud, passing the visitor's IP and location in Umami's `x-umami-client-*` headers so visits aren't all counted as one visitor from the worker's location.

## Deploy

```bash
cd worker
npx wrangler login     # once
npx wrangler deploy
```

The deploy prints the URL, `https://orange-dew-4c3a.musmank006.workers.dev`.

## The tag on each page

Every page (the portfolio, the Lab list, each Lab page and `templates/page.html`) has this before `</head>`:

```html
<script defer src="https://orange-dew-4c3a.musmank006.workers.dev/p.js" data-website-id="b28fa752-87ce-42f6-a7c6-0e58e1adf2bc" data-host-url="https://orange-dew-4c3a.musmank006.workers.dev" data-domains="usmankhan.io"></script>
```

`data-domains` stops visits from `localhost` being counted.
