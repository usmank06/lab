// Analytics proxy for usmankhan.io.
//
// Serves Umami's tracker and forwards its events from this worker's own
// address, so blockers that target cloud.umami.is don't drop the requests.
// The visitor's IP and location are passed on in Umami Cloud's own
// x-umami-client-* headers; otherwise every visit would look like it came
// from this worker.

const UMAMI = 'https://cloud.umami.is';
const SCRIPT_PATH = '/p.js';

// Headers Umami's tracker sends with every event. They have to be allowed in
// the CORS preflight or the browser drops the request, and forwarded on.
const TRACKER_HEADERS = ['x-umami-website-id', 'x-umami-hostname', 'x-umami-cache'];

export default {
  async fetch(request) {
    const url = new URL(request.url);

    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: cors(request) });
    }

    if (url.pathname === SCRIPT_PATH) {
      const res = await fetch(`${UMAMI}/script.js`, {
        cf: { cacheEverything: true, cacheTtl: 86400 },
      });
      return new Response(res.body, {
        status: res.status,
        headers: {
          'content-type': 'application/javascript; charset=utf-8',
          'cache-control': 'public, max-age=86400',
        },
      });
    }

    if (url.pathname.startsWith('/api/')) {
      const headers = new Headers();
      headers.set('content-type', request.headers.get('content-type') || 'application/json');
      headers.set('user-agent', request.headers.get('user-agent') || '');
      for (const name of TRACKER_HEADERS) {
        const value = request.headers.get(name);
        if (value) headers.set(name, value);
      }

      const ip = request.headers.get('cf-connecting-ip');
      if (ip) {
        headers.set('x-umami-client-ip', ip);
        headers.set('x-forwarded-for', ip);
      }

      const cf = request.cf || {};
      if (cf.country) headers.set('x-umami-client-country', cf.country);
      if (cf.regionCode) headers.set('x-umami-client-region', cf.regionCode);
      if (cf.city) headers.set('x-umami-client-city', latin1(cf.city));

      const hasBody = !['GET', 'HEAD'].includes(request.method);
      const res = await fetch(`${UMAMI}${url.pathname}${url.search}`, {
        method: request.method,
        headers,
        body: hasBody ? await request.arrayBuffer() : undefined,
      });

      const out = new Response(res.body, res);
      for (const [k, v] of Object.entries(cors(request))) out.headers.set(k, v);
      return out;
    }

    return new Response('Not found', { status: 404 });
  },
};

function cors(request) {
  return {
    'access-control-allow-origin': request.headers.get('origin') || '*',
    'access-control-allow-methods': 'GET, POST, OPTIONS',
    'access-control-allow-headers':
      request.headers.get('access-control-request-headers') ||
      ['content-type', ...TRACKER_HEADERS].join(', '),
    'access-control-max-age': '86400',
    vary: 'origin',
  };
}

// Umami reads location headers as latin1 bytes and decodes them as UTF-8,
// so city names with accents have to be sent as their UTF-8 bytes.
function latin1(s) {
  return String.fromCharCode(...new TextEncoder().encode(s));
}
