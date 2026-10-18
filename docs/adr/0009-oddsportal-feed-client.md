---
status: accepted
---

# Read OddsPortal live odds from its JSON feed instead of the rendered page

## Context

The live odds job scraped OddsPortal through the OddsHarvester fork, which drives headless Chromium and parses the rendered HTML. By October 2026 that scrape returned 0 matches on every run, on both hosts. The production database's last `oddsportal_live` snapshot was 2026-08-10 (#484).

OddsPortal had rebuilt its site as a Next.js app. A read-only investigation on 2026-10-17 (#484) found:

- **Listing pages.** No element carries the `eventRow` class the fork searches for. The fixtures are server-rendered as JSON inside the page's React Server Components payload (`self.__next_f.push` scripts): one object per event with `encodeEventId`, `home-name`, `away-name`, `date-start-timestamp` and `url`.
- **Match pages.** Old-style match URLs return 404. Matches live at `/{sport}/h2h/{team}/{team}/#{eventId}`. The page builds its odds table from an encrypted JSON feed, `/proxy/match-event/1-{sportId}-{eventId}-{bettingType}-{scope}-{hashedBookmakers}.dat`. None of the fork's CSS selectors match the new markup.
- **Encryption.** Feed responses are base64 of `ciphertext:ivHex`: AES-256-CBC with a PBKDF2-SHA256 key (1000 iterations), optionally gzipped. The password and salt are constants in the site's own JavaScript bundle.
- **No browser needed.** Plain HTTP GETs with the site's XHR headers return the listing, `/proxy/api/home-data` (which supplies `hashedBookmakers` and `geoIPcode`), the bookmaker names (`/proxy/ajax-providers-bonus-data/0/`) and every feed. This was verified for EPL 1X2 and over/under and for MLB home/away.
- **Reliability.** Browser sessions intermittently received empty-body 503s, from home and from a GitHub Actions runner. Headless Chromium also sends `sec-ch-ua: "HeadlessChrome"`. A feed can return the error envelope `{"d": {"E": "dcapiexc"}, "refresh": 20}`; a retry after the hinted delay succeeded.
- **Richer data.** The feed carries what the page only showed visually or not at all: decimal odds per bookmaker and outcome, opening odds and times, last-change times, Betfair Exchange matched volume, and an `act` flag that is false for suspended (struck-through) prices.

Two fixes were considered in #484: rewrite the fork's selectors for the new markup, or read the feed directly. A selector rewrite would keep the browser, its intermittent 503s and the per-redesign breakage. It would also still have to parse odds from rendered text.

## Decision

The live odds job reads OddsPortal over plain HTTP with an in-repo client, `odds_lambda.oddsportal_client`. It parses the listing payload for events, fetches one feed per event and market, and decrypts it with the site's published key. The live path uses no browser and no OddsHarvester.

The results job (`fetch-oddsportal-results`) is outside this decision and still uses OddsHarvester.

## Consequences

- The live scrape no longer depends on page markup, Chromium or the OddsHarvester fork. Breakage moves to three narrower points: the RSC event fields, the feed URL scheme and the encryption constants. A changed key fails loudly as a decryption error rather than as a silent 0-match scrape.
- The feed and its key are undocumented. OddsPortal can change either without notice, and the client has to be updated by reading their JavaScript again.
- Odds arrive as decimals with Betfair volumes. Suspended prices are dropped using `act`, which the DOM scrape could not see. The adapter no longer parses fractional strings or Betfair's `"99/10099/100(300)"` cells.
- Kickoff times come from the listing's `date-start-timestamp`. The guard against H2H pages hydrating a past meeting is no longer needed.
- The bookmaker set is whatever OddsPortal serves to the scraper's region (`hashedBookmakers`, `geoIPcode`). From the UK it is 19 UK books including Betfair Exchange, without Pinnacle or US books.
- The same client can serve the results job and historical backfills, but neither uses it yet.

## Evidence

- Issue: #484.
- Site bundle: the decryption function and constants are in the Next.js chunk that throws `"Invalid encrypted AJAX payload"`. Search the site's `/_next/static/chunks/*.js` for that string.
- Test fixtures captured on 2026-10-17: `tests/fixtures/oddsportal/`.
