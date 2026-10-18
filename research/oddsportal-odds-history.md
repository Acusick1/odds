# OddsPortal per-bookmaker odds history for past EPL matches (#508)

Verdict: NOT USABLE as a source of full per-bookmaker odds history, and NOT USABLE as a Pinnacle source. The ~7 bookmakers and missing Pinnacle/Betfair in the 2021-24 DB snapshots are a limit of the site's logged-out UK view, not of the scrape. The one scrape-side loss is Betfair Exchange, which was scraped and then dropped at ingest.

Probed 2026-10-18 from a London residential IP (ipinfo: GB, Hyperoptic) with Playwright Firefox, headless, logged out, consent declined. No proxy, no account. The Playwright MCP tools were not available to this agent, so I drove Firefox through the repo's Python Playwright instead. Script: `oddsportal-odds-history/probe_history.py`; captured tooltip text: `oddsportal-odds-history/*.json`.

## 1. What a past-match page shows (Chelsea 2-1 Bournemouth, 2023-24, 2024-05-19)

Page: `https://www.oddsportal.com/football/h2h/bournemouth-OtpNdwrc/chelsea-4fGZN2oK/#0rgRCqPd` (the id after `#` is the match).

- 9 bookmakers on 1X2 Full Time: 888sport, bet365, Betfred, BetUK, BetVictor, Betway, bwin, Unibetuk, William Hill. Betfair Exchange sits in a separate "Betting Exchanges" table (back and lay with matched amounts, e.g. back 1.44 (8249)). No Pinnacle.
- 8 other 2023-24 match pages loaded in a row: 9 bookmakers each.
- Hovering an odds cell opens an "ODDS MOVEMENT" tooltip. It holds exactly two points: the last movement (timestamp, value, delta) and the opening odds (timestamp, value). Example, bet365 draw: `19 May, 15:48  5.25  +0.92 | Opening odds: 12 May, 11:59  4.33`. See `chelsea_bournemouth_2023-24_modals.json`.
- It is not a full series. An upcoming match (Bournemouth v Sunderland, 18 books) has the same two-point tooltip. The existing scrape agrees: across `data/external/oddsportal/epl_{2021-2022,2023-2024,2025-2026}.json`, every `odds_history` list has length 1 (6,820 / 7,854 / 13,725 lists; 1 empty).
- Betfair Exchange has no tooltip history in the scrape: 0 of 376 rows in 2023-24 carry `odds_history_data`. Each row is a single string with the matched amount fused on, like `"10/110/1(1648)"`.

## 2. How the data is exposed

- The page is client rendered (Next.js). The 1X2 grid arrives in one XHR: `/proxy/match-event/1-1-<matchId>-1-2-<hash>.dat?geo=BG&lang=en`. It is a base64 blob (encrypted), about 3 KB for 9 books and 4.5 KB for 18. The size scales with the visible bookmaker count, which leaves no room for hidden books.
- Each tooltip hover fires one XHR: `/proxy/match-event-history/1-<matchId>-1-2-0-<bookmakerId>/?geo=BG&lang=en` (observed ids 16, 21, 27; bet365 is 16). The body is about 300 bytes, encrypted, consistent with the two points above.
- The exchange table loads from `/proxy/ajax-betting-exchanges/1-<matchId>-1-2/`.
- No plain JSON endpoint was seen. I did not try to decrypt the payloads.
- `geo=BG` is sent even though the IP is GB; I could not tell where it comes from.

## 3. Why ~7 books, and why no Pinnacle

- From this UK IP the bookmaker list is the same UK-licensed set on every sport and league: NBA (17), LaLiga (18), recent EPL (18), 2023-24 EPL (9). Pinnacle never appears, and `/bookmakers/` does not list it.
- The `centroquote.it` mirror returned the same 9 books for the same match.
- So the list is set by the site for this viewer; it is not an OddsHarvester filter. The fork's `--bookies-filter` defaults to `all` (`OddsHarvester/src/oddsharvester/cli/options.py`, `utils/bookies_filter_enum.py`), `scripts/scrape_oddsportal.py` never passes it, and today's match page has no classic/crypto selector. The "All Bonuses" tab is a promotions list, not a filter.
- The set per past match is whatever OddsPortal serves at scrape time, and it drifts. The March 2026 scrape of 2023-24 had bet365, Betfred, BetVictor, Betway, Unibetuk, bwin, BetMGM (130/380), BetUK (101/380) and 10bet (10). Today's page has 888sport and William Hill instead of BetMGM and 10bet.
- Local dev DB, EPL 2022-23, bookmaker rows across 1x2 and totals snapshots: bet365 2,196, betvictor 2,196, bwin 2,194, unibet_uk 1,436, betway 1,436, betfred 1,434, 10bet 1,370. Pinnacle appears only on the 760 football-data.co.uk snapshots. Betfair Exchange appears on no 2021-24 OddsPortal snapshot.
- Not proven: that another region shows Pinnacle. That needs a non-UK exit; the fork notes say books vary by region (`OddsHarvester/docs/agentic-gotchas.md` section 7). All I can say is that a UK logged-out view does not.

## 4. Betfair Exchange was scraped, then dropped at ingest

The scrape JSON holds `Betfair Exchange` on every 2021-24 match (back prices plus matched liquidity). `odds_lambda.oddsportal_common.build_raw_data` skips any bookmaker whose `odds_history_data` has fewer than `num_outcomes` entries (`packages/odds-lambda/odds_lambda/oddsportal_common.py`, guard `if not hist or len(hist) < num_outcomes: continue`). The exchange row has none, so it never reaches the DB. The existing files could be re-ingested with an exchange-specific parser. Limits: closing back price only, no opening and no timestamp, and the price string needs splitting from the matched amount.

## 5. Rate limits and Cloudflare

- About 25 page loads and 20 hovers in a session, 2.5 to 3.5 s apiece: all HTTP 200, with no challenge, CAPTCHA or rate-limit response.
- One failure: the legacy match URL stored in the March scrape (`/football/england/premier-league-2023-2024/chelsea-bournemouth-0rgRCqPd/`) is now a 404 by curl and `NS_ERROR_NET_EMPTY_RESPONSE` in Firefox, 3 of 3 tries. The `/football/h2h/<a>/<b>/#<id>` form works, and results pages now expose only those links. The fork already reads the `#` fragment (`_extract_fragment_match_id` in `OddsHarvester/src/oddsharvester/core/base_scraper.py`).
- Layout risk: the fork's selectors (`div.border-black-borders.flex.h-9`, `div.flex-center.flex-col.font-bold`, `h3:text('Odds movement')`) matched 0 rows on today's page. The new markup uses `tr.h-9` rows, `p.cursor-pointer` odds cells and an "ODDS MOVEMENT" heading. I did not run the harvester itself, so I cannot say whether its user agent gets the old layout.

## 6. Can the fork extract it?

Yes for what exists: opening plus last movement per bookmaker and outcome via the hover tooltip, which is what it does today. No for a full timestamped series, because the site does not expose one. That would need a different source (own snapshots over time, or Odds API or Betfair history).

## Cost

No paid calls, no Odds API units, no login.
