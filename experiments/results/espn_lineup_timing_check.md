# ESPN Lineup Timing Check — Man Utd vs Leeds

**Match:** Manchester United vs Leeds United  
**Kickoff:** 19:00 UTC, 2026-04-13  
**Check time:** ~18:00 UTC (KO-60), 2026-04-13  
**EPL team sheet submission deadline:** 1 hour before KO (18:00 UTC)

---

## Method

Attempted to fetch the following ESPN API endpoints using WebFetch:

1. `https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard?dates=20260413`
2. `https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard` (no date param)
3. `https://www.espn.com/soccer/scoreboard/_/league/ENG.1` (web page)
4. `https://api.espn.com/v1/sports/soccer/eng.1/scoreboard?dates=20260413` (alternate API host)

All four requests returned **HTTP 403 Forbidden**.

---

## Findings

| Step | Result |
|------|--------|
| Scoreboard API (`site.api.espn.com`) | **403 Forbidden** |
| Scoreboard web page (`www.espn.com`) | **403 Forbidden** |
| Alternate API host (`api.espn.com`) | **403 Forbidden** |
| Event ID retrieved | **No** |
| Summary endpoint called | **No** (could not obtain event ID) |
| Lineup/roster fields checked | **No** (all requests blocked) |
| `rosters` field populated | **Unknown** |
| `starter` flags populated | **Unknown** |
| `formationPlace` values populated | **Unknown** |

---

## Root Cause

ESPN blocks automated HTTP requests that lack browser-like headers (e.g., `User-Agent`, `Accept`, `Referer`, session cookies). The `WebFetch` tool does not support custom request headers, so every request is rejected at the CDN/WAF layer before reaching ESPN's data servers.

---

## Conclusion

**NO — ESPN is NOT viable as a pre-KO lineup source in this context.**

This conclusion holds for two independent reasons:

1. **Access blocked:** ESPN's API and website return 403 for all automated requests without browser-like headers/cookies. Any production use would require a headless browser (Playwright/Puppeteer), which adds significant infrastructure complexity and is fragile against detection countermeasures.

2. **Timing uncertainty (unverified):** Even if access were obtained, it is unknown whether ESPN populates `rosters`/`starter`/`formationPlace` fields at team-sheet submission time (T-60 min) or only after kickoff. Based on general knowledge of ESPN's soccer data pipeline, lineup data typically populates close to kickoff or after the match starts — but this could not be confirmed today due to the 403 block.

**Recommended alternatives for pre-KO lineup data:**
- **football-data.org API** — free tier includes lineups, updates at team-sheet time
- **API-Football (RapidAPI)** — comprehensive lineup endpoint, well-documented timing
- **OddsPortal scraper** — already in-stack; could be extended to scrape team news sections
- Direct scraping of BBC Sport / Sky Sports match pages via Playwright (already used for OddsPortal)
