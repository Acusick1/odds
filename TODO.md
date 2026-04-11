# TODO

IMPORTANT: This doc is only for items to be done. Upon item completion, any necessary information should be persisted elsewhere, and the item removed completely from this doc.

## Model / Data

- [ ] Publish updated model to S3 — best feature set is tabular + standings (MSE 0.000726)
- [ ] Betfair Exchange historical data — buy Advanced data for real-time market-implied probabilities as sharp reference (replaces Pinnacle, which has been unreliable since July 2025)

## Production

- [ ] OddsPortal match rate — was ~55%, possibly caused by overlay modal not Cloudflare. Modal fix deployed 2026-04-09 (#269). Check match rate after a few days to confirm
- [ ] OddsHarvester fork cleanup — reduce TimeoutError log noise (Playwright TimeoutError vs Python TimeoutError) and reduce modal wait from 2s to 500ms. Bundle into next fork update

## Data Sources

- [ ] CLAUDE.md says The Odds API is "currently disabled" — it's not, `fetch-odds-epl` and `fetch-scores-epl` Lambda jobs are actively calling it for EPL. Update CLAUDE.md to reflect that both Odds API (US bookmakers) and OddsPortal (UK bookmakers) are active for EPL
- [ ] Investigate when NBA data stopped flowing into prod DB. Terraform only has EPL in `sport_configs`, but unclear when NBA was removed. Check EventBridge rule history / git log for the Terraform change
## Betting Agent (odds-mcp)

## Open Issues (lower priority)

- [ ] #158: Fix MLflow autologging crash during hyperparameter tuning with CV
