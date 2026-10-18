"""Read-only probe of historicdata.betfair.com using the repo's Betfair cert login. Run: uv run python api_probe.py"""

import betfairlightweight
import requests
from odds_core.config import get_settings

bf = get_settings().betfair
c = betfairlightweight.APIClient(
    bf.username, bf.password, app_key=bf.app_key, cert_files=(bf.cert_file, bf.cert_key)
)
c.login()
H = {"ssoid": c.session_token, "X-Application": bf.app_key, "content-type": "application/json"}
B = "https://historicdata.betfair.com/api/"
print(requests.get(B + "GetMyData", headers=H, timeout=60).text)
print(requests.get(B + "GetStandardRateCard", headers=H, timeout=60).text)
f = {
    "sport": "Soccer",
    "plan": "Basic Plan",
    "fromDay": 1,
    "fromMonth": 8,
    "fromYear": 2024,
    "toDay": 31,
    "toMonth": 12,
    "toYear": 2024,
    "marketTypesCollection": ["MATCH_ODDS"],
    "countriesCollection": ["GB"],
}
print(requests.post(B + "GetAdvBasketDataSize", headers=H, json=f, timeout=60).text)
print(
    requests.post(B + "DownloadListOfFiles", headers=H, json=f, timeout=60).text
)  # 400 until a (free) Basic purchase is made
