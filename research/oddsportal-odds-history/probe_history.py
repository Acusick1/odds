import asyncio
import json
import sys

from playwright.async_api import async_playwright

URL = sys.argv[1]
out = sys.argv[2]


async def main():
    async with async_playwright() as p:
        b = await p.firefox.launch(headless=True)
        ctx = await b.new_context(
            viewport={"width": 1400, "height": 2000}, locale="en-GB", timezone_id="Europe/London"
        )
        page = await ctx.new_page()
        feeds = []
        page.on(
            "request",
            lambda r: feeds.append(r.url)
            if "/proxy/" in r.url and ("match-event" in r.url or "geo=" in r.url or "odds" in r.url)
            else None,
        )
        await page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(6000)
        try:
            await page.click('button:has-text("Reject All")', timeout=3000)
        except Exception as e:
            print("no consent", e)
        rows = page.locator("tr.h-9").filter(has=page.locator('a[href^="/proxy/bookmakers/"]'))
        n = await rows.count()
        print("rows", n)
        res = {}
        for i in range(n):
            row = rows.nth(i)
            names = await row.locator("p.text-black-main").evaluate_all("e=>e.map(x=>x.innerText)")
            nm = names[0] if names else (await row.inner_text()).split("\n")[0]
            blocks = row.locator("p.cursor-pointer")
            bc = await blocks.count()
            res[nm] = {"row_text": (await row.inner_text()).replace("\n", " | "), "hist": []}
            if nm.lower().startswith(("bet365", "betfred")) or i == 0:
                for j in range(bc):
                    await blocks.nth(j).hover()
                    await page.wait_for_timeout(1500)
                    try:
                        h = page.locator("text=Odds movement").first
                        await h.wait_for(timeout=4000)
                        html = await h.evaluate('n=>n.closest("div").parentElement.innerText')
                        res[nm]["hist"].append(html)
                    except Exception as e:
                        res[nm]["hist"].append("ERR " + str(e)[:80])
        print(json.dumps(res, indent=1))
        print("\n".join(sorted(set(feeds))))
        print([(c["name"], c["domain"]) for c in await ctx.cookies()])
        json.dump(res, open(out, "w"), indent=1)
        await b.close()


asyncio.run(main())
