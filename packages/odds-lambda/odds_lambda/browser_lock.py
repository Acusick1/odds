"""Module-level semaphore to serialize Playwright browser access."""

import asyncio

playwright_semaphore = asyncio.Semaphore(1)
