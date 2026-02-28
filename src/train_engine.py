"""
Thin async HTTP client that fans out requests to all FSDP train-server ranks.

Every method sends the same request to every rank in parallel so that
FSDP collectives (all-gather / reduce-scatter) stay in lockstep.
"""

import asyncio
import logging
from typing import Any

import aiohttp

log = logging.getLogger(__name__)

TIMEOUT = aiohttp.ClientTimeout(total=1200)


async def _post(session: aiohttp.ClientSession, url: str, **kw) -> dict:
    async with session.post(url, timeout=TIMEOUT, **kw) as resp:
        resp.raise_for_status()
        return await resp.json()


class TrainEngine:

    def __init__(self, base_url: str, base_port: int, num_ranks: int):
        self.addresses = [f"{base_url}:{base_port + r}" for r in range(num_ranks)]

    async def _post_all(self, endpoint: str, **kw) -> list[dict]:
        """POST to every rank simultaneously and collect responses."""
        async with aiohttp.ClientSession() as session:
            return await asyncio.gather(
                *[_post(session, f"{addr}{endpoint}", **kw) for addr in self.addresses],
            )

    async def wait_for_ready(self, poll_interval: float = 3.0):
        """Block until every rank's /health endpoint responds."""
        for addr in self.addresses:
            while True:
                try:
                    async with aiohttp.ClientSession() as s:
                        async with s.get(f"{addr}/health", timeout=TIMEOUT) as r:
                            if r.status == 200:
                                break
                except Exception:
                    pass
                await asyncio.sleep(poll_interval)
        log.info("All %d train servers ready", len(self.addresses))

    # ── collective endpoints (must hit every rank) ───────────────────

    async def create_online_dataset(self, rollout_batches: list[dict[str, Any]]):
        return await self._post_all(
            "/create_online_dataset",
            json={"rollout_batches": rollout_batches},
        )

    async def train_1_iter(self) -> list[dict]:
        return await self._post_all("/train_1_iter")

    async def save_weights(self) -> list[dict]:
        return await self._post_all("/save_weights")

    async def plan_weight_sync(self) -> list[dict]:
        return await self._post_all("/plan_weight_sync")

    async def init_weight_sync(
        self,
        master_address: str,
        master_port: int,
        world_size: int,
    ) -> list[dict]:
        return await self._post_all(
            "/init_weight_sync",
            json={
                "master_address": master_address,
                "master_port": master_port,
                "world_size": world_size,
            },
        )

    async def prepare_weight_sync(self) -> list[dict]:
        return await self._post_all("/prepare_weight_sync", json={})

    async def broadcast_weights(self, packed: bool) -> list[dict]:
        return await self._post_all("/broadcast_weights", json={"packed": packed})
