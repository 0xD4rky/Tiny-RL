"""HTTP client for rollout server (vLLM process)."""

from __future__ import annotations

import asyncio
import logging

import aiohttp

log = logging.getLogger(__name__)

TIMEOUT = aiohttp.ClientTimeout(total=1800)


async def _post(session: aiohttp.ClientSession, url: str, **kw) -> dict:
    async with session.post(url, timeout=TIMEOUT, **kw) as resp:
        resp.raise_for_status()
        return await resp.json()


class RolloutEngine:
    def __init__(self, base_url: str, port: int):
        self.address = f"{base_url}:{port}"

    async def wait_for_ready(self, poll_interval: float = 3.0):
        while True:
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.get(f"{self.address}/health", timeout=TIMEOUT) as r:
                        if r.status == 200:
                            break
            except Exception:
                pass
            await asyncio.sleep(poll_interval)
        log.info("Rollout server ready at %s", self.address)

    async def generate(self, questions: list[str], answers: list[str]) -> dict:
        async with aiohttp.ClientSession() as session:
            return await _post(
                session,
                f"{self.address}/generate",
                json={"questions": questions, "answers": answers},
            )

    async def init_weight_sync(self, mode: str, init_info: dict) -> dict:
        async with aiohttp.ClientSession() as session:
            return await _post(
                session,
                f"{self.address}/init_weight_sync",
                json={"mode": mode, "init_info": init_info},
            )

    async def update_weights_nccl(self, update_info: dict) -> dict:
        async with aiohttp.ClientSession() as session:
            return await _post(
                session,
                f"{self.address}/update_weights_nccl",
                json={"update_info": update_info},
            )

    async def reload_from_disk(self, model_path: str) -> dict:
        async with aiohttp.ClientSession() as session:
            return await _post(
                session,
                f"{self.address}/reload_from_disk",
                json={"model_path": model_path},
            )

    async def get_param_metadata(self) -> dict:
        async with aiohttp.ClientSession() as session:
            return await _post(session, f"{self.address}/get_param_metadata", json={})

    async def get_memory_regions(self) -> dict:
        async with aiohttp.ClientSession() as session:
            return await _post(session, f"{self.address}/get_memory_regions", json={})

    async def register_mrs(self, regions: list[dict]) -> dict:
        async with aiohttp.ClientSession() as session:
            return await _post(
                session,
                f"{self.address}/register_mrs",
                json={"regions": regions},
            )

    async def apply_rdma_routes(self, routes: list[dict], step: int) -> dict:
        async with aiohttp.ClientSession() as session:
            return await _post(
                session,
                f"{self.address}/apply_rdma_routes",
                json={"routes": routes, "step": step},
            )
