# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from typing import Optional
import redis
import json
import asyncio


class CacheClient(ABC):
    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc_value, traceback):
        pass

    @abstractmethod
    async def get(self, args: dict) -> Optional[dict]:
        pass

    @abstractmethod
    async def upload(self, args: dict, result: dict) -> None:
        pass


class RedisCacheClient(CacheClient):
    def __init__(self):
        self.redis = redis.Redis(host='localhost', port=6379, db=0)

    async def get(self, args: dict) -> Optional[dict]:
        key = self.get_key(args)
        value = self.redis.get(key)
        if value is None:
            return None
        return json.loads(value)

    async def upload(self, args: dict, result: dict) -> None:
        key = self.get_key(args)
        self.redis.set(key, json.dumps(result))

    def get_key(self, args: dict) -> str:
        return " ".join(f"{k}={v}" for k, v in sorted(args.items(), key=lambda x: x[0]))


async def simple_test():
    x = {'a': 1, 'b': 2}
    z = {'a': 1, 'b': 3}
    redis_cache = RedisCacheClient()
    await redis_cache.upload(x, x)
    assert await redis_cache.get(x) == x, redis_cache.get(x)
    assert await redis_cache.get(z) is None
    print('passed')

if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(simple_test())
