"""
cache.py

Custom caching class, currently used for FeatureBase
"""
# needed for defaultdict annotation if < python 3.9
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Union


class CacheType(Enum):
    """Enumerates the supported cache types"""

    DEPENDENCY = 1
    DEPTH = 2


@dataclass()
class FeatureCache:
    """Provides caching for the defined types"""

    enabled: bool = False
    cache: defaultdict[dict] = field(default_factory=lambda: defaultdict(dict))

    def get(
        self,
        cache_type: CacheType,
        hashkey: int,
    ) -> Optional[Union[List[Any], Any]]:
        """Gets the cache entry, if enabled and defined

        Args:
            cache_type (CacheType): type of cache
            hashkey (int): hash key

        Returns:
            Optional[Union[List[Any], Any]]: payload assigned to the hashkey
        """
        if not self.enabled or cache_type not in self.cache:
            return None
        return self.cache[cache_type].get(hashkey, None)

    def add(self, cache_type: CacheType, hashkey: int, payload: Any):
        """Adds an entry to the cache, if enabled

        Args:
            cache_type (CacheType): type of cache
            hashkey (int): hash key
            payload (Any): payload to assign
        """
        if self.enabled:
            self.cache[cache_type][hashkey] = payload

    def clear_all(self):
        """Clears the cache collections"""
        self.cache.clear()


feature_cache = FeatureCache()
