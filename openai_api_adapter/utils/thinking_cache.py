"""
Thinking blocks cache for preserving Claude's reasoning during tool use.

When Claude uses extended thinking with tools, thinking blocks must be
passed back to the API to maintain reasoning continuity. Since OpenAI
format doesn't have thinking blocks, we cache them server-side and
restore them when tool results are sent back.

Cache key: tool_call_id (unique per tool invocation)
Cache value: list of thinking block dicts

Thread Safety: Uses a lock to protect cache operations in async environments.
"""

from threading import Lock
from typing import Any

from cachetools import TTLCache

from openai_api_adapter.config import settings
from openai_api_adapter.utils.logger import logger

# Global cache instance with configurable TTL and max size
_thinking_cache: TTLCache[str, list[dict[str, Any]]] = TTLCache(
    maxsize=settings.thinking_cache_maxsize,
    ttl=settings.thinking_cache_ttl,
)

# Lock for thread-safe cache operations
_cache_lock = Lock()


def cache_thinking_blocks(tool_call_ids: list[str], thinking_blocks: list[dict[str, Any]]) -> None:
    """
    Cache thinking blocks associated with tool call IDs.

    Thread-safe: Uses lock to protect concurrent cache writes.

    Args:
        tool_call_ids: List of tool call IDs from the response
        thinking_blocks: List of thinking block dicts to cache
    """
    if not thinking_blocks or not tool_call_ids:
        logger.warning(f"cache_thinking_blocks called with empty data: tool_call_ids={tool_call_ids}, thinking_blocks_count={len(thinking_blocks) if thinking_blocks else 0}")
        return

    # Log what we're caching for debugging
    block_types = [b.get("type", "unknown") for b in thinking_blocks]
    logger.info(f"Caching {len(thinking_blocks)} thinking blocks (types: {block_types}) for tool_call_ids: {tool_call_ids}")

    # Associate thinking blocks with each tool call ID
    # All tool calls in the same response share the same thinking blocks
    with _cache_lock:
        for tool_call_id in tool_call_ids:
            _thinking_cache[tool_call_id] = thinking_blocks
            logger.debug(f"Cached thinking blocks for tool_call_id={tool_call_id}")
        logger.info(f"Cache size after write: {len(_thinking_cache)}")


def get_thinking_blocks(tool_call_id: str) -> list[dict[str, Any]] | None:
    """
    Retrieve cached thinking blocks for a tool call ID.

    Thread-safe: Uses lock to protect concurrent cache reads.

    Args:
        tool_call_id: The tool call ID to look up

    Returns:
        List of thinking block dicts, or None if not found/expired
    """
    with _cache_lock:
        # Log cache state for debugging
        cache_keys = list(_thinking_cache.keys())
        logger.debug(f"Cache lookup: tool_call_id={tool_call_id}, cache_size={len(_thinking_cache)}, cached_keys={cache_keys[:5]}{'...' if len(cache_keys) > 5 else ''}")

        blocks = _thinking_cache.get(tool_call_id)
        if blocks:
            logger.info(f"Cache HIT: Retrieved {len(blocks)} thinking blocks for tool_call_id={tool_call_id}")
        else:
            logger.warning(f"Cache MISS: No thinking blocks found for tool_call_id={tool_call_id}")
        return blocks


def remove_thinking_blocks(tool_call_id: str) -> None:
    """
    Remove cached thinking blocks for a tool call ID.

    Thread-safe: Uses lock to protect concurrent cache modifications.
    Call this after the thinking blocks have been used to free memory.

    Args:
        tool_call_id: The tool call ID to remove
    """
    with _cache_lock:
        if tool_call_id in _thinking_cache:
            del _thinking_cache[tool_call_id]
            logger.debug(f"Removed thinking blocks for tool_call_id={tool_call_id}")


def remove_thinking_blocks_batch(tool_call_ids: list[str]) -> None:
    """
    Remove cached thinking blocks for multiple tool call IDs.

    Thread-safe: Uses lock to protect concurrent cache modifications.
    More efficient than calling remove_thinking_blocks multiple times.

    Args:
        tool_call_ids: List of tool call IDs to remove
    """
    if not tool_call_ids:
        return

    with _cache_lock:
        for tool_call_id in tool_call_ids:
            if tool_call_id in _thinking_cache:
                del _thinking_cache[tool_call_id]
                logger.debug(f"Removed thinking blocks for tool_call_id={tool_call_id}")


def get_cache_stats() -> dict[str, Any]:
    """
    Get cache statistics for monitoring.

    Thread-safe: Uses lock to protect concurrent cache reads.

    Returns:
        Dict with cache size, max size, and TTL info
    """
    with _cache_lock:
        return {
            "current_size": len(_thinking_cache),
            "max_size": _thinking_cache.maxsize,
            "ttl_seconds": _thinking_cache.ttl,
        }
