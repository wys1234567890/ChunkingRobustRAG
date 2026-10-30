from pathlib import Path

from src.cache import CacheStore, build_cache_key


def test_cache_key_is_stable():
    messages = [{"role": "user", "content": "hello"}]
    first = build_cache_key(
        messages,
        model="model-a",
        temperature=0,
        max_tokens=10,
        prompt_version="p1",
        config_version="c1",
    )
    second = build_cache_key(
        messages,
        model="model-a",
        temperature=0,
        max_tokens=10,
        prompt_version="p1",
        config_version="c1",
    )
    assert first == second
    assert len(first) == 64


def test_cache_store_round_trip(tmp_path: Path):
    store = CacheStore(tmp_path)
    key = "a" * 64
    value = {"status": "success", "text": "cached"}
    store.save(key, value)
    assert store.load(key) == value

