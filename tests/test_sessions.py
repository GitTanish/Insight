from webapp.sessions import SessionStore


def test_ttl_expires_idle_sessions():
    now = [1000.0]
    store = SessionStore(max_sessions=4, ttl_seconds=60, clock=lambda: now[0])
    session = store.get_or_create("a")
    assert store.get("a") is session

    now[0] += 61

    assert store.get("a") is None
    assert len(store) == 0


def test_get_touches_lru_and_caps_session_count():
    now = [0.0]
    store = SessionStore(max_sessions=2, clock=lambda: now[0])
    first = store.get_or_create("a")
    store.get_or_create("b")

    store.get("a")
    now[0] += 1
    third = store.get_or_create("c")

    assert store.get("b") is None
    assert store.get("a") is first
    assert store.get("c") is third


def test_expired_sessions_are_not_counted_against_capacity():
    now = [0.0]
    store = SessionStore(max_sessions=2, ttl_seconds=10, clock=lambda: now[0])
    store.get_or_create("old")
    now[0] += 20

    store.get_or_create("new")

    assert len(store) == 1


def test_drop_and_empty_sid():
    store = SessionStore()
    store.get_or_create("x")
    store.drop("x")
    assert store.get("x") is None
    assert store.get("") is None
