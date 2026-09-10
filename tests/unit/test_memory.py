"""Memory tests: promotion policy, dedupe, episodic records, retention, staleness,
cross-turn references, and secret non-storage."""
from ara.memory import MemoryManager


def test_user_attribute_is_promoted(uow):
    mm = MemoryManager(uow)
    ok, _ = mm.should_promote_to_semantic("my name is Manikandan and I work at Acme")
    assert ok
    mid = mm.remember_user_fact(tenant_id="ten_a", user_id="u1", content="my name is Manikandan")
    assert mid is not None


def test_transient_statement_is_not_promoted(uow):
    mm = MemoryManager(uow)
    ok, reason = mm.should_promote_to_semantic("today I want to focus on revenue")
    assert not ok
    assert mm.remember_user_fact(tenant_id="ten_a", user_id="u1",
                                 content="today I want to focus on revenue") is None


def test_secrets_are_never_promoted(uow):
    mm = MemoryManager(uow)
    ok, reason = mm.should_promote_to_semantic("my password is hunter2")
    assert not ok and "sensitive" in reason
    assert mm.remember_user_fact(tenant_id="ten_a", user_id="u1", content="my api key is sk-abc") is None


def test_semantic_dedupe_replaces_not_duplicates(uow):
    mm = MemoryManager(uow)
    mm.remember_user_fact(tenant_id="ten_a", user_id="u1", content="my timezone is IST")
    mm.remember_user_fact(tenant_id="ten_a", user_id="u1", content="my timezone is IST")
    rows = uow.memories.search(tenant_id="ten_a", user_id="u1", kind="semantic", query="timezone")
    active = [r for r in rows if r["active"]]
    assert len(active) == 1


def test_episodic_memory_recorded_and_recalled(uow):
    mm = MemoryManager(uow)
    mm.record_episode(tenant_id="ten_a", user_id="u1", task_id="task_1", goal="Q3 revenue lookup",
                      outcome="COMPLETED", summary="Found revenue 50.2M in annual report p12")
    recalled = mm.recall(tenant_id="ten_a", user_id="u1", query="Q3 revenue")
    assert recalled and recalled[0]["kind"] == "episodic"
    assert "task_1" in recalled[0]["meta"]["task_id"] or "task_1" in recalled[0]["content"]


def test_tenant_isolation_of_memories(uow):
    mm = MemoryManager(uow)
    mm.remember_user_fact(tenant_id="ten_a", user_id="u1", content="my name is Alice")
    assert mm.recall(tenant_id="ten_b", user_id="u1", query="name") == []


def test_retention_expiry_purge(uow):
    mm = MemoryManager(uow)
    mid = mm.remember_user_fact(tenant_id="ten_a", user_id="u1", content="my timezone is IST",
                                retention="sensitive")
    # force expiry
    uow.db.execute("UPDATE memories SET expires_at='2000-01-01T00:00:00+00:00' WHERE id=?", (mid,))
    assert mm.recall(tenant_id="ten_a", user_id="u1", query="timezone") == []  # filtered as expired
    assert mm.purge_expired() == 1
    rows = uow.db.query("SELECT active FROM memories WHERE id=?", (mid,))
    assert rows[0]["active"] == 0


def test_memory_records_carry_policy_fields(uow):
    mm = MemoryManager(uow)
    mm.remember_user_fact(tenant_id="ten_a", user_id="u1", content="my name is Bob", confidence=0.95)
    row = uow.memories.search(tenant_id="ten_a", user_id="u1", kind="semantic", query="name")[0]
    assert row["source"] == "user" and row["confidence"] == 0.95
    assert row["retention"] == "standard" and row["expires_at"] is not None
    assert row["scope"] == "user"


def test_stale_memory_not_recalled_after_replacement(uow):
    mm = MemoryManager(uow)
    mm.remember_user_fact(tenant_id="ten_a", user_id="u1", content="my manager is Alice")
    mm.remember_user_fact(tenant_id="ten_a", user_id="u1", content="my manager is Bob")  # update
    rows = mm.recall(tenant_id="ten_a", user_id="u1", query="manager")
    assert len(rows) == 1 and "Bob" in rows[0]["content"]


def test_conversational_context_bounded(uow):
    cid = uow.conversations.create(tenant_id="ten_a", user_id="u1")
    for i in range(15):
        uow.conversations.add_message(cid, "user" if i % 2 == 0 else "assistant", f"msg {i}")
    mm = MemoryManager(uow)
    ctx = mm.context_for_task(tenant_id="ten_a", user_id="u1", query="msg", conversation_id=cid)
    assert len(ctx["recent_messages"]) == 10
