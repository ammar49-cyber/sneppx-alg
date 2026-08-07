"""Tests for MLflow-class observability: registry, lineage, prompts, gateway, eval monitor."""

import os
import tempfile
import numpy as np
from SneppX_ALG.interface_bindings.mlflow_like import (
    JSONStore,
    ModelRegistry,
    LineageStore,
    PromptRegistry,
    AIGateway,
    EvalMonitor,
    SneppxTrackingClient,
    GatewayRoute,
)


def test_json_store_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        store = JSONStore(d)
        store.put("coll", "a", {"x": 1, "y": [1.5, "two"]})
        assert store.get("coll", "a")["x"] == 1
        assert store.get("coll", "missing") is None
        assert len(store.list("coll")) == 1
        assert store.delete("coll", "a") is True
        assert store.delete("coll", "a") is False
        assert store.list("coll") == []
    print("  test_json_store_roundtrip PASS")


def test_model_registry_lifecycle():
    with tempfile.TemporaryDirectory() as d:
        reg = ModelRegistry(root=d)
        reg.create_registered_model("gpt", tags={"task": "lm"}, description="demo")
        try:
            reg.create_registered_model("gpt")
            raise AssertionError("expected duplicate error")
        except ValueError:
            pass
        v1 = reg.register_model("gpt", run_id="run_1", metrics={"acc": 0.8})
        v2 = reg.register_model("gpt", run_id="run_2", metrics={"acc": 0.9})
        assert v1.version == 1 and v2.version == 2
        assert reg.get_model_version("gpt", "latest") == v2
        assert reg.get_model_version("gpt", 1).run_id == "run_1"
        assert reg.list_registered_models() == ["gpt"]
        reg.transition_model_version_stage("gpt", 2, "Staging")
        v2b = reg.get_model_version("gpt", 2)
        assert v2b.stage == "Staging"
        reg.transition_model_version_stage("gpt", 2, "Production")
        assert [v.version for v in reg.get_latest_versions("gpt", ["Production"])] == [2]
        reg.set_registered_model_alias("gpt", "champion", 2)
        assert reg.get_model_version_by_alias("gpt", "champion").version == 2
        assert reg.delete_registered_model("gpt") is True
        assert reg.get_registered_model("gpt") is None
    print("  test_model_registry_lifecycle PASS")


def test_lineage_dag():
    with tempfile.TemporaryDirectory() as d:
        ls = LineageStore(root=d)
        parent = ls.create_run("base", experiment_name="sft", dataset_hash="h0")
        child = ls.create_run("tune", parent_run_id="base", source_uri="fs:///ckpt")
        grand = ls.create_run("eval", parent_run_id="tune")
        assert ls.get_run("base").child_run_ids == ["tune"]
        names = [r.run_id for r in ls.lineage_of("eval")]
        assert names == ["base", "tune", "eval"]
        assert [r.run_id for r in ls.descendants_of("base")] == ["tune", "eval"]
        ls.set_tags("base", {"kind": "data"})
        assert ls.get_run("base").tags["kind"] == "data"
        ls.associate_model("tune", "gpt")
        assert "gpt" in ls.get_run("tune").model_names
        try:
            ls.create_run("base")
            raise AssertionError("expected duplicate error")
        except ValueError:
            pass
    print("  test_lineage_dag PASS")


def test_prompt_registry():
    with tempfile.TemporaryDirectory() as d:
        pr = PromptRegistry(root=d)
        p1 = pr.register_prompt("summarize", "Summarize {text} in {language}.")
        assert sorted(p1.variables) == ["language", "text"]
        pr.register_prompt("summarize", "Sum: {text}")
        assert pr.get_prompt("summarize", 1).version == 1
        assert pr.get_prompt("summarize", "latest").version == 2
        assert pr.render("summarize", version=1, text="hi", language="en") == "Summarize hi in en."
        assert pr.render("summarize", text="hi") == "Sum: hi"
        try:
            pr.render("summarize", version=1, text="hi")
            raise AssertionError("expected missing variable error")
        except ValueError:
            pass
        assert pr.list_prompts() == ["summarize"]
    print("  test_prompt_registry PASS")


def test_gateway_routing_and_usage():
    with tempfile.TemporaryDirectory() as d:
        gw = AIGateway(root=d)
        gw.add_route("fast", model="gpt-lite", priority=10)
        gw.add_route("slow", model="gpt-lite", priority=0)
        gw.add_route("pro", model="gpt-pro", priority=5)
        routes = gw.list_routes()
        assert routes[0].name == "fast"
        assert gw.query("x", model="gpt-lite") == "x"
        assert gw.get_route("fast").usage["calls"] == 1
        gw.set_api_key("pro", "k-123")
        assert gw.get_route("pro").api_key == "k-123"
        gw.enable("pro", False)
        # disabled routes are never selected, even when requested explicitly
        assert gw.query("x", route="pro") == "x"
        assert gw.get_route("fast").usage["calls"] == 2
        # with all routes disabled, routing raises
        for r in gw.list_routes():
            gw.enable(r.name, False)
        try:
            gw.query("x")
            raise AssertionError("expected no-route error")
        except RuntimeError:
            pass
        report = gw.usage_report()
        assert report["fast"]["calls"] == 2.0
    print("  test_gateway_routing_and_usage PASS")


def test_gateway_custom_backend():
    with tempfile.TemporaryDirectory() as d:
        seen = []
        def backend(model, inputs):
            seen.append((model, inputs))
            return f"{model}:{inputs}"
        gw = AIGateway(backend=backend, root=d)
        gw.add_route("r", model="m1")
        assert gw.query("q", route="r") == "m1:q"
        assert seen == [("m1", "q")]
    print("  test_gateway_custom_backend PASS")


def test_eval_monitor():
    with tempfile.TemporaryDirectory() as d:
        em = EvalMonitor(root=d)
        em.log_eval("r1", "accuracy", 0.80, model_name="m")
        em.log_eval("r2", "accuracy", 0.85, model_name="m")
        em.log_eval("r3", "accuracy", 0.79, model_name="m")
        hist = em.history("m", "accuracy")
        assert len(hist) == 3
        dlt = em.deltas("m", "accuracy")
        assert dlt["latest"] == 0.79
        assert abs(dlt["delta"] - (-0.06)) < 1e-9
        assert dlt["trend"] == "down"
        alerts = em.regressions(threshold=-0.04)
        assert any(s.value == 0.79 for s in alerts)
        dlt1 = em.deltas("other", "accuracy")
        assert dlt1["delta"] is None and dlt1["trend"] == "flat"
    print("  test_eval_monitor PASS")


def test_client_facade_and_persistence():
    with tempfile.TemporaryDirectory() as d:
        client = SneppxTrackingClient(root=d)
        mv = client.register_model("gpt", run_id="run_1", metrics={"acc": 0.9})
        client.transition_model_version_stage("gpt", mv.version, "Production")
        client.create_run("run_1", experiment_name="sft", dataset_hash="h0")
        client.associate_model("run_1", "gpt")
        client.register_prompt("qa", "Answer: {question}")
        client.add_route("main", model="gpt", priority=1)
        assert client.query("q", model="gpt") == "q"
        assert client.render_prompt("qa", question="why") == "Answer: why"
        assert client.lineage_of("run_1")[0].run_id == "run_1"
        # reload from disk: same root, fresh instances
        client2 = SneppxTrackingClient(root=d)
        assert client2.get_model_version("gpt", 1).stage == "Production"
        assert client2.get_run("run_1").dataset_hash == "h0"
        assert client2.get_prompt("qa").version == 1
        assert client2.gateway.get_route("main").usage["calls"] == 1.0
    print("  test_client_facade_and_persistence PASS")


def test_stage_transition_guard():
    with tempfile.TemporaryDirectory() as d:
        reg = ModelRegistry(root=d)
        reg.create_registered_model("m")
        v = reg.register_model("m")
        reg.transition_model_version_stage("m", v.version, "Production")
        try:
            reg.transition_model_version_stage("m", v.version, "Archived")
            raise AssertionError("expected transition guard error")
        except ValueError:
            pass
        try:
            reg.transition_model_version_stage("m", v.version, "Bogus")
            raise AssertionError("expected invalid stage error")
        except ValueError:
            pass
    print("  test_stage_transition_guard PASS")


if __name__ == "__main__":
    test_json_store_roundtrip()
    test_model_registry_lifecycle()
    test_lineage_dag()
    test_prompt_registry()
    test_gateway_routing_and_usage()
    test_gateway_custom_backend()
    test_eval_monitor()
    test_client_facade_and_persistence()
    test_stage_transition_guard()
    print("ALL observability (mlflow_like) TESTS PASS")
