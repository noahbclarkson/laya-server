"""The HTTP layer, validation, routing, and truncation, without a real model.

Laya's own router and sequence builder run for real; only the checkpoint is a
stand-in, so these run in seconds and need no download.
"""

import pytest
import torch
from fastapi.testclient import TestClient
from laya import Router

from server import Engine, build_app, parse_args


class FakeTokenizer:
    """One token per word, which is all Laya's sequence builder needs."""

    mask_token = "[MASK]"
    pad_token_id, cls_token_id, sep_token_id, mask_token_id = 0, 1, 2, 3

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [10 + len(word) for word in text.split()]}


class FakeAgent:
    """Answers the way a Laya agent does, with fixed values."""

    def __init__(self):
        self.cfg = {"max_len": 64, "head_max_len": 24, "encoder": "fake-encoder"}
        self.tok = FakeTokenizer()
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.calls = []
        self.error = None

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        if self.error:
            raise self.error
        answers = {}
        for qid, q in questions.items():
            if q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": 0.75, "confidence": 0.75}
            elif q["type"] == "choice":
                options = list(q["criteria"])
                share = round(1 / len(options), 4)
                answers[qid] = {
                    "type": "choice",
                    "choice": options[0],
                    "probabilities": dict.fromkeys(options, share),
                    "confidence": 0.0,
                }
            else:
                levels = q["criteria"]
                answers[qid] = {
                    "type": "score",
                    "score": (len(levels) - 1) / 2,
                    "legend": {str(i): level for i, level in enumerate(levels)},
                    "probabilities": {
                        str(i): round(1 / len(levels), 4) for i in range(len(levels))
                    },
                    "confidence": 0.0,
                }
        return {
            "model": "laya-rl-agent",
            "answers": answers,
            "usage": {"input_tokens": 7, "output_tokens": 0},
        }


def serve(api_key=None, **options):
    agent = FakeAgent()
    router = Router()
    router.attach("english", agent)
    engine = Engine(router, ["english"], release_date="2026-09-20", **options)
    return TestClient(build_app(engine, api_key)), agent


NOUL = {"type": "noul", "instructions": "Is this a set of financial statements?"}


def evaluate(client, state="Statement of financial position", questions=None, **body):
    return client.post(
        "/v1/systemone", json={"state": state, "questions": questions or {"q": NOUL}, **body}
    )


def test_an_evaluation_comes_back_in_the_typesafe_shape():
    client, _ = serve()
    response = evaluate(
        client,
        questions={
            "fs": NOUL,
            "kind": {
                "type": "choice",
                "instructions": "Which document?",
                "criteria": {"accounts": None, "invoice": "A bill"},
            },
            "size": {"type": "score", "instructions": "How long?", "criteria": ["short", "long"]},
        },
        model="laya",
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "laya-english"
    assert set(body["answers"]) == {"fs", "kind", "size"}
    assert body["answers"]["size"]["legend"] == {"0": "short", "1": "long"}
    assert body["usage"]["input_tokens"] == 7
    assert body["routing"]["model"] == "english"
    assert "truncation" not in body
    assert response.headers["x-typesafe-request-id"].startswith("laya-")
    assert float(response.headers["x-laya-elapsed-ms"]) >= 0


def test_a_missing_model_means_laya():
    client, _ = serve()
    assert evaluate(client).json()["model"] == "laya-english"


def test_a_checkpoint_can_be_named():
    client, _ = serve()
    response = evaluate(client, model="laya-english")
    assert response.json()["routing"]["reason"] == "explicit model='english'"


def test_missing_instructions_reach_laya_as_an_empty_string():
    client, agent = serve()
    assert evaluate(client, questions={"q": {"type": "noul"}}).status_code == 200
    _, questions = agent.calls[-1]
    assert questions["q"]["instructions"] == ""


def test_every_malformed_question_is_reported_at_once():
    client, agent = serve()
    response = evaluate(
        client,
        questions={
            "a": {"type": "maybe"},
            "b": {"type": "choice", "instructions": "?", "criteria": {}},
            "c": {"type": "noul", "instructions": 5},
            "d": {"type": "score", "instructions": "?", "criteria": "low, high"},
        },
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    for path in ("questions.a.type", "questions.b.criteria", "questions.c.instructions"):
        assert path in detail
    assert "questions.d.criteria" in detail
    assert agent.calls == []


@pytest.mark.parametrize(
    "body, status, fragment",
    [
        (b"{not json", 400, "not JSON"),
        (b"[1, 2]", 400, "found array"),
        (b'{"state": 42, "questions": {"q": {"type": "noul"}}}', 422, "state: must be"),
        (b'{"state": "x", "questions": {}}', 422, "at least one question"),
        (b'{"state": "x", "questions": {"q": {"type": "noul"}}, "model": 3}', 422, "model:"),
    ],
)
def test_a_malformed_body_names_what_is_wrong(body, status, fragment):
    client, _ = serve()
    response = client.post("/v1/systemone", content=body)
    assert response.status_code == status
    assert fragment in response.json()["detail"]
    assert response.headers["x-typesafe-request-id"].startswith("laya-")


def test_an_unserved_model_is_refused_with_the_served_ones_named():
    client, _ = serve()
    detail = evaluate(client, model="laya-multilingual").json()["detail"]
    assert "not served" in detail
    assert "laya, laya-english" in detail


def test_a_jev_name_is_refused_with_the_fix_unless_aliased():
    client, _ = serve()
    response = evaluate(client, model="jev-latest")
    assert response.status_code == 422
    assert "--jev-alias" in response.json()["detail"]


def test_with_the_alias_a_jev_name_is_answered_by_laya_and_says_so():
    client, _ = serve(jev_alias=True)
    for name in ("jev-latest", "jev-1.13.0"):
        body = evaluate(client, model=name).json()
        assert body["model"] == "laya-english"
        assert body["routing"]["requested"] == name


def test_the_model_list_follows_the_alias():
    plain = [m["name"] for m in serve()[0].get("/v1/models").json()["models"]]
    aliased = [m["name"] for m in serve(jev_alias=True)[0].get("/v1/models").json()["models"]]
    assert plain == ["laya", "laya-english"]
    assert aliased == ["laya", "laya-english", "jev-latest", "jev-preview"]


def test_routing_to_an_unloaded_checkpoint_says_how_to_load_it():
    client, agent = serve()
    response = evaluate(
        client, state="Der Kunde wurde zweimal belastet und möchte eine Erstattung."
    )
    assert response.status_code == 422
    assert "--models english,multilingual" in response.json()["detail"]
    assert agent.calls == []


def test_a_long_state_reports_what_each_question_kept():
    client, _ = serve()
    response = evaluate(client, state="word " * 200)
    body = response.json()
    assert body["truncation"]["state_tokens"] == 200
    assert 0 < body["truncation"]["kept_tokens"]["q"] < 200
    assert response.headers["x-laya-truncated"] == "q"


def test_a_long_state_can_be_refused_instead():
    client, agent = serve(reject_truncated=True)
    response = evaluate(client, state="word " * 200)
    assert response.status_code == 422
    assert "200 tokens" in response.json()["detail"]
    assert agent.calls == []


def test_a_laya_value_error_is_unprocessable():
    client, agent = serve()
    agent.error = ValueError("question 'q' options exceed head_max_len=24")
    response = evaluate(client)
    assert response.status_code == 422
    assert "head_max_len" in response.json()["detail"]


def test_a_token_is_required_only_when_configured():
    client, _ = serve(api_key="s3cret")
    assert client.get("/v1/models").status_code == 401
    assert evaluate(client).status_code == 401
    wrong = client.get("/v1/models", headers={"authorization": "Bearer nope"})
    assert wrong.status_code == 401
    right = client.get("/v1/models", headers={"authorization": "Bearer s3cret"})
    assert right.status_code == 200
    assert client.get("/healthz").status_code == 200


def test_health_reports_the_device_and_the_loaded_checkpoints():
    body = serve()[0].get("/healthz").json()
    assert body == {
        "status": "ok",
        "loaded": ["laya-english"],
        "device": "cpu",
        "dtype": "torch.float32",
    }


def test_options_fall_back_to_the_environment(monkeypatch):
    monkeypatch.setenv("LAYA_SERVER_MODELS", "english, ML")
    monkeypatch.setenv("LAYA_SERVER_PORT", "9000")
    monkeypatch.setenv("LAYA_SERVER_JEV_ALIAS", "true")
    args = parse_args([])
    assert args.checkpoints == ["english", "multilingual"]
    assert args.port == 9000
    assert args.jev_alias is True
    assert args.reject_truncated is False
    assert parse_args(["--models", "typed"]).checkpoints == ["typed-decisions"]


def test_an_unknown_checkpoint_is_a_usage_error():
    with pytest.raises(SystemExit):
        parse_args(["--models", "klingon"])
