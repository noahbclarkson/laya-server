"""A local server that answers TypeSafe System One requests with Laya.

It speaks the wire format documented at https://docs.typesafe.ai/api, so any
TypeSafe client can use it by pointing its base URL here:

    POST /v1/systemone   evaluate typed questions against a state
    GET  /v1/models      the model names this server answers to
    GET  /healthz        device, loaded checkpoints

Model names:

    laya                    route each request by script and language
    laya-english            ModernBERT-large, 421M, English
    laya-multilingual       mmBERT-base, 322M, 100+ languages
    laya-typed-decisions    ModernBERT-large, 421M, fine-tuned on four workflows
    jev-*                   the same as `laya`, only with --jev-alias

Only the checkpoints named in --models are loaded, and all of them are loaded
before the server starts listening, so no request pays for a download or a
model build.

Laya keeps only the first few hundred tokens of a state. When it cuts one
short, the response carries a `truncation` object and an `x-laya-truncated`
header, or the request fails with a 422 under --reject-truncated.
"""

import argparse
import hmac
import json
import os
import sys
import threading
import time
import uuid
from typing import Any

AUTO = "laya"
CHECKPOINTS = {
    "laya-english": "english",
    "laya-multilingual": "multilingual",
    "laya-typed-decisions": "typed-decisions",
}
NAMES = {checkpoint: name for name, checkpoint in CHECKPOINTS.items()}
JEV_PREFIX = "jev-"
JEV_ALIASES = ("jev-latest", "jev-preview")
QUESTION_TYPES = ("noul", "choice", "score")
REQUEST_ID_HEADER = "x-typesafe-request-id"


class Rejected(Exception):
    """A request the server will not run, and the HTTP status that says why."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


class Engine:
    """The loaded checkpoints and everything that touches them.

    `router` is a `laya.Router` with every name in `checkpoints` already loaded.
    """

    def __init__(
        self,
        router: Any,
        checkpoints: list[str],
        *,
        reject_truncated: bool = False,
        jev_alias: bool = False,
        release_date: str = "unknown",
    ):
        self.router = router
        self.checkpoints = checkpoints
        self.reject_truncated = reject_truncated
        self.jev_alias = jev_alias
        self.release_date = release_date
        # One request at a time on the model. The GPU gains little from overlapping
        # requests, and the fast tokenizer raises "Already borrowed" when two
        # threads use it at once.
        self.lock = threading.Lock()

    def agent(self, checkpoint: str) -> Any:
        return self.router.load(checkpoint)

    def device(self) -> dict[str, Any]:
        import torch

        agent = self.agent(self.checkpoints[0])
        info: dict[str, Any] = {"device": str(agent.device), "dtype": str(agent.dtype)}
        if agent.device.type == "cuda":
            index = agent.device.index
            if index is None:
                index = torch.cuda.current_device()
            info["gpu"] = torch.cuda.get_device_name(index)
            info["hip"] = torch.version.hip
            info["vram_reserved_gb"] = round(torch.cuda.memory_reserved(index) / 2**30, 2)
        return info

    def models(self) -> list[dict[str, str]]:
        loaded = ", ".join(NAMES[c] for c in self.checkpoints)
        cards = [
            {
                "name": AUTO,
                "description": f"Routes each request by script and language to one of the "
                f"loaded checkpoints: {loaded}.",
                "release_date": self.release_date,
            }
        ]
        for checkpoint in self.checkpoints:
            cfg = self.agent(checkpoint).cfg
            max_len = cfg.get("max_len", 512)
            head = cfg.get("head_max_len", 192)
            cards.append(
                {
                    "name": NAMES[checkpoint],
                    "description": f"Laya {checkpoint} checkpoint on {cfg.get('encoder')}: a "
                    f"{max_len}-token context per question, shared by the question, its "
                    f"options, and the state, which keeps at least {max_len - head} tokens.",
                    "release_date": self.release_date,
                }
            )
        if self.jev_alias:
            for alias in JEV_ALIASES:
                cards.append(
                    {
                        "name": alias,
                        "description": f"Alias for `{AUTO}`, so a client configured for Jev "
                        f"works unchanged. Every jev-* name is accepted the same way; the "
                        f"response's `model` names the Laya checkpoint that answered.",
                        "release_date": self.release_date,
                    }
                )
        return cards

    def evaluate(self, body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
        state = body.get("state")
        if not isinstance(state, (str, dict, list)):
            raise Rejected(422, f"state: must be a string, object, or array; found {_kind(state)}")
        questions = _questions(body.get("questions"))
        model = body.get("model") or AUTO
        if not isinstance(model, str):
            raise Rejected(422, f"model: must be a string; found {_kind(model)}")

        with self.lock:
            checkpoint, routing = self._route(model, state, questions)
            agent = self.agent(checkpoint)
            truncation = _truncation(agent, state, questions)
            if truncation and self.reject_truncated:
                most = max(truncation["kept_tokens"].values())
                raise Rejected(
                    422,
                    f"state: {truncation['state_tokens']} tokens, but {NAMES[checkpoint]} keeps "
                    f"at most {most} with these questions; split the state or turn off "
                    f"--reject-truncated",
                )
            try:
                result = agent.system_one(state, questions)
            except ValueError as err:
                raise Rejected(422, f"questions: {err}") from err

        result["model"] = NAMES[checkpoint]
        result["routing"] = routing
        headers = {}
        if truncation:
            result["truncation"] = truncation
            headers["x-laya-truncated"] = ",".join(truncation["kept_tokens"])
        return result, headers

    def _route(
        self, model: str, state: Any, questions: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        requested = model
        if self.jev_alias and model.startswith(JEV_PREFIX):
            model = AUTO

        if model == AUTO:
            decision = self.router.route(state, questions)
            if decision.model not in self.checkpoints:
                wanted = ",".join([*self.checkpoints, decision.model])
                raise Rejected(
                    422,
                    f"model: routing chose {NAMES[decision.model]} ({decision.reason}), which "
                    f"this server has not loaded; restart with --models {wanted} or name a "
                    f"loaded model",
                )
            checkpoint = decision.model
        else:
            checkpoint = CHECKPOINTS.get(model)
            if checkpoint not in self.checkpoints:
                served = ", ".join([AUTO] + [NAMES[c] for c in self.checkpoints])
                hint = ""
                if model.startswith(JEV_PREFIX):
                    hint = "; start the server with --jev-alias to answer Jev model names"
                raise Rejected(
                    422, f"model: {model!r} is not served here; choose one of {served}{hint}"
                )
            decision = self.router.route(state, questions, model=checkpoint)

        routing = dict(decision)
        if requested != model:
            routing["requested"] = requested
        return checkpoint, routing


def _questions(raw: Any) -> dict[str, dict[str, Any]]:
    """Checks the questions map and fills in what Laya requires but the API does not."""
    if not isinstance(raw, dict) or not raw:
        raise Rejected(422, "questions: must be an object with at least one question")
    problems = []
    questions = {}
    for qid, question in raw.items():
        path = f"questions.{qid}"
        if not isinstance(question, dict):
            problems.append(f"{path}: must be an object; found {_kind(question)}")
            continue
        kind = question.get("type")
        if kind not in QUESTION_TYPES:
            problems.append(
                f"{path}.type: must be one of {', '.join(QUESTION_TYPES)}; found {kind!r}"
            )
            continue
        instructions = question.get("instructions")
        if instructions is not None and not isinstance(instructions, (str, dict, list)):
            problems.append(
                f"{path}.instructions: must be a string, object, or array; "
                f"found {_kind(instructions)}"
            )
        criteria = question.get("criteria")
        if kind == "choice" and not (isinstance(criteria, (dict, list)) and criteria):
            problems.append(f"{path}.criteria: a choice needs at least one option")
        if kind == "score" and not (isinstance(criteria, list) and criteria):
            problems.append(f"{path}.criteria: a score needs a list of at least one level")
        if kind == "noul" and criteria is not None and not isinstance(criteria, dict):
            problems.append(
                f"{path}.criteria: must be an object with `true` and `false`; "
                f"found {_kind(criteria)}"
            )
        # The API makes instructions optional; Laya indexes it unconditionally.
        questions[qid] = dict(question, instructions="" if instructions is None else instructions)
    if problems:
        raise Rejected(422, "; ".join(problems))
    return questions


def _truncation(agent: Any, state: Any, questions: dict[str, dict[str, Any]]) -> dict | None:
    """How much of the state each question kept, when any of them lost some.

    Measured with Laya's own sequence builder: the state's share of a sequence
    is its length minus the length of the same sequence built on an empty state.
    """
    from laya import Agent
    from laya.common import build_sequence

    max_len = agent.cfg.get("max_len", 512)
    head = agent.cfg.get("head_max_len", 192)
    total = None
    kept = {}
    for qid, question in questions.items():
        q = Agent._to_internal(question)
        bare = len(build_sequence(agent.tok, "", q, max_len, head)[0])
        if total is None:
            total = len(build_sequence(agent.tok, state, q, sys.maxsize, head)[0]) - bare
        kept[qid] = len(build_sequence(agent.tok, state, q, max_len, head)[0]) - bare
    short = {qid: n for qid, n in kept.items() if n < total}
    if not short:
        return None
    return {"state_tokens": total, "kept_tokens": short}


def default_device() -> str | None:
    """The first discrete GPU, or None to leave the choice to Laya.

    ROCm lists a Ryzen's integrated graphics as a GPU too, often as device 0,
    and Laya's own default of `cuda` means device 0.
    """
    import torch

    if not torch.cuda.is_available():
        return None
    for index in range(torch.cuda.device_count()):
        if not getattr(torch.cuda.get_device_properties(index), "is_integrated", False):
            return f"cuda:{index}"
    return None


def release_date() -> str:
    """The date the checkpoint repository last changed, or `unknown` offline."""
    try:
        from huggingface_hub import HfApi
        from laya.router import BUNDLE_REPO

        info = HfApi().model_info(BUNDLE_REPO, timeout=5)
        return info.last_modified.date().isoformat()
    except Exception:
        return "unknown"


def _kind(value: Any) -> str:
    if value is None:
        return "null"
    kinds = {
        bool: "boolean",
        int: "number",
        float: "number",
        str: "string",
        dict: "object",
        list: "array",
    }
    return kinds.get(type(value), type(value).__name__)


def build_app(engine: Engine, api_key: str | None = None):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from starlette.concurrency import run_in_threadpool

    app = FastAPI(title="laya-server", docs_url=None, redoc_url=None)

    def error(status: int, detail: str, request_id: str) -> JSONResponse:
        return JSONResponse(
            {"detail": detail}, status_code=status, headers={REQUEST_ID_HEADER: request_id}
        )

    def authorized(request: Request) -> bool:
        if api_key is None:
            return True
        given = request.headers.get("authorization", "")
        return hmac.compare_digest(given.encode(), f"Bearer {api_key}".encode())

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "loaded": [NAMES[c] for c in engine.checkpoints], **engine.device()}

    @app.get("/v1/models")
    def models(request: Request):
        if not authorized(request):
            return error(401, "missing or wrong bearer token", f"laya-{uuid.uuid4().hex}")
        return {"models": engine.models()}

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        request_id = f"laya-{uuid.uuid4().hex}"
        if not authorized(request):
            return error(401, "missing or wrong bearer token", request_id)
        try:
            body = json.loads(await request.body())
        except ValueError as err:
            return error(400, f"the body is not JSON: {err}", request_id)
        if not isinstance(body, dict):
            return error(400, f"the body must be a JSON object; found {_kind(body)}", request_id)

        started = time.perf_counter()
        try:
            result, headers = await run_in_threadpool(engine.evaluate, body)
        except Rejected as err:
            print(f"[laya-server] {request_id} {err.status} {err.detail}", flush=True)
            return error(err.status, err.detail, request_id)
        elapsed = (time.perf_counter() - started) * 1000
        truncated = ", state truncated" if "truncation" in result else ""
        print(
            f"[laya-server] {request_id} 200 {result['model']}, "
            f"{len(result['answers'])} question(s), {elapsed:.0f} ms{truncated}",
            flush=True,
        )
        headers.update({REQUEST_ID_HEADER: request_id, "x-laya-elapsed-ms": f"{elapsed:.1f}"})
        return JSONResponse(result, headers=headers)

    return app


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    env = os.environ.get
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Every option can also be set with the environment variable in brackets.",
    )
    parser.add_argument(
        "--host",
        default=env("LAYA_SERVER_HOST", "127.0.0.1"),
        help="address to bind; anything but loopback exposes the model "
        "(LAYA_SERVER_HOST, default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(env("LAYA_SERVER_PORT", "8765")),
        help="(LAYA_SERVER_PORT, default: %(default)s)",
    )
    parser.add_argument(
        "--models",
        default=env("LAYA_SERVER_MODELS", "english"),
        help="checkpoints to load, comma separated: english, multilingual, typed-decisions "
        "(LAYA_SERVER_MODELS, default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default=env("LAYA_SERVER_DEVICE") or None,
        help="torch device such as cuda:1 or cpu "
        "(LAYA_SERVER_DEVICE, default: the first discrete GPU, else the CPU)",
    )
    parser.add_argument(
        "--api-key",
        default=env("LAYA_SERVER_API_KEY") or None,
        help="require this bearer token (LAYA_SERVER_API_KEY, default: none)",
    )
    parser.add_argument(
        "--reject-truncated",
        action="store_true",
        default=_flag("LAYA_SERVER_REJECT_TRUNCATED"),
        help="answer 422 instead of evaluating a state Laya would cut short "
        "(LAYA_SERVER_REJECT_TRUNCATED=1)",
    )
    parser.add_argument(
        "--jev-alias",
        action="store_true",
        default=_flag("LAYA_SERVER_JEV_ALIAS"),
        help="answer jev-* model names as `laya`, so a client configured for Jev needs only "
        "its base URL changed (LAYA_SERVER_JEV_ALIAS=1)",
    )
    args = parser.parse_args(argv)

    from laya.router import normalise_name

    try:
        names = (n for n in args.models.split(",") if n.strip())
        args.checkpoints = list(dict.fromkeys(normalise_name(n) for n in names))
    except ValueError as err:
        parser.error(f"--models: {err}")
    if not args.checkpoints:
        parser.error("--models: name at least one checkpoint")
    return args


def main() -> None:
    args = parse_args()

    from laya import Router

    print(f"[laya-server] loading {', '.join(args.checkpoints)} ...", flush=True)
    started = time.perf_counter()
    router = Router(device=args.device or default_device(), max_loaded=len(args.checkpoints))
    router.preload(args.checkpoints)
    engine = Engine(
        router,
        args.checkpoints,
        reject_truncated=args.reject_truncated,
        jev_alias=args.jev_alias,
        release_date=release_date(),
    )
    print(
        f"[laya-server] loaded in {time.perf_counter() - started:.1f} s on {engine.device()}; "
        f"listening on http://{args.host}:{args.port}"
        f"{', answering jev-* as laya' if args.jev_alias else ''}",
        flush=True,
    )

    import uvicorn

    uvicorn.run(
        build_app(engine, args.api_key), host=args.host, port=args.port, log_level="warning"
    )


if __name__ == "__main__":
    main()
