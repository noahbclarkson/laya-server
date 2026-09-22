# laya-server

[![CI](https://github.com/noahbclarkson/laya-server/actions/workflows/ci.yml/badge.svg)](https://github.com/noahbclarkson/laya-server/actions/workflows/ci.yml)
[![Licence: MIT OR Apache-2.0](https://img.shields.io/badge/licence-MIT%20OR%20Apache--2.0-blue)](#licence)

Run [Laya](https://github.com/NandhaKishorM/laya), an open-weights System 1
decision model, behind the [TypeSafe System One](https://docs.typesafe.ai/api)
HTTP API. Any TypeSafe client, such as the Rust
[`typesafe-api`](https://github.com/noahbclarkson/typesafe-api-rs) crate, can
then use a local model by changing only its base URL.

```sh
docker compose up -d
export TYPESAFE_BASE_URL=http://localhost:8765 TYPESAFE_API_KEY=local
# your existing Jev code now runs against Laya
```

> This is a community project. It is not affiliated with or endorsed by
> TypeSafe or by Convai Innovations, who publish Laya.

## What you get

| Endpoint | Does |
| --- | --- |
| `POST /v1/systemone` | evaluates `noul`, `choice`, and `score` questions against a state, with the same request and answer shapes as TypeSafe |
| `GET /v1/models` | the model names this server answers to |
| `GET /healthz` | the device, the GPU, the loaded checkpoints, and VRAM in use |

Every response carries an `x-typesafe-request-id`, and errors come back as
`{"detail": "..."}` with the status TypeSafe clients already branch on: `400`
for a body that is not JSON, `401` for a wrong token, `422` for anything the
server will not evaluate, with every problem named.

## Quick start with Docker

```sh
git clone https://github.com/noahbclarkson/laya-server
cd laya-server
docker compose up -d
docker compose ps        # "healthy" once the checkpoint has loaded
```

The first start downloads the English checkpoint (about 850 MB) into the
`models` volume; later starts take seconds. The port is published on
`127.0.0.1` only.

```sh
curl -s localhost:8765/v1/systemone -H 'content-type: application/json' -d '{
  "state": "STATEMENT OF FINANCIAL POSITION AS AT 31 MARCH 2026 ...",
  "questions": {
    "is_accounts": {"type": "noul", "instructions": "Is this a set of financial statements?"}
  }
}'
```

### A drop-in for Jev

The image starts with the Jev alias on: any `jev-*` model name, including the
`jev-latest` default that TypeSafe clients send, is answered as `laya`. A
client configured for Jev therefore needs only its base URL changed. The alias
never hides what answered: the response's `model` is always the Laya
checkpoint, such as `laya-english`, and `routing.requested` records the name
that was asked for. Set `LAYA_SERVER_JEV_ALIAS=0` to refuse Jev names instead.

### With a GPU

| Host | GPU | How |
| --- | --- | --- |
| Linux | NVIDIA | `docker compose -f compose.yaml -f compose.cuda.yaml up -d` (needs the NVIDIA Container Toolkit) |
| Windows | NVIDIA | the same, under Docker Desktop's WSL 2 backend |
| Linux | AMD | `RENDER_GID=$(getent group render \| cut -d: -f3) docker compose -f compose.yaml -f compose.rocm.yaml up -d` |
| Windows | AMD | run natively; see below. Docker Desktop cannot pass an AMD GPU through without AMD's WSL bridge ([ROCDXG](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html)), which this repository does not set up |
| any | none | the default `docker compose up -d` |

The GPU images are built from PyTorch's `cu130` and `rocm7.2` wheel indexes.
Only the CPU image and the native Windows AMD install are exercised by CI or
by hand so far; reports from the GPU images are welcome.

## Native install on Windows with an AMD GPU

This is the fast path on a Radeon under Windows. It uses AMD's ROCm build of
PyTorch, and needs Python 3.13 (`py -3.13`).

```powershell
.\setup.ps1                        # RX 7600 (gfx1102); -GfxTarget gfx1100 for an RX 7900, -Cpu for no GPU
.venv\Scripts\python server.py     # add --jev-alias for a drop-in Jev endpoint
```

`setup.ps1` finishes by running `check_gpu.py`:

```text
torch 2.13.0+rocm10.0.0, CUDA None, HIP 7.15.26333
  cuda:0  AMD Radeon(TM) Graphics      gfx1036  integrated
  cuda:1  AMD Radeon RX 7600           gfx1102  discrete
laya-server will use cuda:1 (AMD Radeon RX 7600).
```

ROCm lists a Ryzen's integrated graphics as a GPU too, often as `cuda:0`, so
Laya's own default of `cuda` would pick the wrong one. The server uses the
first discrete GPU instead; `--device` overrides it.

Elsewhere, install PyTorch for your hardware from
[pytorch.org](https://pytorch.org/get-started/locally/) first, then
`pip install -r requirements.txt`.

## Configuration

Every option is a flag or an environment variable; the variables are what the
compose files set.

| Flag | Variable | Default | |
| --- | --- | --- | --- |
| `--models` | `LAYA_SERVER_MODELS` | `english` | checkpoints to load, comma separated: `english`, `multilingual`, `typed-decisions` |
| `--jev-alias` | `LAYA_SERVER_JEV_ALIAS=1` | off natively, on in Docker | answer `jev-*` names as `laya` |
| `--reject-truncated` | `LAYA_SERVER_REJECT_TRUNCATED=1` | off | answer `422` rather than evaluate a state Laya would cut short |
| `--api-key` | `LAYA_SERVER_API_KEY` | none | require `Authorization: Bearer <key>` |
| `--device` | `LAYA_SERVER_DEVICE` | first discrete GPU, else CPU | a torch device, such as `cuda:1` or `cpu` |
| `--host` | `LAYA_SERVER_HOST` | `127.0.0.1` (`0.0.0.0` in Docker) | address to bind |
| `--port` | `LAYA_SERVER_PORT` | `8765` | port to listen on; in Docker, the host port |

Every checkpoint in `--models` is loaded before the server starts listening,
so no request waits on a download. Each holds 1.3 to 1.7 GB, so all three fit
in 8 GB of VRAM.

## Models

| `model` | Checkpoint | Context per question |
| --- | --- | --- |
| `laya` | Laya's router: English text to `english`, other scripts and languages to `multilingual` | |
| `laya-english` | ModernBERT-large, 421M parameters | 512 tokens |
| `laya-multilingual` | mmBERT-base, 322M, 100+ languages | 1024 tokens |
| `laya-typed-decisions` | ModernBERT-large, 421M, fine-tuned on four typed-decisions workflows | 1024 tokens |
| `jev-*` | the same as `laya`, with the Jev alias on | |

When the router picks a checkpoint that is not loaded, as it does for German
text with only `english` loaded, the request fails with a `422` that names the
`--models` value that would serve it.

## Using it from Rust

With the Jev alias on, `Client::from_env()` works unchanged:

```sh
TYPESAFE_BASE_URL=http://localhost:8765
TYPESAFE_API_KEY=local        # the crate requires one; the server ignores it unless LAYA_SERVER_API_KEY is set
TYPESAFE_DEFAULT_MODEL=laya   # optional with the alias, required without it
```

[`verify/`](verify) is a small crate that runs `typesafe-api` against a live
server: the model list, a typed evaluation read back through
`#[derive(Evaluation)]`, error handling, the Jev alias, and truncation
reporting.

```sh
cd verify
cargo test -- --nocapture                                       # http://127.0.0.1:8765
LAYA_SERVER_URL=http://127.0.0.1:9000 cargo test -- --nocapture # somewhere else
```

## Where it differs from TypeSafe

- **Laya reads only the start of the state.** Each question gets its own 512
  or 1024-token context, shared with the question and its options, so the
  state keeps roughly 320 to 480 tokens on `english` and 768 to 990 on the
  others. TypeSafe takes 32k. When a state is cut short the response carries
  `"truncation": {"state_tokens": N, "kept_tokens": {"<question id>": n}}` and
  an `x-laya-truncated` header listing the affected questions. TypeSafe
  clients ignore the field; `typesafe-api` users can read it with
  `Client::send_raw`, or have the server refuse with `--reject-truncated`.
- **The base checkpoints are weak zero-shot.** Laya's own
  [benchmarks](https://github.com/NandhaKishorM/laya#honest-limits) put them
  below the majority-class baseline on typed decisions; the gains come from
  fine-tuning. Check accuracy on your own data before relying on it.
- Answers carry Laya's extra fields (`action`, and `confidence` on a `noul`),
  and the response carries Laya's `routing` decision. TypeSafe clients ignore
  them.
- Missing `instructions` reach Laya as an empty string, and
  `usage.output_tokens` is always `0`.

## Performance

Measured with `english`, warm, with a four-option `choice` per question:

| Setup | 1 question | 5 questions | 10 questions |
| --- | --- | --- | --- |
| RX 7600, native Windows, ROCm, bf16 | 33 ms | 76 ms | 138 ms |
| Ryzen 5 9600X, Docker Desktop, CPU | 359 ms | 1.56 s | 3.07 s |

On a CPU, raise the client's timeout for requests with many questions: the
`typesafe-api` default is 10 seconds per attempt.

## Development

```sh
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
pip install -r requirements.txt -r requirements-dev.txt
ruff format . && ruff check . && pytest
```

The unit tests run Laya's real router and sequence builder against a stand-in
checkpoint, so they take a second and download nothing. CI also builds the
Docker image, waits for it to become healthy, and runs `verify/` against it.

## Licence

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your
option. Laya's code and weights are Apache-2.0, by Convai Innovations.
