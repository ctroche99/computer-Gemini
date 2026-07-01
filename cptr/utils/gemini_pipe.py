"""Vertex / Gemini PIPE adapter for cptr.

A self-contained "tack-on" that lets Computer talk to Google Gemini — either
through **Vertex AI** (service-account auth, project + location) or the Google
Generative Language API (API key). It ports only the core message-calling logic
of the Open WebUI Google Gemini pipe (google.genai client + streaming) and
speaks Computer's existing normalized streaming-event protocol, so it plugs into
the agentic loop exactly like the anthropic/openai adapters in ``cptr.utils.ai``.

Everything Gemini-specific lives in this module. The rest of the codebase only
adds ``provider == "vertex"`` branches, keeping upstream merges clean.

A "vertex" connection dict looks like::

    {
        "provider": "vertex",
        "api_key": "<encrypted service-account JSON or Google API key>",
        "enabled": True,               # the on/off "toggle"
        "prefix_id": "vertex",
        "data": {
            "auth_mode": "vertex",      # "vertex" | "api_key"
            "vertex_project": "my-gcp-project",
            "vertex_location": "us-central1",
            "thinking_budget": -1,      # optional: -1 dynamic, 0 off, N tokens
            "include_thoughts": False,  # optional
            "models": ["gemini-2.5-pro", "gemini-2.5-flash"],
        },
    }

The ``api_key`` field is decrypted by the caller before being handed here (the
adapter receives the *connection* dict and pulls the already-decrypted secret
from ``connection["_api_key_plain"]`` when present, else ``connection["api_key"]``).
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

logger = logging.getLogger(__name__)

# Scope required for Vertex AI when building credentials from a service account.
_VERTEX_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# Fallback model list when auto-discovery isn't possible / configured.
_DEFAULT_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]

_INSTALL_HINT = (
    "The Vertex/Gemini pipe requires the 'google-genai' package. "
    "Install it with: pip install 'cptr[vertex]'"
)


def _import_genai():
    """Lazy-import google-genai so base cptr installs don't need the dependency."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:  # pragma: no cover - exercised only without the dep
        raise RuntimeError(_INSTALL_HINT) from e
    return genai, types


# ── Connection helpers ───────────────────────────────────────


def _conn_data(connection: dict) -> dict:
    data = connection.get("data")
    return data if isinstance(data, dict) else {}


def _secret(connection: dict) -> str:
    """Return the decrypted secret (service-account JSON or API key).

    Callers decrypt the ``api_key`` field and stash the plaintext under
    ``_api_key_plain``; fall back to the raw field for direct/testing use.
    """
    return (connection.get("_api_key_plain") or connection.get("api_key") or "").strip()


def build_client(connection: dict):
    """Build a genai.Client for a vertex connection (Vertex or API-key auth)."""
    genai, _ = _import_genai()
    data = _conn_data(connection)
    auth_mode = data.get("auth_mode") or "vertex"
    secret = _secret(connection)

    if auth_mode == "vertex":
        project = data.get("vertex_project")
        location = data.get("vertex_location") or "global"
        if not project:
            raise ValueError(
                "Vertex project ID is not set. Set it in Settings ▸ Vertex."
            )
        credentials = _resolve_vertex_credentials(secret)
        logger.debug("[vertex] client project=%s location=%s", project, location)
        return genai.Client(
            vertexai=True,
            project=project,
            location=location,
            credentials=credentials,
        )

    # API-key mode (Google Generative Language API)
    if not secret:
        raise ValueError("Google API key is not set. Set it in Settings ▸ Vertex.")
    return genai.Client(api_key=secret)


def _looks_like_sa_json(value: str) -> bool:
    """Heuristic: does this string look like service-account key JSON?"""
    return value.startswith("{") or (
        '"private_key"' in value
        or '"client_email"' in value
        or "service_account" in value
    )


def _resolve_vertex_credentials(secret: str):
    """Turn a pasted service-account key into credentials.

    Accepts the full JSON key contents (preferred) or a filesystem path to a JSON
    key file. Empty → rely on ambient Application Default Credentials (e.g. the
    GCE/GKE metadata server).

    Credentials are detected by *content*, not just a leading brace, so a paste
    that dropped its opening ``{`` still parses instead of being mistaken for a
    file path. A secret blob is NEVER written to GOOGLE_APPLICATION_CREDENTIALS,
    so it can never be echoed back through an ADC "file not found" error.
    """
    if not secret:
        return None

    stripped = secret.strip()

    if _looks_like_sa_json(stripped):
        from google.oauth2 import service_account

        try:
            info = json.loads(stripped)
        except json.JSONDecodeError:
            # Tolerate a paste that lost its outer braces; validate the retry.
            candidate = stripped
            if not candidate.startswith("{"):
                candidate = "{" + candidate
            if not candidate.rstrip().endswith("}"):
                candidate = candidate + "}"
            try:
                info = json.loads(candidate)
            except json.JSONDecodeError as e:
                raise ValueError(
                    "Service account credentials are not valid JSON. Paste the "
                    "complete key contents, including the surrounding { }."
                ) from e
        return service_account.Credentials.from_service_account_info(
            info, scopes=_VERTEX_SCOPES
        )

    # Not JSON → only accept a genuine single-line path to an existing file.
    import os

    if "\n" not in stripped and os.path.isfile(stripped):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = stripped
        return None

    raise ValueError(
        "Service account credentials could not be read: value is neither valid "
        "JSON nor a path to an existing key file. Paste the full service-account "
        "JSON (including the surrounding { }), or provide a valid file path."
    )


# ── Message / tool conversion ────────────────────────────────


def _extract_system(messages: list[dict]) -> str:
    parts: list[str] = []
    for m in messages:
        if m.get("role") == "system":
            content = m.get("content", "")
            if isinstance(content, str) and content:
                parts.append(content)
    return "\n\n".join(parts)


def _decode_thought_signature(fc_id: str) -> bytes | None:
    """Decode a Gemini 3 thought_signature stashed in a tool_call's fc_id.

    We store signatures as ``vsig_<standard-base64>``. Returns the raw bytes, or
    None when the field holds something else (empty, or a foreign id such as an
    OpenAI ``fc_...`` from a model switched mid-chat).
    """
    if not isinstance(fc_id, str) or not fc_id.startswith("vsig_"):
        return None
    try:
        import base64

        return base64.standard_b64decode(fc_id[len("vsig_") :])
    except Exception:
        return None


def _call_id_to_name(messages: list[dict]) -> dict[str, str]:
    """Map assistant tool-call ids → function names (Gemini matches by name)."""
    mapping: dict[str, str] = {}
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                cid = tc.get("id", "")
                name = (tc.get("function") or {}).get("name", "")
                if cid and name:
                    mapping[cid] = name
    return mapping


def _text_and_images(content: Any, types) -> list:
    """Convert a canonical content value into a list of genai Parts."""
    parts: list = []
    if isinstance(content, str):
        if content:
            parts.append(types.Part.from_text(text=content))
        return parts
    if isinstance(content, list):
        for block in content:
            btype = block.get("type")
            if btype == "text":
                text = block.get("text", "")
                if text:
                    parts.append(types.Part.from_text(text=text))
            elif btype == "image":
                import base64

                raw = base64.b64decode(block.get("base64", ""))
                parts.append(
                    types.Part.from_bytes(
                        data=raw, mime_type=block.get("media_type", "image/jpeg")
                    )
                )
    return parts


def to_genai_contents(messages: list[dict], types) -> list:
    """Canonical messages → list[types.Content] (system handled separately)."""
    call_names = _call_id_to_name(messages)
    contents: list = []
    pending_tool_parts: list = []  # merge contiguous tool results into one turn

    def flush_tools():
        nonlocal pending_tool_parts
        if pending_tool_parts:
            contents.append(types.Content(role="user", parts=pending_tool_parts))
            pending_tool_parts = []

    for m in messages:
        role = m.get("role")
        if role == "system":
            continue

        if role == "tool":
            name = call_names.get(m.get("tool_call_id", ""), "") or "tool"
            content = m.get("content", "")
            if isinstance(content, list):
                text_parts = [
                    b.get("text", "") for b in content if b.get("type") == "text"
                ]
                response_val: Any = "\n".join(text_parts)
            else:
                response_val = content
            pending_tool_parts.append(
                types.Part.from_function_response(
                    name=name, response={"result": response_val}
                )
            )
            continue

        flush_tools()

        if role == "assistant" and m.get("tool_calls"):
            parts: list = []
            text = m.get("content", "")
            if isinstance(text, str) and text:
                parts.append(types.Part.from_text(text=text))
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                raw_args = fn.get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    args = {}
                # Restore the Gemini 3 thought_signature captured in fc_id (if any)
                # so the replayed functionCall part is accepted; plain call otherwise.
                sig = _decode_thought_signature(tc.get("fc_id", ""))
                if sig is not None:
                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(
                                name=fn.get("name", ""), args=args or {}
                            ),
                            thought_signature=sig,
                        )
                    )
                else:
                    parts.append(
                        types.Part.from_function_call(name=fn.get("name", ""), args=args or {})
                    )
            contents.append(types.Content(role="model", parts=parts))
            continue

        # Plain user / assistant text (and multimodal user content)
        genai_role = "model" if role == "assistant" else "user"
        parts = _text_and_images(m.get("content", ""), types)
        if parts:
            contents.append(types.Content(role=genai_role, parts=parts))

    flush_tools()
    return contents


def _sanitize_schema(schema: Any) -> Any:
    """Drop JSON-schema keys Gemini's FunctionDeclaration rejects."""
    if isinstance(schema, dict):
        return {
            k: _sanitize_schema(v)
            for k, v in schema.items()
            if k not in ("additionalProperties", "$schema", "title")
        }
    if isinstance(schema, list):
        return [_sanitize_schema(v) for v in schema]
    return schema


def to_genai_tools(tools: list[dict], types) -> list | None:
    if not tools:
        return None
    declarations = []
    for t in tools:
        declarations.append(
            types.FunctionDeclaration(
                name=t["name"],
                description=t.get("description", ""),
                parameters=_sanitize_schema(t.get("parameters", {})) or None,
            )
        )
    return [types.Tool(function_declarations=declarations)]


def _build_config(connection: dict, system: str, tools: list, request_params: dict | None, types):
    data = _conn_data(connection)
    rp = request_params or {}

    config_kwargs: dict[str, Any] = {}
    if system:
        config_kwargs["system_instruction"] = system
    tool_objs = to_genai_tools(tools, types)
    if tool_objs:
        config_kwargs["tools"] = tool_objs
    if rp.get("temperature") is not None:
        config_kwargs["temperature"] = rp["temperature"]
    max_out = rp.get("max_output_tokens") or rp.get("max_tokens") or rp.get("max_completion_tokens")
    if max_out:
        config_kwargs["max_output_tokens"] = max_out

    # Optional thinking config (Gemini 2.5 models)
    thinking_budget = rp.get("thinking_budget", data.get("thinking_budget"))
    include_thoughts = data.get("include_thoughts", False)
    if thinking_budget is not None or include_thoughts:
        tc_kwargs: dict[str, Any] = {}
        if thinking_budget is not None:
            try:
                tc_kwargs["thinking_budget"] = int(thinking_budget)
            except (TypeError, ValueError):
                pass
        if include_thoughts:
            tc_kwargs["include_thoughts"] = True
        if tc_kwargs:
            config_kwargs["thinking_config"] = types.ThinkingConfig(**tc_kwargs)

    return types.GenerateContentConfig(**config_kwargs)


def _reasoning_item(text: str) -> dict:
    return {
        "type": "reasoning",
        "id": f"rs_{uuid.uuid4().hex}",
        "status": "completed",
        "content": [{"type": "text", "text": text}],
    }


# ── Streaming (agentic loop) ─────────────────────────────────


async def stream_gemini(
    form_data,
    connection: dict,
    *,
    request_params: dict | None = None,
) -> AsyncIterator[dict]:
    """Stream a Gemini response as normalized cptr events.

    Yields the same event shapes as the anthropic/openai adapters:
    ``text_delta``, ``tool_call``, ``output`` (reasoning), ``usage``, ``done``.
    ``form_data`` is a ``cptr.utils.ai.ChatCompletionForm``.
    """
    _, types = _import_genai()
    client = build_client(connection)

    contents = to_genai_contents(form_data.messages, types)
    config = _build_config(
        connection, form_data.instructions, form_data.tools, request_params, types
    )

    logger.info(
        "[stream] vertex/gemini model=%s contents=%d tools=%d",
        form_data.model,
        len(contents),
        len(form_data.tools),
    )

    stream = await client.aio.models.generate_content_stream(
        model=form_data.model, contents=contents, config=config
    )

    # Gemini streams usage_metadata cumulatively on EVERY chunk. The agentic loop
    # treats a "usage" event with no pending tool calls as end-of-turn (save +
    # return), so emitting it per chunk truncates plain text replies to the first
    # token. Track the latest usage and emit it exactly once, after the stream —
    # matching the anthropic/openai adapters.
    input_tokens = 0
    output_tokens = 0
    async for chunk in stream:
        candidates = getattr(chunk, "candidates", None) or []
        if candidates:
            content = getattr(candidates[0], "content", None)
            for part in (getattr(content, "parts", None) or []):
                fn = getattr(part, "function_call", None)
                if fn is not None:
                    args = dict(getattr(fn, "args", None) or {})
                    # Gemini 3 attaches a thought_signature (bytes) to each
                    # functionCall part; it MUST be echoed back when the call is
                    # replayed in history or the next turn is rejected with 400.
                    # Stash it (base64) in the event "id", which the agentic loop
                    # persists as the tool_call's fc_id — so it round-trips.
                    sig = getattr(part, "thought_signature", None)
                    event_id = ""
                    if sig:
                        import base64

                        event_id = "vsig_" + base64.standard_b64encode(sig).decode()
                    yield {
                        "type": "tool_call",
                        "call_id": f"call_{uuid.uuid4().hex[:24]}",
                        "id": event_id,
                        "name": fn.name,
                        "arguments": args,
                    }
                    continue

                text = getattr(part, "text", None)
                if text:
                    if getattr(part, "thought", False):
                        yield {"type": "output", "item": _reasoning_item(text)}
                    else:
                        yield {"type": "text_delta", "content": text}

        usage = getattr(chunk, "usage_metadata", None)
        if usage is not None:
            # Values are cumulative; keep the most recent non-zero readings.
            input_tokens = getattr(usage, "prompt_token_count", 0) or input_tokens
            output_tokens = getattr(usage, "candidates_token_count", 0) or output_tokens

    yield {"type": "usage", "input_tokens": input_tokens, "output_tokens": output_tokens}
    yield {"type": "done"}


# ── Non-streaming (utility tasks: titles, tags, summaries) ───


async def gemini_completion(
    connection: dict,
    model: str,
    messages: list[dict],
    system: str = "",
    max_tokens: int = 200,
) -> str:
    """Simple non-streaming completion. Returns text content."""
    _, types = _import_genai()
    client = build_client(connection)

    contents = to_genai_contents(messages, types)
    config_kwargs: dict[str, Any] = {"max_output_tokens": max_tokens}
    if system:
        config_kwargs["system_instruction"] = system
    config = types.GenerateContentConfig(**config_kwargs)

    resp = await client.aio.models.generate_content(
        model=model, contents=contents, config=config
    )
    return getattr(resp, "text", "") or ""


# ── Model discovery ──────────────────────────────────────────


async def list_vertex_models(connection: dict) -> list[str]:
    """Return configured models, else discover from the API, else defaults."""
    configured = _conn_data(connection).get("models")
    if configured:
        return list(configured)

    try:
        client = build_client(connection)
        models: list[str] = []
        async for m in await client.aio.models.list():
            name = getattr(m, "name", "") or ""
            # Names come back like "models/gemini-2.5-pro" or "publishers/google/..."
            short = name.split("/")[-1]
            actions = getattr(m, "supported_actions", None) or getattr(
                m, "supported_generation_methods", None
            ) or []
            if short.startswith("gemini") and (not actions or "generateContent" in actions):
                models.append(short)
        if models:
            return sorted(set(models))
    except Exception as e:  # discovery is best-effort
        # Log only the exception type — messages can embed credential text.
        logger.warning("[vertex] model discovery failed: %s", type(e).__name__)

    return list(_DEFAULT_MODELS)


async def verify_vertex(connection: dict) -> tuple[bool, str]:
    """Authenticate against the API for real. Returns (ok, message)."""
    try:
        client = build_client(connection)
    except Exception as e:
        return False, str(e)
    try:
        count = 0
        async for _ in await client.aio.models.list():
            count += 1
            if count >= 1:
                break
        return True, "Connected"
    except Exception as e:
        return False, str(e)
