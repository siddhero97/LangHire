"""
Multi-provider LLM factory.
Creates browser_use chat models from user settings.
Supports: OpenAI, Anthropic (direct), AWS Bedrock, Google Gemini.
"""


class _PatchedSession:
    """Wraps a boto3.Session to inject a BotoConfig into every client() call."""

    def __init__(self, session, config):
        self._session = session
        self._config = config

    def client(self, service_name, **kwargs):
        kwargs.setdefault("config", self._config)
        return self._session.client(service_name, **kwargs)

    def __getattr__(self, name):
        return getattr(self._session, name)


def create_llm(settings: dict):
    """Create a browser_use BaseChatModel from settings dict."""
    provider = settings.get("provider", "openai")

    if provider == "openai":
        from browser_use.llm import ChatOpenAI
        cfg = settings.get("openai", {})
        kwargs = {}
        max_tokens = cfg.get("max_completion_tokens") or settings.get("max_completion_tokens")
        if max_tokens:
            kwargs["max_completion_tokens"] = int(max_tokens)
        return ChatOpenAI(
            model=cfg.get("model", "gpt-4o"),
            api_key=cfg.get("api_key", "").strip(),
            **kwargs,
        )

    elif provider == "anthropic":
        from browser_use.llm import ChatAnthropic
        cfg = settings.get("anthropic", {})
        kwargs = {}
        max_tokens = cfg.get("max_completion_tokens") or cfg.get("max_tokens") or settings.get("max_completion_tokens")
        if max_tokens:
            kwargs["max_tokens"] = int(max_tokens)
        return ChatAnthropic(
            model=cfg.get("model", "claude-sonnet-4-5"),
            api_key=cfg.get("api_key", "").strip(),
            **kwargs,
        )

    elif provider == "bedrock":
        from browser_use.llm import ChatAWSBedrock
        import boto3
        from botocore.config import Config as BotoConfig
        cfg = settings.get("bedrock", {})
        region = cfg.get("region", "us-west-2")
        model = cfg.get("model", "us.anthropic.claude-sonnet-4-6")
        auth_mode = cfg.get("auth_mode", "profile")

        if auth_mode == "keys" and cfg.get("access_key") and cfg.get("secret_key"):
            session = boto3.Session(
                aws_access_key_id=cfg["access_key"],
                aws_secret_access_key=cfg["secret_key"],
                region_name=region,
            )
        else:
            session = boto3.Session(
                profile_name=cfg.get("profile_name", "default"),
                region_name=region,
            )

        # Disable request compression to avoid "Error -3 decompressing"
        # with Bedrock's converse API
        boto_config = BotoConfig(disable_request_compression=True)
        patched_session = _PatchedSession(session, boto_config)
        return ChatAWSBedrock(model=model, session=patched_session)

    elif provider == "ollama":
        # Use browser-use's native ChatOllama (not ChatOpenAI against the /v1 shim).
        from browser_use.llm import ChatOllama
        cfg = settings.get("ollama", {})
        base_url = cfg.get("base_url", "http://localhost:11434").rstrip("/")
        max_tokens = cfg.get("max_completion_tokens") or settings.get("max_completion_tokens") or 4096
        return ChatOllama(
            model=cfg.get("model") or "llama3.1",
            host=base_url,  # native Ollama host (NOT the /v1 OpenAI shim)
            timeout=300,
            ollama_options={
                "num_ctx": 32768,    # large context: agent prompt + DOM/screenshot is big
                "num_predict": int(max_tokens),  # room for full JSON output
                "temperature": 0.0,
            },
        )

    elif provider == "gemini":
        from browser_use.llm import ChatGoogle
        cfg = settings.get("gemini", {})
        kwargs = {}
        max_tokens = cfg.get("max_completion_tokens") or cfg.get("max_output_tokens") or settings.get("max_completion_tokens")
        if max_tokens:
            kwargs["max_output_tokens"] = int(max_tokens)
        return ChatGoogle(
            model=cfg.get("model", "gemini-3.8-flash"),
            api_key=cfg.get("api_key", "").strip(),
            **kwargs,
        )

    elif provider == "openrouter":
        from browser_use.llm import ChatOpenAI
        cfg = settings.get("openrouter", {})
        kwargs = {}
        max_tokens = cfg.get("max_completion_tokens") or settings.get("max_completion_tokens")
        if max_tokens:
            kwargs["max_completion_tokens"] = int(max_tokens)
        return ChatOpenAI(
            model=cfg.get("model", "google/gemini-2.5-flash"),
            api_key=cfg.get("api_key", "").strip(),
            base_url="https://openrouter.ai/api/v1",
            **kwargs,
        )

    elif provider == "openai_compatible":
        from browser_use.llm import ChatOpenAI
        cfg = settings.get("openai_compatible", {})
        kwargs = {}
        max_tokens = cfg.get("max_completion_tokens") or settings.get("max_completion_tokens")
        if max_tokens:
            kwargs["max_completion_tokens"] = int(max_tokens)
        return ChatOpenAI(
            model=cfg.get("model", "default"),
            api_key=cfg.get("api_key") or "not-needed",
            base_url=cfg.get("base_url"),
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


async def test_connection(llm) -> str:
    """Send a test message and return the response."""
    import asyncio
    import zlib
    from browser_use.llm.messages import UserMessage
    from browser_use.llm.exceptions import ModelProviderError
    try:
        # Increase timeout to 60s for local models
        response = await asyncio.wait_for(
            llm.ainvoke([UserMessage(content="Say 'hello' in one word.")]),
            timeout=60,
        )
        return f"Model responded: {response.completion}"
    except (asyncio.TimeoutError, TimeoutError):
        raise RuntimeError("LLM test timed out after 60 seconds. Your model might be slow to load or the server is busy.")
    except zlib.error:
        raise RuntimeError(
            "Decompression error communicating with AWS Bedrock. "
            "This is often caused by a corporate proxy or VPN. "
            "Try disabling your VPN or proxy, or use a direct connection."
        )
    except ModelProviderError as e:
        raise RuntimeError(_friendly_llm_error(getattr(e, "message", str(e)), getattr(e, "status_code", None)))
    except Exception as e:
        raise RuntimeError(_friendly_llm_error(str(e), getattr(e, "status_code", None)))


def _friendly_llm_error(message: str, status_code=None) -> str:
    """Map an LLM provider failure to a friendly, actionable message."""
    msg = (message or "").lower()
    code = status_code

    # Authentication / invalid API key
    if code in (401, 403) or any(
        s in msg for s in ("api_key", "api key", "authentication", "unauthorized", "invalid api", "invalid key")
    ):
        return "Invalid API key. Please check your key and try again."

    # Rate limit / quota
    if code == 429 or any(s in msg for s in ("rate limit", "ratelimit", "quota", "too many requests")):
        return "Rate limit or quota exceeded. Wait a moment and try again."

    # Connection / network
    if any(
        s in msg
        for s in ("connection", "connect", "network", "timed out", "timeout", "name resolution",
                  "could not resolve", "dns", "unreachable", "refused")
    ):
        return "Cannot reach the API. Check your internet connection and base URL."

    # Fall back to the raw provider message so the user still sees something useful.
    return message or "LLM request failed."
