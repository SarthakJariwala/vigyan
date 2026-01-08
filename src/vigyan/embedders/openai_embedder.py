from lancedb.embeddings import get_registry

from ..core.interfaces import Embedder


class OpenAIEmbedder(Embedder):
    """Embedder backed by LanceDB's OpenAI registry provider.

    Requires an OpenAI-compatible API key in the environment.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-large",
        dim: int | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
    ) -> None:
        # Create via LanceDB registry to avoid hard dependency on openai package
        reg = get_registry().get("openai")
        kwargs = {"name": model}
        if dim is not None:
            kwargs["dim"] = dim
        if base_url:
            kwargs["base_url"] = base_url
        if api_key_env:
            kwargs["api_key_env"] = api_key_env
        self._fn = reg.create(**kwargs)
        self._model_name = model

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._fn.ndims()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._fn.generate_embeddings(texts)
