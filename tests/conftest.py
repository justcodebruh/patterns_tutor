"""Shared fixtures. Tests never send traces, even if LANGSMITH_TRACING is set in the shell."""

import pytest
from langsmith import tracing_context


@pytest.fixture(autouse=True)
def no_langsmith_tracing():
    with tracing_context(enabled=False):
        yield
