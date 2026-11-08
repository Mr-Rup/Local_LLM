"""Configuration management for LocalAI.

Loads default configuration from config/settings.toml with optional
environment variable overrides.
"""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    name: str = "LocalAI — Private LLM Workbench"
    version: str = "0.1.0"
    environment: str = "development"


class OllamaConfig(BaseModel):
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: int = 120
    default_model: str = "llama3.2:3b"
    default_temperature: float = 0.7
    default_max_tokens: int = 2048
    default_top_p: float = 0.9


class DocumentsConfig(BaseModel):
    upload_dir: Path = Path("data/documents")
    max_file_size_mb: int = 25
    supported_extensions: List[str] = [".pdf", ".docx", ".txt"]
    default_chunk_size: int = 500
    default_chunk_overlap: int = 50
    default_top_k: int = 3


class ExtractionConfig(BaseModel):
    max_input_length: int = 8000
    temperature: float = 0.1


class BenchmarkingConfig(BaseModel):
    output_dir: Path = Path("data/benchmarks")
    default_repetitions: int = 3
    warmup_runs: int = 1
    default_temperature: float = 0.0


class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    documents: DocumentsConfig = Field(default_factory=DocumentsConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    benchmarking: BenchmarkingConfig = Field(default_factory=BenchmarkingConfig)


def find_config_file() -> Optional[Path]:
    """Find the settings.toml file searching from current directory up to workspace root."""
    current = Path.cwd()
    for directory in [current, *current.parents]:
        candidate = directory / "config" / "settings.toml"
        if candidate.is_file():
            return candidate
    return None


@lru_cache()
def get_settings() -> Settings:
    """Load settings from settings.toml and overlay environment variables."""
    config_dict: dict = {}
    config_path = find_config_file()

    if config_path and config_path.is_file():
        with open(config_path, "rb") as f:
            config_dict = tomllib.load(f)

    # Allow environment variable overrides
    if "OLLAMA_BASE_URL" in os.environ:
        config_dict.setdefault("ollama", {})["base_url"] = os.environ["OLLAMA_BASE_URL"]
    if "DEFAULT_MODEL" in os.environ:
        config_dict.setdefault("ollama", {})["default_model"] = os.environ["DEFAULT_MODEL"]
    if "APP_ENV" in os.environ:
        config_dict.setdefault("app", {})["environment"] = os.environ["APP_ENV"]

    settings = Settings(**config_dict)

    # Ensure local data directories exist
    settings.documents.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.benchmarking.output_dir.mkdir(parents=True, exist_ok=True)

    return settings
