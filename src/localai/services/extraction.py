"""Structured extraction service powered by Pydantic schema validation."""

from __future__ import annotations

import json
import re
import time
from typing import Dict, List, Optional, Type

from pydantic import BaseModel, ValidationError

from localai.config import get_settings
from localai.providers.base import BaseLLMProvider, GenerationOptions
from localai.providers.ollama import OllamaProvider
from localai.schemas.extraction import (
    ExtractionEnvelope,
    InvoiceSchema,
    MeetingNotesSchema,
    ResumeSchema,
)

AVAILABLE_SCHEMAS: Dict[str, Type[BaseModel]] = {
    "Invoice / Receipt": InvoiceSchema,
    "Candidate Resume": ResumeSchema,
    "Meeting Notes": MeetingNotesSchema,
}


class ExtractionService:
    """Extracts typed, validated JSON structures from arbitrary unstructured text."""

    def __init__(self, provider: Optional[BaseLLMProvider] = None) -> None:
        self.provider = provider or OllamaProvider()
        self.settings = get_settings()

    def get_available_schemas(self) -> List[str]:
        """Return list of supported schema names."""
        return list(AVAILABLE_SCHEMAS.keys())

    def extract(
        self,
        text: str,
        schema_name: str,
        model: Optional[str] = None,
    ) -> ExtractionEnvelope:
        """Extract structured data adhering to the specified schema."""
        if schema_name not in AVAILABLE_SCHEMAS:
            raise ValueError(
                f"Unknown schema: {schema_name}. Available: {list(AVAILABLE_SCHEMAS.keys())}"
            )

        schema_cls = AVAILABLE_SCHEMAS[schema_name]
        schema_json_spec = json.dumps(schema_cls.model_json_schema(), indent=2)

        system_prompt = (
            "You are a strict data extraction engine. "
            "Your task is to extract information from the user's text and structure it into a JSON object "
            "that strictly conforms to the following JSON Schema:\n"
            f"{schema_json_spec}\n\n"
            "Rules:\n"
            "1. Output ONLY the raw JSON object. Do not include introductory text, explanations, or markdown code fences.\n"
            "2. If a field cannot be determined or is not present in the text, use null (or empty array [] for lists).\n"
            "3. Do not invent, hallucinate, or extrapolate facts not present in the source text."
        )

        prompt = f'Source Text to extract from:\n"""\n{text.strip()}\n"""\n\nExtracted JSON:'

        options = GenerationOptions(
            temperature=self.settings.extraction.temperature,
            format="json",
            max_tokens=2048,
        )

        target_model = model or self.settings.ollama.default_model
        t0 = time.perf_counter()

        gen_result = self.provider.generate(
            prompt=prompt,
            model=target_model,
            options=options,
            system=system_prompt,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        raw_output = gen_result.text.strip()
        cleaned_json = self._extract_json_substring(raw_output)

        try:
            parsed_dict = json.loads(cleaned_json)
        except json.JSONDecodeError as exc:
            return ExtractionEnvelope(
                schema_name=schema_name,
                is_valid=False,
                data=None,
                validation_errors=[f"JSON Syntax Error: {exc}"],
                raw_response=raw_output,
                model_used=target_model,
                latency_ms=round(elapsed_ms, 1),
            )

        # Validate with Pydantic
        try:
            validated_obj = schema_cls.model_validate(parsed_dict)
            return ExtractionEnvelope(
                schema_name=schema_name,
                is_valid=True,
                data=validated_obj.model_dump(),
                validation_errors=[],
                raw_response=raw_output,
                model_used=target_model,
                latency_ms=round(elapsed_ms, 1),
            )
        except ValidationError as val_err:
            errors = [f"{err['loc']}: {err['msg']}" for err in val_err.errors()]
            return ExtractionEnvelope(
                schema_name=schema_name,
                is_valid=False,
                data=parsed_dict if isinstance(parsed_dict, dict) else None,
                validation_errors=errors,
                raw_response=raw_output,
                model_used=target_model,
                latency_ms=round(elapsed_ms, 1),
            )

    def _extract_json_substring(self, text: str) -> str:
        """Strip markdown fences or extraneous text surrounding JSON."""
        # Check for ```json ... ``` or ``` ... ```
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip()

        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1].strip()

        return text
