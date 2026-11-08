"""Unit tests for structured extraction schemas and extraction service."""

from unittest.mock import MagicMock

from localai.providers.base import GenerationResult
from localai.schemas.extraction import InvoiceSchema
from localai.services.extraction import ExtractionService


def test_invoice_schema_validation_success() -> None:
    data = {
        "vendor_name": "ABC Technologies",
        "invoice_number": "INV-2026-104",
        "invoice_date": "2026-11-04",
        "total_amount": 45000.0,
        "currency": "INR",
        "tax_or_gst": "19ABCDE1234F1Z5",
        "items": [
            {
                "description": "Cloud hosting services",
                "quantity": 1,
                "unit_price": 45000.0,
                "total_price": 45000.0,
            }
        ],
    }
    model = InvoiceSchema.model_validate(data)
    assert model.vendor_name == "ABC Technologies"
    assert model.total_amount == 45000.0
    assert len(model.items) == 1


def test_invoice_schema_missing_fields_defaults_to_none() -> None:
    # Partial data missing due_date, tax_or_gst, items
    data = {
        "vendor_name": "Acme Corp",
        "total_amount": 120.50,
    }
    model = InvoiceSchema.model_validate(data)
    assert model.vendor_name == "Acme Corp"
    assert model.total_amount == 120.50
    assert model.invoice_number is None
    assert model.items == []


def test_extraction_service_clean_json_from_code_blocks() -> None:
    service = ExtractionService(provider=MagicMock())
    raw_markdown = """Here is the extracted invoice:
```json
{
  "vendor_name": "Global Tech",
  "total_amount": 999.0
}
```
Hope this helps!"""
    cleaned = service._extract_json_substring(raw_markdown)
    assert cleaned == '{\n  "vendor_name": "Global Tech",\n  "total_amount": 999.0\n}'


def test_extraction_service_end_to_end_mocked() -> None:
    mock_provider = MagicMock()
    mock_provider.generate.return_value = GenerationResult(
        text='{"vendor_name": "Vertex AI Inc", "total_amount": 5000.0, "currency": "USD"}',
        model="llama3.2:3b",
    )

    service = ExtractionService(provider=mock_provider)
    result = service.extract(
        "Bill from Vertex AI Inc for USD 5000", schema_name="Invoice / Receipt"
    )

    assert result.is_valid is True
    assert result.data is not None
    assert result.data["vendor_name"] == "Vertex AI Inc"
    assert result.data["total_amount"] == 5000.0
    assert result.validation_errors == []
