"""Pydantic schemas for structured data extraction."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class InvoiceItem(BaseModel):
    description: str = Field(description="Item or service description")
    quantity: Optional[float] = Field(default=1.0, description="Quantity of items")
    unit_price: Optional[float] = Field(default=None, description="Price per individual unit")
    total_price: Optional[float] = Field(default=None, description="Total line item amount")


class InvoiceSchema(BaseModel):
    """Schema for extracting financial invoice and billing metadata."""

    vendor_name: Optional[str] = Field(
        default=None, description="Name of company or vendor billing"
    )
    invoice_number: Optional[str] = Field(
        default=None, description="Invoice or reference identifier"
    )
    invoice_date: Optional[str] = Field(
        default=None, description="Date of the invoice (YYYY-MM-DD preferred)"
    )
    due_date: Optional[str] = Field(default=None, description="Payment due date")
    total_amount: Optional[float] = Field(default=None, description="Total invoice amount payable")
    currency: Optional[str] = Field(default="USD", description="Currency symbol or 3-letter code")
    tax_or_gst: Optional[str] = Field(
        default=None, description="Tax identifier, VAT, or GST number"
    )
    items: List[InvoiceItem] = Field(
        default_factory=list, description="List of invoiced line items"
    )
    payment_terms: Optional[str] = Field(default=None, description="Terms of payment, e.g., Net 30")


class ResumeSchema(BaseModel):
    """Schema for candidate profile and resume data extraction."""

    candidate_name: Optional[str] = Field(default=None, description="Full name of applicant")
    email: Optional[str] = Field(default=None, description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number")
    location: Optional[str] = Field(default=None, description="City, country or region")
    summary: Optional[str] = Field(default=None, description="Professional headline or summary")
    skills: List[str] = Field(default_factory=list, description="Core technical and domain skills")
    years_of_experience: Optional[float] = Field(
        default=None, description="Estimated total years of experience"
    )
    latest_role: Optional[str] = Field(default=None, description="Most recent job title or company")
    education_degree: Optional[str] = Field(
        default=None, description="Highest degree or field of study"
    )


class MeetingNotesSchema(BaseModel):
    """Schema for extracting key takeaways and action items from meeting notes."""

    title: Optional[str] = Field(default="Meeting Notes", description="Topic or meeting title")
    date: Optional[str] = Field(default=None, description="Date meeting occurred")
    attendees: List[str] = Field(default_factory=list, description="Participants in the discussion")
    key_decisions: List[str] = Field(
        default_factory=list, description="Agreements or decisions confirmed"
    )
    action_items: List[str] = Field(
        default_factory=list, description="Tasks assigned with assignees if noted"
    )
    next_steps: Optional[str] = Field(
        default=None, description="Follow-up timeline or next meeting"
    )


class ExtractionEnvelope(BaseModel):
    """Wrapper encapsulating extraction result, validation status, and telemetry."""

    schema_name: str
    is_valid: bool
    data: Optional[Dict[str, Any]] = None
    validation_errors: List[str] = Field(default_factory=list)
    raw_response: str = ""
    model_used: str = ""
    latency_ms: float = 0.0
