"""Registered job step sets, keyed by job kind.

1.4-1.6 register their processing steps here as they land; the runner refuses
jobs whose kind has no registered step set.
"""

from __future__ import annotations

from core.jobs.runner import StepDefinition

STEP_SETS: dict[str, tuple[StepDefinition, ...]] = {}
