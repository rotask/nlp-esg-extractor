from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from nlp_esg.types import KPIExtraction


class Extractor(ABC):
    name: str = "base"

    @abstractmethod
    def extract(self, report: Any, kpi_key: str) -> KPIExtraction:
        ...
