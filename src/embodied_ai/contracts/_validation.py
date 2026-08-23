"""Shared validation helpers for dependency-light contracts."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def require_sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence")
    return value


def require_tuple(value: object, name: str) -> tuple[Any, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"{name} must be a tuple to preserve contract immutability")
    return value


def require_non_empty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without surrounding whitespace")
    return value


def require_identifier(value: object, name: str) -> str:
    text = require_non_empty(value, name)
    if not _IDENTIFIER_RE.fullmatch(text):
        raise ValueError(
            f"{name} must use lowercase letters, numbers, '.', '_' or '-' separators"
        )
    return text


def require_schema_version(value: object, expected: str) -> None:
    if value != expected:
        raise ValueError(f"unsupported schema version {value!r}; expected {expected!r}")


def require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def require_non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def require_positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def require_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def require_relative_posix_path(value: object, name: str) -> str:
    text = require_non_empty(value, name)
    path = PurePosixPath(text)
    if (
        "\\" in text
        or path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or text != path.as_posix()
    ):
        raise ValueError(f"{name} must be a normalized relative POSIX path")
    return text


def require_sha256(value: object, name: str) -> str:
    text = require_non_empty(value, name)
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest")
    return text
