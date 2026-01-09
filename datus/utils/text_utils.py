# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import re
import unicodedata


def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return text

    text = unicodedata.normalize("NFKC", text)

    # 2. Replace common invisible whitespace
    # NBSP  # zero width  # BOM
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")

    # 3.Remove control characters(retain \n \t)
    text = re.sub(r"[\x00-\x08\x0B-\x1F\x7F]", "", text)

    # 4. Uniform line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    return text.strip()


def strip_markdown_code_block(text: str) -> str:
    """
    Strip markdown code block markers from text if present at start/end.

    Args:
        text: Input text containing code block with markdown markers

    Returns:
        Text with markdown markers removed
    """
    if not isinstance(text, str):
        return text

    lines = text.strip().split("\n")
    if not lines:
        return text

    # Remove ```<lang> at start if present
    if lines[0].strip().startswith("```"):
        lines = lines[1:]

    # Remove ``` at end if present
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    return "\n".join(lines)
