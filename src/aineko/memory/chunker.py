"""Text chunking for memory storage.

Splits text into overlapping chunks, preferring natural boundaries
(paragraph breaks, then line breaks) over hard cuts.
"""

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MIN_CHUNK_SIZE = 20


def chunk_text(
    content: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    min_size: int = MIN_CHUNK_SIZE,
) -> list[str]:
    """Split text into chunks on natural boundaries.

    Priority: paragraph boundary (\\n\\n) > line boundary (\\n) > hard cut.
    Consecutive chunks overlap by ``overlap`` characters for continuity.

    Returns empty list for empty / whitespace-only input.
    Chunks shorter than ``min_size`` after stripping are discarded.
    """
    if not content or not content.strip():
        return []

    chunks: list[str] = []
    start = 0
    length = len(content)

    while start < length:
        end = min(start + chunk_size, length)

        if end < length:
            # Try paragraph boundary first
            para = content.rfind("\n\n", start + chunk_size // 2, end)
            if para != -1:
                end = para
            else:
                # Fall back to line boundary
                line = content.rfind("\n", start + chunk_size // 2, end)
                if line != -1:
                    end = line

        chunk = content[start:end].strip()
        if len(chunk) >= min_size:
            chunks.append(chunk)

        if end >= length:
            break
        start = max(end - overlap, start + 1)  # +1 to guarantee progress

    return chunks
