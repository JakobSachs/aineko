"""Tests for text chunking."""

from hypothesis import given, settings
from hypothesis import strategies as st

from aineko.memory.chunker import chunk_text


class TestChunkTextBasics:
    def test_empty_string(self):
        assert chunk_text("") == []

    def test_whitespace_only(self):
        assert chunk_text("   \n\n\t  ") == []

    def test_short_text_single_chunk(self):
        text = "Hello, this is a short note."
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_text_under_min_size_discarded(self):
        text = "hi"
        assert chunk_text(text, min_size=50) == []

    def test_text_at_min_size_kept(self):
        text = "x" * 50
        chunks = chunk_text(text, min_size=50)
        assert len(chunks) == 1


class TestChunkTextBoundaries:
    def test_splits_at_paragraph_boundary(self):
        para1 = "First paragraph. " * 30  # ~510 chars
        para2 = "Second paragraph. " * 30
        text = para1.strip() + "\n\n" + para2.strip()
        chunks = chunk_text(text, chunk_size=600)
        # Should split at the \n\n rather than mid-word
        assert len(chunks) >= 2
        assert "First paragraph" in chunks[0]
        assert "Second paragraph" in chunks[-1]

    def test_falls_back_to_line_boundary(self):
        # No paragraph breaks, but line breaks exist
        lines = ["Line number " + str(i) + "." + " padding" * 5 for i in range(20)]
        text = "\n".join(lines)
        chunks = chunk_text(text, chunk_size=200)
        assert len(chunks) >= 2
        # No chunk should end mid-line (unless forced)
        for chunk in chunks[:-1]:  # last chunk is whatever remains
            assert chunk.endswith(".") or chunk.endswith("padding")

    def test_hard_cut_when_no_newlines(self):
        text = "x" * 2000  # no newlines at all
        chunks = chunk_text(text, chunk_size=800)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 800


class TestChunkTextOverlap:
    def test_consecutive_chunks_overlap(self):
        # Build text long enough for multiple chunks
        text = "word " * 500  # ~2500 chars
        chunks = chunk_text(text, chunk_size=800, overlap=100)
        assert len(chunks) >= 3
        for i in range(len(chunks) - 1):
            tail = chunks[i][-80:]  # end of current chunk
            head = chunks[i + 1][:120]  # start of next chunk
            # There should be some shared content
            # (exact overlap depends on boundary snapping, but some must exist)
            shared = set(tail.split()) & set(head.split())
            assert len(shared) > 0, f"No overlap between chunk {i} and {i+1}"

    def test_zero_overlap(self):
        text = "word " * 500
        chunks = chunk_text(text, chunk_size=800, overlap=0)
        assert len(chunks) >= 2
        # Total content should cover the original (no gaps)
        joined = " ".join(chunks)
        assert joined.count("word") >= 450  # allow some loss from boundary snapping


class TestChunkTextCustomParams:
    def test_small_chunk_size(self):
        text = "Hello world. " * 20
        chunks = chunk_text(text, chunk_size=100, overlap=10, min_size=10)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_large_chunk_size_single_chunk(self):
        text = "Hello world. " * 20
        chunks = chunk_text(text, chunk_size=10000)
        assert len(chunks) == 1

    def test_returns_list_of_strings(self):
        chunks = chunk_text("some text here")
        assert isinstance(chunks, list)
        for chunk in chunks:
            assert isinstance(chunk, str)


class TestChunkTextProperties:
    @given(
        content=st.text(min_size=0, max_size=5000),
        chunk_size=st.integers(min_value=100, max_value=2000),
        overlap=st.integers(min_value=0, max_value=200),
        min_size=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=50)
    def test_never_crashes(self, content, chunk_size, overlap, min_size):
        result = chunk_text(
            content, chunk_size=chunk_size, overlap=overlap, min_size=min_size
        )
        assert isinstance(result, list)
        for chunk in result:
            assert isinstance(chunk, str)

    @given(content=st.text(min_size=100, max_size=5000))
    @settings(max_examples=30)
    def test_chunks_respect_max_size(self, content):
        size = 300
        chunks = chunk_text(content, chunk_size=size, overlap=50)
        for chunk in chunks:
            assert len(chunk) <= size

    @given(content=st.text(min_size=100, max_size=3000))
    @settings(max_examples=30)
    def test_no_empty_chunks(self, content):
        chunks = chunk_text(content, chunk_size=400, min_size=20)
        for chunk in chunks:
            assert len(chunk.strip()) >= 20
