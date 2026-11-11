from app.services.ingestion.chunker import chunk_segments, make_chunk_id
from app.services.ingestion.parsers import Segment

SENTENCES = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(60))


def _chunk(segments, size=200, overlap=40, checksum="c" * 64):
    return chunk_segments(segments, checksum, size, overlap)


def test_chunks_respect_size_and_are_never_empty():
    chunks = _chunk([Segment(SENTENCES)])
    assert len(chunks) > 3
    assert all(0 < len(c.content) <= 200 for c in chunks)
    assert all(c.content.strip() for c in chunks)


def test_ids_and_indexes_are_deterministic_and_checksum_scoped():
    a = _chunk([Segment(SENTENCES)])
    b = _chunk([Segment(SENTENCES)])
    assert [c.id for c in a] == [c.id for c in b]
    assert [c.chunk_index for c in a] == list(range(len(a)))
    assert a[0].id == make_chunk_id("c" * 64, 0)
    other = _chunk([Segment(SENTENCES)], checksum="d" * 64)
    assert {c.id for c in a}.isdisjoint({c.id for c in other})


def test_overlap_is_present_and_bounded():
    chunks = _chunk([Segment(SENTENCES)], size=200, overlap=60)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        shared = prev.char_end - nxt.char_start
        assert 0 < shared <= 60
        assert nxt.content[: shared // 2] in prev.content  # shared text really repeats


def test_zero_overlap_has_no_shared_text():
    chunks = _chunk([Segment(SENTENCES)], size=200, overlap=0)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.char_start >= prev.char_end


def test_all_text_is_covered():
    chunks = _chunk([Segment(SENTENCES)])
    joined = " ".join(c.content for c in chunks)
    for i in range(60):
        assert f"Sentence number {i} " in joined


def test_prefers_paragraph_boundaries():
    p1, p2 = "A" * 5 + " first paragraph text here.", "B" * 5 + " second paragraph text here."
    chunks = _chunk([Segment(f"{p1}\n\n{p2}")], size=40, overlap=0)
    assert [c.content for c in chunks] == [p1, p2]


def test_prefers_sentence_boundaries_over_midword_cuts():
    text = "First sentence is here. Second sentence follows it. Third sentence ends."
    chunks = _chunk([Segment(text)], size=40, overlap=0)
    assert all(c.content.endswith(".") for c in chunks)


def test_oversized_token_is_hard_split():
    token = "".join(f"{i:03d}" for i in range(150))  # 450 chars, no spaces, no repeats
    chunks = _chunk([Segment(token)], size=200, overlap=0)
    assert [len(c.content) for c in chunks] == [200, 200, 50]
    assert "".join(c.content for c in chunks) == token


def test_page_and_section_metadata_preserved_and_segments_not_merged():
    segs = [
        Segment("Alpha content on page one.", page_number=1),
        Segment("Beta content on page two.", page_number=2, section="Intro"),
    ]
    chunks = _chunk(segs, size=500, overlap=0)
    assert [(c.page_number, c.section) for c in chunks] == [(1, None), (2, "Intro")]
    assert [c.chunk_index for c in chunks] == [0, 1]


def test_duplicate_chunks_are_dropped_and_indexes_stay_contiguous():
    segs = [
        Segment("Repeated header text.", page_number=1),
        Segment("repeated   HEADER text.", page_number=2),
        Segment("Unique body.", page_number=3),
    ]
    chunks = _chunk(segs, size=500, overlap=0)
    assert [c.page_number for c in chunks] == [1, 3]
    assert [c.chunk_index for c in chunks] == [0, 1]


def test_empty_input_yields_no_chunks():
    assert _chunk([]) == []
    assert _chunk([Segment("")]) == []
