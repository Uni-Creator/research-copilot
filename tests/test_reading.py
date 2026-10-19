import numpy as np
import pytest

from copilot.graph.state import Chunk
from copilot.tools.chunking import abstract_chunk_id, chunk_pages, paper_id_of
from copilot.tools.embeddings import HashingEmbedder
from copilot.tools.pdf import clean_text, parse_pdf_bytes
from copilot.tools.vectorstore import VectorStore

LONG = "Sign language production maps text to pose sequences. " * 40  # ~2000 chars, one paragraph


def test_chunking_packs_windows_and_filters():
    pages = [(1, "Tiny.\n\n" + "A reasonably long sentence about transformers and pose. " * 5), (2, LONG)]
    chunks = chunk_pages("arxiv:1", pages, size=500, overlap=50, max_chunks=10)
    assert chunks[0].chunk_id == "arxiv:1#1" and chunks[0].page == 1
    assert all(len(c.text) >= 80 for c in chunks)  # "Tiny." never becomes a chunk
    assert {c.page for c in chunks} == {1, 2} and len([c for c in chunks if c.page == 2]) >= 3
    assert all(len(c.text) <= 500 for c in chunks)
    assert len(chunk_pages("arxiv:1", pages, size=500, max_chunks=2)) == 2
    assert paper_id_of("arxiv:1#7") == "arxiv:1" and abstract_chunk_id("arxiv:1") == "arxiv:1#abs"
    assert paper_id_of("doi:10.1/x#abs") == "doi:10.1/x"


def test_clean_text_dehyphenates_and_joins_lines():
    assert clean_text("trans-\nformer models are\nbig.\n\n\n\nNew para") == "transformer models are big.\n\nNew para"


def test_vectorstore_search_filter_and_persistence(tmp_path):
    store = VectorStore(HashingEmbedder())
    store.add(
        [
            Chunk(chunk_id="p1#1", paper_id="p1", text="diffusion model generates sign pose sequences from gloss"),
            Chunk(chunk_id="p1#2", paper_id="p1", text="evaluation uses back translation BLEU on RWTH-PHOENIX-Weather"),
            Chunk(chunk_id="p2#1", paper_id="p2", text="convolutional network for image classification on ImageNet"),
        ]
    )
    assert store.add([Chunk(chunk_id="p1#1", paper_id="p1", text="dup")]) == 0  # ids are unique
    top = store.search("diffusion sign pose generation", k=1)
    assert top[0][0].chunk_id == "p1#1"
    assert [c.paper_id for c, _ in store.search("anything", k=5, paper_ids={"p2"})] == ["p2"]
    assert store.has_paper("p2") and not store.has_paper("p9")

    store.save(tmp_path)
    again = VectorStore.load(tmp_path, HashingEmbedder())
    assert len(again) == 3 and again.search("BLEU back translation", k=1)[0][0].chunk_id == "p1#2"
    assert np.allclose(again._mat, store._mat)


def test_parse_pdf_bytes_stops_at_references():
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    for text in [
        "Title page of the paper",
        "Method section about pose transformers.",
        "Results: we obtain BLEU 12.3.",
        "References\n[1] Someone et al.",
        "Appendix after refs",
    ]:
        doc.new_page().insert_text((72, 72), text)
    pages = parse_pdf_bytes(doc.tobytes(), max_pages=10)
    assert [p for p, _ in pages] == [1, 2, 3]  # page 4 is references -> dropped, page 5 never read
    assert "BLEU 12.3" in pages[2][1]
    assert len(parse_pdf_bytes(doc.tobytes(), max_pages=2)) == 2
