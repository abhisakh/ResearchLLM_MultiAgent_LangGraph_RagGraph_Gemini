"""VectorDBWrapper: add, dedupe, persist, reset and search (fake embeddings, temp folder)."""
import pytest

CHUNKS = [
    {"text": "CsSnI3 is a lead-free perovskite material studied as a less toxic alternative to lead-halide perovskites."},
    {"text": "The cubic phase of CsSnI3 has a theoretical bandgap near 1.3 eV, which suits solar cells."},
    {"text": "Rapid oxidation of Sn2+ to Sn4+ leads to poor long-term stability."},
    {"text": "Excess SnF2 during solution processing partly mitigates the instability."},
    {"text": "An irrelevant chunk about quantum computing algorithms."},
]


@pytest.fixture
def db(vector_db_module):
    return vector_db_module.VectorDBWrapper()


def test_new_database_is_empty(db):
    assert db.index.ntotal == 0
    assert db.text_store == []


def test_add_chunks_indexes_every_unique_text(db):
    db.add_chunks(CHUNKS)
    assert db.index.ntotal == len(CHUNKS)
    assert len(db.text_store) == len(CHUNKS)


def test_adding_the_same_text_twice_does_not_duplicate_it(db):
    db.add_chunks(CHUNKS)
    db.add_chunks(CHUNKS)
    assert db.index.ntotal == len(CHUNKS)


def test_blank_chunks_are_ignored(db):
    db.add_chunks([{"text": "   "}, {"text": ""}, {}])
    assert db.index.ntotal == 0


def test_index_survives_a_reload(db, vector_db_module):
    db.add_chunks(CHUNKS)
    reloaded = vector_db_module.VectorDBWrapper()
    assert reloaded.index.ntotal == len(CHUNKS)
    assert [c["text"] for c in reloaded.text_store] == [c["text"] for c in CHUNKS]


@pytest.mark.parametrize(
    "query, expected_word",
    [
        ("cubic phase bandgap", "cubic"),
        ("oxidation of Sn2+ to Sn4+", "oxidation"),
        ("quantum computing algorithms", "quantum"),
    ],
)
def test_search_ranks_the_matching_chunk_first(db, query, expected_word):
    db.add_chunks(CHUNKS)
    results = db.search(query, k=3)
    top_chunk, top_score = results[0]
    assert expected_word in top_chunk["text"].lower()
    assert top_score > results[-1][1]


def test_search_returns_at_most_the_number_of_stored_chunks(db):
    db.add_chunks(CHUNKS[:2])
    assert len(db.search("perovskite", k=30)) == 2


def test_search_on_an_empty_database_returns_nothing(db):
    assert db.search("anything") == []


def test_query_with_no_embeddable_words_returns_nothing(db):
    db.add_chunks(CHUNKS)
    assert db.search("??? !!!") == []


def test_reset_clears_memory_and_disk(db, vector_db_module):
    db.add_chunks(CHUNKS)
    db.reset_db()
    assert db.index.ntotal == 0
    assert db.text_store == []
    assert vector_db_module.VectorDBWrapper().index.ntotal == 0
