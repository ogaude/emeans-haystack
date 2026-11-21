# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the examples: a small multi-topic corpus and the Quick Start indexing pipeline."""

from pathlib import Path
from typing import Any

from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersDocumentEmbedder
from haystack_integrations.components.preprocessors.emeans import EmeansDocumentClusterer, EmeansFitter

# Each topic is a (subjects, statements) pair; every subject combined with every statement gives one sentence.
_FRAGMENTS: dict[str, tuple[list[str], list[str]]] = {
    "solar energy": (
        [
            "Solar panels",
            "Photovoltaic cells",
            "A rooftop solar array",
            "Solar farms",
            "Silicon solar modules",
            "Solar inverters",
        ],
        [
            "convert sunlight into electricity.",
            "lose some efficiency when they get too hot.",
            "can feed surplus power back into the grid.",
            "are cheaper to install now than a decade ago.",
            "work best when pointed toward the sun.",
        ],
    ),
    "cooking": (
        [
            "A cast iron skillet",
            "Fresh pasta",
            "A good chef's knife",
            "Slow-braised beef",
            "Homemade bread dough",
            "A simple tomato sauce",
        ],
        [
            "tastes better when the seasoning is added early.",
            "needs patience and a gentle flame.",
            "is the base of many family recipes.",
            "should rest before it is served.",
            "is easy to cook once the ingredients are prepped.",
        ],
    ),
    "football": (
        [
            "The striker",
            "A well-drilled defence",
            "The goalkeeper",
            "The midfield trio",
            "The home crowd",
            "The head coach",
        ],
        [
            "decided the match with a late goal.",
            "controlled possession for most of the second half.",
            "was praised after the league fixture.",
            "pressed high up the pitch from the first whistle.",
            "kept the team in the title race this season.",
        ],
    ),
    "astronomy": (
        [
            "A neutron star",
            "The Andromeda galaxy",
            "A distant exoplanet",
            "Jupiter's largest moons",
            "A supernova remnant",
            "The space telescope",
        ],
        [
            "is visible only with a powerful instrument.",
            "reveals how stars form and die.",
            "orbits far beyond our solar system's planets.",
            "was photographed in remarkable detail last year.",
            "helps astronomers measure the age of the universe.",
        ],
    ),
    "programming": (
        [
            "A unit test",
            "The compiler",
            "A recursive function",
            "The version control history",
            "A well-named variable",
            "The code review",
        ],
        [
            "catches bugs before they reach production.",
            "makes the software easier to maintain.",
            "is where most developers spend their debugging time.",
            "shows how the codebase evolved over time.",
            "keeps the program readable for the next developer.",
        ],
    ),
    "gardening": (
        [
            "Tomato seedlings",
            "A compost heap",
            "The raised vegetable bed",
            "Fresh mulch",
            "A pruned rose bush",
            "The herb garden",
        ],
        [
            "needs regular watering during a dry summer.",
            "improves the soil for the next planting season.",
            "grows best in a sunny, sheltered spot.",
            "keeps weeds down and moisture in the ground.",
            "attracts bees and other useful pollinators.",
        ],
    ),
    "music": (
        [
            "The jazz quartet",
            "A grand piano",
            "The symphony orchestra",
            "An acoustic guitar",
            "The lead singer",
            "A vinyl record",
        ],
        [
            "filled the concert hall with a warm sound.",
            "was recorded live over a single weekend.",
            "plays a melody that stays in your head for days.",
            "gave the album its distinctive rhythm.",
            "sounds richer when the room acoustics are good.",
        ],
    ),
    "medicine": (
        [
            "A vaccine",
            "The surgeon",
            "Regular blood pressure checks",
            "An antibiotic course",
            "The clinical trial",
            "A physiotherapy programme",
        ],
        [
            "lowers the risk of serious illness in patients.",
            "is prescribed only after a careful diagnosis.",
            "helps the body recover after an operation.",
            "is reviewed by doctors before approval.",
            "became standard treatment in hospitals worldwide.",
        ],
    ),
    "finance": (
        [
            "The central bank",
            "A diversified portfolio",
            "Rising interest rates",
            "The quarterly earnings report",
            "A savings account",
            "The bond market",
        ],
        [
            "affects how much households pay on their mortgages.",
            "reflects investors' expectations about inflation.",
            "helps spread risk across different asset classes.",
            "moved share prices on the stock exchange.",
            "rewards savers who keep money invested for decades.",
        ],
    ),
    "travel": (
        [
            "The night train",
            "A small coastal village",
            "The mountain hiking trail",
            "An old city centre",
            "The budget airline",
            "The local market",
        ],
        [
            "is a memorable stop on a long holiday.",
            "offers travellers stunning views and quiet mornings.",
            "is best explored on foot with a good map.",
            "gets crowded during the summer tourist season.",
            "makes a great base for a week of sightseeing.",
        ],
    ),
}

TOPICS: dict[str, list[str]] = {
    topic: [f"{subject} {statement}" for subject in subjects for statement in statements]
    for topic, (subjects, statements) in _FRAGMENTS.items()
}


def corpus_documents() -> list[Document]:
    """Return one `Document` per corpus text, with the topic stored in `meta["topic"]`."""
    return [Document(content=text, meta={"topic": topic}) for topic, texts in TOPICS.items() for text in texts]


def output_dir() -> Path:
    """Return the (gitignored) folder where examples write their model files."""
    path = Path(__file__).parent / ".output"
    path.mkdir(exist_ok=True)
    return path


def index_corpus(store: InMemoryDocumentStore, model_path: str) -> dict[str, Any]:
    """Run the Quick Start indexing pipeline over the corpus and return its result (including the fitter outputs)."""
    indexing = Pipeline()
    indexing.add_component("embedder", SentenceTransformersDocumentEmbedder())
    indexing.add_component("fitter", EmeansFitter(target_cluster_size=50, model_path=model_path, overwrite=True))
    indexing.add_component("clusterer", EmeansDocumentClusterer())
    indexing.add_component("writer", DocumentWriter(document_store=store))
    indexing.connect("embedder.documents", "fitter.documents")
    indexing.connect("embedder.documents", "clusterer.documents")
    indexing.connect("fitter.model_path", "clusterer.model_path")
    indexing.connect("clusterer.documents", "writer.documents")
    return indexing.run({"embedder": {"documents": corpus_documents()}}, include_outputs_from={"fitter"})
