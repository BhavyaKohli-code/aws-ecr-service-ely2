"""Semantic catalog: business words mapped to tables, retrieved per question.

Add a table by dropping a JSON file in ``data/semantic/tables/``.
Add a phrase in ``data/semantic/phrases.json``.
Add a document card in ``data/semantic/documents/``.
The model sees only the cards that match the question, so the catalog can grow
to thousands of tables and documents without putting them all in the prompt.
"""

from app.semantic.engine import answer_question
from app.semantic.queries import FigureResult

__all__ = ["FigureResult", "answer_question"]
