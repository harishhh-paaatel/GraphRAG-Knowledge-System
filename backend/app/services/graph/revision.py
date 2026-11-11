"""The graph revision counter that ties the persisted graph to SQLite."""

from sqlalchemy.orm import Session

from app.models import GraphState
from app.models.base import utcnow


def current_revision(session: Session) -> int:
    """Revision recorded in SQLite (0 when no change was ever recorded)."""
    state = session.get(GraphState, 1)
    return state.revision if state is not None else 0


def bump_revision(session: Session) -> int:
    """Increment the revision inside the caller's transaction and return the new value.

    Because it commits (or rolls back) with the data change itself, the revision can never
    claim a change that did not happen, nor miss one that did.
    """
    state = session.get(GraphState, 1)
    if state is None:
        state = GraphState(id=1, revision=0)
        session.add(state)
    state.revision += 1
    state.updated_at = utcnow()
    session.flush()
    return state.revision
