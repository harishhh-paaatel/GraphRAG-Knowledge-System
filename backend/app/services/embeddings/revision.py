"""The vector revision counter: "has SQLite's set of chunks changed since the index was synced?"."""

from sqlalchemy.orm import Session

from app.models import VectorState
from app.models.base import utcnow


def current_vector_revision(session: Session) -> int:
    state = session.get(VectorState, 1)
    return state.revision if state is not None else 0


def bump_vector_revision(session: Session) -> int:
    """Increment inside the caller's transaction, so it commits or rolls back with the chunks."""
    state = session.get(VectorState, 1)
    if state is None:
        state = VectorState(id=1, revision=0)
        session.add(state)
    state.revision += 1
    state.updated_at = utcnow()
    session.flush()
    return state.revision
