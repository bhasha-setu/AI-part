"""Item selection for a session.

Order: due reviews first (earliest-due first), then new items (content order),
capped at ``cap``. Items that have a record but are not yet due are skipped.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from .content import Item
from .learner_model import LearnerModel, _parse


def _default_cap() -> int:
    from .course import default_course

    return default_course().level.max_items_per_session


def select_items(
    items: list[Item],
    learner: LearnerModel,
    now: datetime,
    cap: Optional[int] = None,
) -> list[Item]:
    """Due reviews first (earliest due first), then new items, capped.

    ``cap`` defaults to the level's ``max_items_per_session``.
    """
    if cap is None:
        cap = _default_cap()
    due: list[tuple] = []
    new: list[Item] = []
    for it in items:
        st = learner.state_for(it.id)
        if st is None:
            new.append(it)
        elif _parse(st.due) <= now:
            due.append((_parse(st.due), it))
        # else: seen recently, not yet due -> skip
    due.sort(key=lambda pair: pair[0])
    ordered = [it for _, it in due] + new
    return ordered[: max(0, cap)]
