import numpy as np


def resolve_distance_only_user_sort(args):
    """Return the effective roster rule used by truth and reconstruction."""
    return bool(
        getattr(args, "distance_only_user_sort", False)
        or (
            getattr(args, "spatial_flight_actor", False)
            and not getattr(args, "completion_priority_user_sort", False)
        )
    )


def order_md_candidates(
    distances, session_ids, can_finish, task_valid=None, *, distance_only=False
):
    """Order records by priority, then distance, then stable session ID."""
    distances = np.asarray(distances)
    session_ids = np.asarray(session_ids)
    can_finish = np.asarray(can_finish, dtype=bool)
    if task_valid is None:
        task_valid = np.ones_like(can_finish, dtype=bool)
    else:
        task_valid = np.asarray(task_valid, dtype=bool)

    priority = np.zeros_like(distances, dtype=np.int8)
    if not distance_only:
        priority = np.where(task_valid, np.where(can_finish, 1, 0), 2)
    return np.lexsort((session_ids, distances, priority))
