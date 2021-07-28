from typing import List, Tuple, Any


class StartBall:
    x,y = None, None
    area: float = None
    pass


class StartZone:
    ball: StartBall = None
    corner_lst: List[Tuple[int, int]] = None
    # zone_contour: List[Any] = None
    thresh_val: float = None
    x, y, w, h = None, None, None, None  # int

