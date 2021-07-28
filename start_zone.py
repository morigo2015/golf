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

    def find(self, frame) -> bool:
        # try to set up Start Zone (ball, border).
        # return True if ok (found and set up), else - False
        return False

    def current_state(self, frame) -> str:
        # analyze current state of StartArea: 'E' - empty, 'B' - ball, 'M' - mess
        return 'E'

    def draw(self, frame):
        # cv.rectangle(frame, (StartArea.x, StartArea.y), (StartArea.x + StartArea.w, StartArea.y + StartArea.h), (255, 0, 0), 1)
        # # cv.drawContours(frame, [StartArea.contour], 0, (0, 0, 255), 3)
        # cv.putText(frame, f"{status}", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
        return frame

