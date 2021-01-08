

class BallFinder:
    def

class BallWatch:
    state = 'unknown'
    ball_spot = None
    ball_spot_frame_cnt = None
    ball_found_lst=[]

    @classmethod
    def next_frame(cls, frame, frame_cnt):
        # return state: unknown, ready, removed

        mask = BgSubtractor.apply(frame, frame_cnt)
        # if frame_cnt < 20: # need several first_search frames to build mask
        #     return 'unknown', -1, -1

        if cls.state == 'unknown':
            ball = cls.find_ball(mask)
            if ball is not None:
                cls.ball_found_lst.append(ball)
            if cls.still_ball_found()


            if cls.ball_spot is None:
                return 'unknown', -1, -1
            cls.ball_spot_frame_cnt = frame_cnt

            pass
        elif cls.state == 'ready':
            pass
        elif cls.state == 'rolling':
            pass
        else:
            print(f"unknown state = {cls.state}")
            exit(-1)

    @classmethod
    def get_mask(cls, frame):
        pass

    @classmethod
    def init_bg(cls, frame):
        pass
