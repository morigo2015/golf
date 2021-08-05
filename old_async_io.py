# my Input-Output Utils

import cv2
import os
import datetime
import time
import glob
from threading import Thread
from queue import Queue, Empty, Full

class InputStream:
    def __init__(self, input_src=None, input_async=None, input_queue_size=None):
        if input_src is None:
            input_src = cfg['input_source']
        self.input_source = input_src

        if self.input_source == 0 or self.input_source == '0' :
            self.source_type = 'camera'
        elif self.input_source[0:7] == 'rtsp://':
            self.source_type = 'rtsp'
        else:
            self.source_type = 'file'

        self.handle = cv2.VideoCapture(input_src) # !!

        if input_async is None:
            input_async = cfg['input_async']  # True - non blocking read from queue , else - bloc
        self.input_async = input_async

        if self.input_async == True:
            if input_queue_size is None:
                input_queue_size = cfg['input_async_queue_size']
            self.async_mgr = _AsyncVideoStream(self.handle,queueSize=input_queue_size)
            self.async_mgr.start()
            time.sleep(1.0)  # to fill queue by frames
        else: # input_async == False
            self.async_mgr = None

        self.frame_cnt = 0
        self.frame_shape = (int(self.handle.get(3)), int(self.handle.get(4)))  # (w,h)
        self.last_frame = None
        self.start_time = datetime.datetime.now()
        Log.log(f'input stream {self.input_source} opened, async mode = {self.input_async}', level='info')


    def read_frame(self):
        if self.input_async:
            ret, self.last_frame = self.async_mgr.read()
            isOpened = ret # Later: add error analyzes and processing
        else: # sync read
            ret, self.last_frame = self.handle.read()
            isOpened = self.handle.isOpened()

        if isOpened == False or ret == False:
            if self.source_type == 'camera' or self.source_type == 'rtsp':
                Log.log(f'input source from {self.source_type} is closed. isOpened={self.handle.isOpened()}, ret={ret}')
                self.last_frame = self._repaired_read()
            else:
                return None
        self.frame_cnt += 1
        if cfg['input_show_counter_div'] >0 and self.frame_cnt % cfg['input_show_counter_div']==0:

            if self.input_async:
                qsz = self.async_mgr.qsize()
                try:
                    cam_fps = self.async_mgr.read_frames_cnt / (datetime.datetime.now()-self.start_time).seconds
                except ZeroDivisionError:
                    cam_fps = 0.
                qinfo_msg = f'Queue:{qsz[0]:3d}/{qsz[1]:3d} Drop:{self.async_mgr.dropped_cnt} fps: {cam_fps:4.1f}'
            else:
                qinfo_msg = ''

            ms_img = (datetime.datetime.now()-self.start_time).seconds * 1000. / self.frame_cnt
            msg = f'Frames processed {self.frame_cnt:5} ({ms_img:4.0f}ms)    {qinfo_msg}    cpu: {Misc.get_cpu_avg()}'
            print(f'\033[s\033[1;1f  ==============   {msg}   ==============\n\033[u',end='',flush=True)
        return self.last_frame

    def _repaired_read(self):
        # there was some problem with input source, try to repair, return newly read frame if ok, or None if fault
        if self.source_type == 'file':
            return None
        # source type is 'camera':
        Log.log(' Something wrong with camera stream. Trying to repair ...')
        for recapture_attempt in range(1,cfg['input_recapture_max']):

            del self.handle
            self.handle = cv2.VideoCapture(self.input_source)

            if cv2.waitKey(1) & 0xFF in cfg['exit_chars']:
                Log.log('repairing of input stream has been cancelled by user')
                return None
            if not self.handle.isOpened():
                print(f'    {datetime.datetime.now()}: still repairing input stream, isOpened = False')
                time.sleep(cfg['input_recapture_delay'])
                continue

            # here: handle.isOpened() == True

            if self.input_async:
                self.async_mgr.re_init(self.handle)
                ret, self.last_frame = self.async_mgr.read()
            else:
                ret, self.last_frame = self.handle.read()

            if ret == True and self.last_frame is not None:
                Log.log('Input source is restored after {} attempts'.format(recapture_attempt))
                return self.last_frame

        Log.log("Input source has NOT been resotored after {} attempts".format(recapture_attempt))
        return None

    def async_queue_size(self):
        if self.input_async:
            return self.async_mgr.qsize()
        else:
            return -1,-1

    def info(self):
        str = ''
        str += 'Input  stream info: source={} shape={}'.format(self.input_source, self.frame_shape)
        str += 'frames={}'.format(self.frame_cnt)
        time_seconds = (datetime.datetime.now() - self.start_time).seconds
        if time_seconds != 0.:
            fps = self.frame_cnt / time_seconds
            str += f' fps={fps:.2f} ({(1./fps)*1000:.0f}ms/img)'
        return str

    def __del__(self):
        if self.input_async:
            self.async_mgr.stop()
        else:
            self.handle.release()

class _AsyncVideoStream:
    def __init__(self, handle, queueSize=10):
        # initialize the file video stream along with the boolean used to indicate if the thread should be stopped or not
        self.stream = handle # cv2.VideoCapture(path)
        self.max_queue_size = queueSize
        self.stopped = False

        # initialize the queue used to store frames read from the video file
        self.Q = Queue(maxsize=queueSize)
        self.dropped_cnt = 0
        self.read_frames_cnt = 0

    def re_init(self,handle):
        # reinit after input stream has been broken and restored
        self.stream = handle
        self.stopped = False
        self.start()

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # main procedure for reading-thread
        while True:
            # to stop the reading-thread,  self.stooped will be set in main thread
            if self.stopped:
                print(f'update: self.stopped={self.stopped}')
                return

            (grabbed, frame) = self.stream.read()
            self.read_frames_cnt += 1

            if not grabbed:
                print('not grabbed')
                self.stop()
                return

            if cfg['input_async_show_qsize']:
                # display the size of the queue on the frame
                qsz = self.qsize() # (cureent size, max size)
                cv2.putText(frame, f"Queue: {qsz[0]} of {qsz[1]}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            try:
                self.Q.put(frame,block=False)
            except Full:
                #print('q full - last frame is dropped!!! ')
                self.dropped_cnt += 1
                time.sleep(
                    cfg['input_async_full_delay'] / 1000.)  # we are in reading thread now and main thread isn't ready yet

    def read(self):
        if self.stopped:
            return False, None
        while True:
            try:
                frame = self.Q.get(block=False)
                return True, frame
            except Empty:
                # print('q empty - lets sleep awhile')
                time.sleep(cfg['input_async_empty_delay'] / 1000.)  # we are in main thread now and there is nothing to do yet
            if self.stopped: return False, None

    def qsize(self):
        return self.Q.qsize(), self.max_queue_size-1

    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        print('stop is called')


# logging in-out operations
class Log:
    file_handle = None

    @staticmethod
    def log(msg, level='warning'):
        if Log.file_handle is None:
            Log.file_handle = open(cfg['log_file_name'], 'a')
        msg_str = f'{level:8s}: {str(datetime.datetime.now())[:22]} {msg}\n'
        #print(f'log: {msg_str}',end='')  # todo verbose leveling
        Log.file_handle.write(msg_str)
        Log.file_handle.flush()

class Misc:
    _cpu_avg_handle = open('/proc/loadavg')

    @staticmethod
    def get_cpu_avg():
        Misc._cpu_avg_handle.seek(0)
        return ' '.join(Misc._cpu_avg_handle.read().split()[0:3])

# -------------------------------------------------------------------------------------------------

import sys

from cam_time_measure import TimeMeasure
_cfg = {}
_cfg['frame_cnt_divider_to_inform'] = 50
_cfg['GUI'] = False
_cfg['write to file'] = True
_cfg['delay_ms'] = 1 # 1900
_cfg['input_async'] = True
_cfg['input_async_queue_size'] = 50

def store_input_stream():
    """
    store input stream into file
    Finish at the end of input stream of Ctrl-C
    all non-defined parameters - from cfg file
    """
    print('Store input stream utility.)')
    print(f'Python interpreter:{sys.executable} OpenCV version:{cv2.__version__}')
    print(f'config: {_cfg}')
    inp_stream = InputStream(input_async=_cfg['input_async'],input_queue_size=_cfg['input_async_queue_size'])
    outp_stream = OutputStream(cfg['out_file_name'], inp_stream.frame_shape, cfg['out_file_fps'],mode='save_old')

    try:
        while True:
            TimeMeasure.set('read')
            inp_frame = inp_stream.read_frame()
            if inp_frame is None:
                print(f'input steam closed')
                break

            try: frame_cnt += 1
            except NameError: frame_cnt=0
            if frame_cnt % _cfg['frame_cnt_divider_to_inform'] ==0:
                print(f'frame counter = {frame_cnt}')

            if _cfg['GUI']:
                TimeMeasure.set('show')
                cv2.imshow('frame',inp_frame)
                TimeMeasure.set('wait')
                ch = cv2.waitKey(1)
                if ch & 0xFF == ord('q'):
                    print('exit char received')
                    break

            if _cfg['write to file']:
                TimeMeasure.set('write')
                outp_stream.write_frame(inp_frame)

            if _cfg['delay_ms']:
                TimeMeasure.set('sleep')
                time.sleep(_cfg['delay_ms']/1000.)

    except KeyboardInterrupt:
        print('cancelled by user')

    print(f'total frames saved: {frame_cnt}')
    print(TimeMeasure.results())

if __name__ == '__main__':
    store_input_stream()