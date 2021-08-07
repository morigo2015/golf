import logging
import tst_util

log_tst2=tst_util.get_logger(__name__,'debug_tst2.log')

def func1():
    print("func1 in module")
    log_tst2.debug("in func1 of log_tst2")