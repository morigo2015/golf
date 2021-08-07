import logging
import log_tst2
import tst_util

log_zone = tst_util.get_logger(__name__, 'debug_tst.log')

def sub():
    print("sub")
    log_zone.debug("sub")
    log_tst2.func1()
    log_zone.debug("after call of log_tst2.func1")


def main():
    # logging.basicConfig(filename='debug_tst.log', level=logging.DEBUG)

    print("main")
    sub()
    log_zone.debug("main")


if __name__ == "__main__":
    main()
