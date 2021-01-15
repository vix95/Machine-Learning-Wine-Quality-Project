from Models.Model import Model
import time

EXECUTING_TIME = []


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2 - time1) * 1000.0))
        EXECUTING_TIME.append([f.__name__, (time2 - time1) * 1000.0])

        return ret

    return wrap


def save_executing_time(obj):
    obj.executing_time = EXECUTING_TIME[0][1]
    EXECUTING_TIME.clear()
