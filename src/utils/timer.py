import datetime
from typing import Dict

class Timer(object):
    def __init__(self, start_iter: int, end_iter: int):
        self.total_iter = end_iter - start_iter
        self.start_iter = start_iter
        self._started = False

    def start(self):
        self._started = True
        self.start_time = datetime.datetime.now()
        self.current_time = datetime.datetime.now()

    def get_time_stat(self, current_iter: int) -> Dict:
        assert self._started, 'Timer has not been started'
        iter_from_start = current_iter - self.start_iter
        _now = datetime.datetime.now()
        runtime = _now - self.start_time
        interval = _now - self.current_time
        time_per_iter = runtime / iter_from_start
        remaining = time_per_iter * (self.total_iter - iter_from_start)
        end_time = _now + remaining
        self.current_time = _now
        return {
            'start_time': Timer.convert_datetime_mdhm(self.start_time),
            'runtime': Timer.convert_timedelta_dhm(runtime),
            'interval': Timer.convert_timedelta_dhm(interval),
            'time_per_iter': Timer.convert_timedelta_dhm(time_per_iter),
            'remaining': Timer.convert_timedelta_dhm(remaining),
            'end_time': Timer.convert_datetime_mdhm(end_time),
        }

    @staticmethod
    def convert_datetime_mdhm(d_time: datetime.datetime) -> str:
        """datetime -> M/DD HH:MM ex: "4/30 15:18"
        """
        return '%d/%02d %02d:%02d' % (d_time.month, d_time.day, d_time.hour, d_time.minute)

    @staticmethod
    def convert_timedelta_dhm(td: datetime.timedelta) -> str:
        d = td.days
        m, _ = divmod(td.seconds, 60)
        h, m = divmod(m, 60)
        return f'{d}d {h}h {m}m'