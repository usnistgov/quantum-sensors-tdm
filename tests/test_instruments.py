import instruments
from instruments import retry
import pytest

class AException(Exception):
    pass

class NFailThenWork():
    def __init__(self, n_fail_before_work):
        self.n_fail_before_work = n_fail_before_work
        self.fails = 0

    def attempt(self):
        if self.fails >= self.n_fail_before_work:
            return
        else:
            self.fails += 1
            raise AException("planned failure") 

def test_retry():
    failer = NFailThenWork(199)
    @retry(tries=3, delay_s=0, logger=False)
    def attempt_fail():
        failer.attempt()

    with pytest.raises(AException):
        attempt_fail()

    work_after_2 = NFailThenWork(2)
    @retry(tries=3, delay_s=0, logger=False)
    def attempt_work():
        work_after_2.attempt()

    attempt_work()
