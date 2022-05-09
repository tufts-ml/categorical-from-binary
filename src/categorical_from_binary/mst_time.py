from datetime import datetime

import pytz
from pytz import timezone


def get_mst_time():
    date_format = "%m_%d_%Y_%H_%M_%S_%Z"
    date = datetime.now(tz=pytz.utc)
    date = date.astimezone(timezone("US/Mountain"))
    mstDateTime = date.strftime(date_format)
    return mstDateTime
