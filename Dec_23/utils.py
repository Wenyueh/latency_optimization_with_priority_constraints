import asyncio, time
from async_timeout import timeout

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def round_2(some_input):
    if isinstance(some_input, float):
        return round(some_input,3)
    if isinstance(some_input, list):
        return [round_2(x) for x in some_input]
    if isinstance(some_input, int):
        return some_input
    

async def cancel(task):
    start = time.time()
    task.cancel()
    try:
        async with timeout(-1):
            await task
        end = time.time()
    except asyncio.CancelledError:
        end = time.time()
        return 'task cancelled'
    except asyncio.exceptions.TimeoutError:
        end = time.time()
        return 'time out for canceling task'
    except Exception as e:
        end = time.time()
        return 'exception:' + str(e)