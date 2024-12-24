import asyncio, time
from async_timeout import timeout
import numpy as np
    

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
    

def compute_average_waiting_time(record_requests):
    waiting_time_with_priority = {}
    for i in range(len(record_requests)):
        print(f"User request {record_requests[i].user_request_id} has a total waiting time of {record_requests[i].waiting_time}")
        if record_requests[i].priority not in waiting_time_with_priority:
            waiting_time_with_priority[record_requests[i].priority] = [record_requests[i].waiting_time]
        else:
            waiting_time_with_priority[record_requests[i].priority].append(record_requests[i].waiting_time)
    for k,v in waiting_time_with_priority.items():
        print(f"Priority {k} has an average waiting time of {np.mean(v)}")
        waiting_time_with_priority[k] = round_2(np.mean(v))

    # sort waiting time with priority by key from small to large
    waiting_time_with_priority = dict(sorted(waiting_time_with_priority.items()))

    return waiting_time_with_priority
