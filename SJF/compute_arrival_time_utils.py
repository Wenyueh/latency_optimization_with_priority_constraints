import argparse, random
import numpy as np

def fully_delayed_batching_helper(args, user_requests, mode):
    if mode == 'priority':
        batching_size = args.priority_predictor_batching_size
        latency = args.priority_predictor_latency
    elif mode == 'length':
        batching_size = args.length_predictor_batching_size
        latency = args.length_predictor_latency
    """
    Fully batching with NO partial batches:
      - We wait until exactly b requests are in the queue,
      - Then we process them as one batch, taking time c.
      - Any leftover < b remain unprocessed.

    Parameters
    ----------
    groups : List[List[int]]
        groups[i] is a list of request IDs arriving at time i*a.
        Example: [
          [1,2,3],       # time 0.0
          [4,5,6,7,8,9], # time 0.5
          [],            # time 1.0
          [10,11,12],    # time 1.5
          ...
        ]
    a : float
        The time gap between each consecutive group of arrivals
    b : int
        The exact batch size required before processing
    c : float
        The processing time for a batch of up to b requests

    Returns
    -------
    finishing_times : dict
        finishing_times[req_id] = the time when that request finishes processing.
        If a request never makes it into a full batch, it won't appear in this dict.
    """

    from collections import deque

    finishing_times = {}
    waiting_queue = deque()  # will store tuples (request_id, arrival_time)
    processor_free_time = 0.0

    # Go through each "group" in chronological order
    for i, group in enumerate(user_requests):
        arrival_time = i * args.user_request_gap
        # Enqueue each incoming request with its arrival time
        for req_id in group:
            waiting_queue.append((req_id, arrival_time))

        # While we can form a full batch, process it immediately
        while len(waiting_queue) >= batching_size:
            # Take exactly b requests from the front
            batch_requests = [waiting_queue.popleft() for _ in range(batching_size)]

            # We can only start after the processor is free
            # and after all b requests in this batch have arrived
            latest_arrival = max(r[1] for r in batch_requests)
            batch_start = max(processor_free_time, latest_arrival)
            batch_finish = batch_start + latency

            # Record finishing time for each request
            for (req_id, _) in batch_requests:
                finishing_times[req_id] = batch_finish

            # Update processor availability
            processor_free_time = batch_finish

    # 2) After reading all arrival groups, see if leftover requests remain
    if waiting_queue:
        # We'll process *one final partial batch* with whatever is left
        batch_requests = list(waiting_queue)
        waiting_queue.clear()

        # The earliest we can start is after the processor is free,
        # and after all leftover requests have arrived
        latest_arrival = max(r[1] for r in batch_requests)
        batch_start = max(processor_free_time, latest_arrival)
        batch_finish = batch_start + latency

        # Record finishing time for these leftover requests
        for (req_id, _) in batch_requests:
            finishing_times[req_id] = batch_finish

    return finishing_times

def fully_delayed_batching(args, user_requests, requests_coming_order):
    length_waiting_time = {}
    priority_waiting_time = {}

    priority_arrival_time = fully_delayed_batching_helper(args, user_requests, mode='priority')
    length_arrival_time = fully_delayed_batching_helper(args, user_requests, mode='length')

    for request_id in list(range(args.user_request_num)):
        length_waiting_time[request_id] = length_arrival_time[request_id] - requests_coming_order[request_id]*args.user_request_gap
        priority_waiting_time[request_id] = priority_arrival_time[request_id] - requests_coming_order[request_id]*args.user_request_gap

    return length_arrival_time, priority_arrival_time, length_waiting_time, priority_waiting_time

# process the requests whenever the predictor is empty and there are unprocessed requests, don't need to wait for the batch to be full
def immediate_batching_helper(args, user_requests, mode):
    if mode == 'priority':
        batching_size = args.priority_predictor_batching_size
        latency = args.priority_predictor_latency
    elif mode == 'length':
        batching_size = args.length_predictor_batching_size
        latency = args.length_predictor_latency
    """
    Immediate processing of requests, with batch size up to b and 
    processing time c for each batch. We never wait to fill the batch.

    Parameters
    ----------
    groups : List[List[int]]
        groups[i] is a list of request IDs arriving at time i*a.
        For example:
          groups = [
             [1, 2, 3],       # arrive at time 0.0
             [4, 5, 6, 7],    # arrive at time 0.5
             [],              # arrive at time 1.0
             [8, 9],          # arrive at time 1.5
             ...
          ]
    a : float
        The time gap between consecutive sub-lists (e.g. 0.5 seconds).
    b : int
        The maximum number of requests that can be processed in one batch.
    c : float
        The processing time for each batch (regardless of the number of requests in it).

    Returns
    -------
    finishing_times : dict
        finishing_times[req_id] = time when that request finishes processing.
    """

    from collections import deque

    # 1) Build a list of (request_id, arrival_time)
    arrivals = []
    for i, group in enumerate(user_requests):
        arrival_time = i * args.user_request_gap
        for req_id in group:
            arrivals.append((req_id, arrival_time))

    # 2) Sort by arrival_time (secondary sort by req_id if you want stable ordering)
    #    This step is often optional if 'groups' is already in ascending chronological order,
    #    but it's safer if you want to guarantee correctness for any input order.
    arrivals.sort(key=lambda x: (x[1], x[0]))

    finishing_times = {}
    waiting_queue = deque()  # will store (req_id, arrival_time)

    # Simulation "clock" and processor state
    current_time = 0.0         # We'll jump forward to significant events
    processor_free_time = 0.0  # The earliest time the processor can start a new batch

    i = 0                      # Index to iterate through 'arrivals'
    n = len(arrivals)

    # 3) Main loop: continue until all arrivals are processed and queue is empty
    while True:
        # If the waiting queue is empty, we have no requests to process right now.
        # We must jump to the time of the next arrival (if any).
        if not waiting_queue:
            if i < n:
                # Jump current_time to the next arrival
                next_arrival_time = arrivals[i][1]
                current_time = max(current_time, next_arrival_time)

                # Bring in all arrivals that happen *exactly* at current_time
                while i < n and arrivals[i][1] == current_time:
                    waiting_queue.append(arrivals[i])  # (req_id, arrival_time)
                    i += 1
            else:
                # No more arrivals at all, and queue is empty => we are done
                break

        # If we have requests waiting, we can process a batch
        if waiting_queue:
            # The batch can start no earlier than:
            #   (1) the current_time
            #   (2) the time the processor is free
            batch_start = max(current_time, processor_free_time)

            # We take up to b requests from the front of the queue
            batch_requests = []
            for _ in range(batching_size):
                if not waiting_queue:
                    break
                batch_requests.append(waiting_queue.popleft())

            # This batch will finish at:
            batch_finish = batch_start + latency

            # Record finishing times for these requests
            for (req_id, _) in batch_requests:
                finishing_times[req_id] = batch_finish

            # Update the processor's next available time
            processor_free_time = batch_finish

            # For simulation timing, we usually set current_time to the moment we
            # started this batch. The next "event" can be either:
            #  - new arrivals that occur before the batch finishes, or
            #  - the batch finishing time if no arrivals happen sooner.
            current_time = batch_start

        # Now we check if new arrivals come in while the processor is busy.
        # That means arrivals with arrival_time <= processor_free_time.
        while i < n and arrivals[i][1] <= processor_free_time:
            waiting_queue.append(arrivals[i])
            i += 1

        # Advance the simulation time to whichever comes next:
        #   (a) The next arrival time if it's before the processor finishes, or
        #   (b) The processor finishing time if that is earlier.
        if current_time < processor_free_time:
            if i < n:
                next_arrival_time = arrivals[i][1]
                if next_arrival_time < processor_free_time:
                    # We'll jump to the next arrival time
                    current_time = next_arrival_time
                    # Bring in all arrivals that occur exactly at that time
                    while i < n and arrivals[i][1] == current_time:
                        waiting_queue.append(arrivals[i])
                        i += 1
                else:
                    # No arrival occurs sooner than the batch finishing
                    current_time = processor_free_time
            else:
                # No more arrivals => jump to batch finish
                current_time = processor_free_time

        # Loop continues until we exhaust both arrivals and the queue

    return finishing_times

def immediate_batching(args, all_requests, requests_coming_order):
    length_waiting_time = {}
    priority_waiting_time = {}

    priority_arrival_time = immediate_batching_helper(args, all_requests, mode='priority')
    length_arrival_time = immediate_batching_helper(args, all_requests, mode='length')

    for request_id in list(range(args.user_request_num)):
        length_waiting_time[request_id] = length_arrival_time[request_id] - requests_coming_order[request_id]*args.user_request_gap
        priority_waiting_time[request_id] = priority_arrival_time[request_id] - requests_coming_order[request_id]*args.user_request_gap

    return length_arrival_time, priority_arrival_time, length_waiting_time, priority_waiting_time


# wait for the batch to be full to start
def compute_arrival_time(args, user_requests):
    waiting_time = {}
    arrival_time = {}
    user_requests_order = {tuple(one_batch): user_requests.index(one_batch) for one_batch in user_requests}
    requests_coming_order = {}
    for k,v in user_requests_order.items():
        for request_id in k:
            requests_coming_order[request_id] = v
    all_requests = [i for one_batch_user_request_ids in user_requests for i in one_batch_user_request_ids]

    if args.processing_mode == 'immediate_processing':
        length_arrival_time, priority_arrival_time, length_waiting_time, priority_waiting_time = immediate_batching(args, user_requests, requests_coming_order)
    elif args.processing_mode == 'fully_delayed_processing':
        length_arrival_time, priority_arrival_time, length_waiting_time, priority_waiting_time = fully_delayed_batching(args, user_requests, requests_coming_order)

    for request_id in all_requests:
        arrival_time[request_id] = max(length_arrival_time[request_id], priority_arrival_time[request_id])
        waiting_time[request_id] = max(length_waiting_time[request_id], priority_waiting_time[request_id])

    time_arrived_requests = {}
    for user_request_id, arrive_time in arrival_time.items():
        if arrive_time in time_arrived_requests:
            time_arrived_requests[arrive_time].append(user_request_id)
        else:
            time_arrived_requests[arrive_time] = [user_request_id]

    return waiting_time, arrival_time, time_arrived_requests

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gap_num', type=int, default=20)

    parser.add_argument('--user_request_gap', type=float, default=1)
    parser.add_argument('--max_concurrent_user_requests', type=int, default=10)
    parser.add_argument('--priority_predictor_batching_size', type=int, default=15)
    parser.add_argument('--priority_predictor_latency', type=float, default=0.1)
    parser.add_argument('--length_predictor_batching_size', type=int, default=15)
    parser.add_argument('--length_predictor_latency', type=float, default=0.1)
    parser.add_argument('--processing_mode', type=str, default='immediate_processing')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # for each time interval, we have a random number of user requests coming
    num_of_user_requests_coming = [random.randint(0, args.max_concurrent_user_requests) for _ in range(args.gap_num)]

    # total number of user requests coming
    args.user_request_num = sum(num_of_user_requests_coming)
    # initialize the user requests
    user_requests = []
    total_number = 0
    for number_of_user_requests in num_of_user_requests_coming:
        user_requests.append(list(range(total_number, total_number+number_of_user_requests)))
        total_number += number_of_user_requests

    print(user_requests)

    user_request_waiting_time_from_predictor, user_request_arrival_time_from_predictor, time_arrived_requests = compute_arrival_time(args, user_requests) 
    print(time_arrived_requests)
