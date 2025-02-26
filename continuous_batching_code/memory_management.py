import heapq
import math, sys
from heapq import heappush, heappop
from semantic_predictor import update_cache_loading_or_recomputation_time_and_extra_saving_time


class TwoDimensionalPriorityQueue:
    def __init__(self):
        self.heap = []

    def push(self, item, priority1, priority2):
        # semantic priority, remaining computation time
        # Use a tuple (priority1, priority2, item) to maintain the heap property
        heappush(self.heap, (priority1, priority2, item))

    def pop(self):
        # Pop the item with the highest priority (smallest priority1, then smallest priority2)
        return heappop(self.heap)[2]

    def peek(self):
        # Peek at the item with the highest priority without popping it
        return self.heap[0][2]

    def is_empty(self):
        return len(self.heap) == 0

class MaxHeap_Memory_Class:
    def __init__(self, args):
        self.total_blocks = args.KV_block_number
        self.block_size = args.block_size
        self.used_blocks = 0
        self.heap = []                  # 3-tuple: (-priority, -remaining_time, request)
        self.request_id2blocks = {}               # {request_id: block_number}
        self.request_id2tokens = {}               # {request_id: token_number}
        self.ongoing_request_ids = set()   # {request_id}

    # how many left
    def storage_left(self):
        return self.total_blocks - self.used_blocks

    def push(self, item, priority1, priority2):
        # semantic priority, remaining computation time
        # Use a tuple (priority1, priority2, item) to maintain the max heap property
        heappush(self.heap, (-priority1, -priority2, item))

    def pop(self):
        # Pop the item with the lowest priority
        return heappop(self.heap)[2]

    def peek(self):
        # Peek at the item with the lowest priority without popping it
        return self.heap[0][2]

    def reconstruct_heap(self, ongoing_request_list):
        ongoing_request_id2request = {request.user_request_id: request for request in ongoing_request_list}
        new_heap = []
        for item in self.heap:
            request_id = item[2].user_request_id
            if request_id not in ongoing_request_id2request.keys():
                new_heap.append(item)
            else:
                request = ongoing_request_id2request[request_id]
                new_heap.append((-request.predicted_priority,-(request.predicted_remaining_computation_time[-1] + request.prefill_cache_loading_time + request.decoding_cache_loading_time), request))

        heapq.heapify(new_heap)
        self.heap = new_heap

    def update_ongoing_requests(self, ongoing_request_list):
        self.ongoing_request_ids = {request.user_request_id for request in ongoing_request_list}

    def usable_tokens_for(self, request_id):
        if request_id in self.request_id2blocks and request_id in self.request_id2tokens:
            return self.request_id2blocks[request_id] * self.block_size - self.request_id2tokens[request_id]
        return 0

    def allocate_memory_for(self, user_request, requested_tokens):
        """

        :param user_request:
        :param requested_tokens:
        :return:
        """
        res = {'preempted_requests': [], 'deallocated_requests': []}    # {{preempted_request_id: list of request object}, {deallocated_request_id: list of request object}}
        request_id = user_request.user_request_id
        priority = user_request.predicted_priority
        remaining_time = user_request.predicted_remaining_computation_time[-1] + user_request.prefill_cache_loading_time + user_request.decoding_cache_loading_time
        if request_id not in self.request_id2blocks:
            self.request_id2blocks[request_id] = 0
            self.request_id2tokens[request_id] = 0
            self.push(user_request, priority, remaining_time)

        if self.usable_tokens_for(request_id) >= requested_tokens:
            self.request_id2tokens[request_id] += requested_tokens
            return res

        requested_blocks = math.ceil((requested_tokens - self.request_id2tokens[request_id]) / self.block_size)

        if self.storage_left() >= requested_blocks:
            self.request_id2blocks[request_id] += requested_blocks
            self.request_id2tokens[request_id] += requested_tokens
            self.used_blocks += requested_blocks
            return res
        else:
            requested_blocks_remaining = requested_blocks
            requested_blocks_remaining -= self.storage_left()
            # self.request_id2blocks[request_id] += self.storage_left()
            # self.request_id2tokens[request_id] += self.storage_left() * self.block_size
            # self.used_blocks += self.storage_left()

            # deallocate the memory from the lowest priority request
            while requested_blocks_remaining > 0:
                lowest_priority_request = self.pop()
                lowest_priority_request_id = lowest_priority_request.user_request_id
                if lowest_priority_request_id in self.ongoing_request_ids:
                    self.ongoing_request_ids.remove(lowest_priority_request_id)
                    res['preempted_requests'].append(lowest_priority_request)

                if lowest_priority_request_id == request_id:
                    assert lowest_priority_request in res['preempted_requests']
                    return res

                # lazy deletion of the completed request and the preempted request due to the failed memory allocation
                if lowest_priority_request_id not in self.request_id2blocks:
                    continue
                else:
                    if self.request_id2blocks[lowest_priority_request_id] == 0:
                        self.request_id2blocks.pop(lowest_priority_request_id)
                        self.request_id2tokens.pop(lowest_priority_request_id)
                        continue

                if self.request_id2blocks[lowest_priority_request_id] <= requested_blocks_remaining:
                    requested_blocks_remaining -= self.request_id2blocks[lowest_priority_request_id]
                    # self.request_id2blocks[request_id] += self.request_id2blocks[lowest_priority_request_id]
                    self.request_id2blocks.pop(lowest_priority_request_id)
                    self.request_id2tokens.pop(lowest_priority_request_id)
                    res['deallocated_requests'].append(lowest_priority_request)
                else:
                    requested_blocks_remaining = 0
                    # self.request_id2blocks[request_id] += requested_blocks_remaining
                    self.request_id2blocks[lowest_priority_request_id] -= requested_blocks_remaining
                    self.request_id2tokens[lowest_priority_request_id] = self.request_id2blocks[lowest_priority_request_id] * self.block_size

                    #TODO: update remaining time for the lowest_priority_request
                    update_cache_loading_or_recomputation_time_and_extra_saving_time(lowest_priority_request)
                    remaining_time = lowest_priority_request.predicted_remaining_computation_time[-1] + lowest_priority_request.prefill_cache_loading_time + lowest_priority_request.decoding_cache_loading_time
                    self.push(lowest_priority_request, priority, remaining_time)

                    res['deallocated_requests'].append(lowest_priority_request)

            # atomic memory allocation for the request
            self.request_id2blocks[request_id] += requested_blocks
            self.request_id2tokens[request_id] += requested_tokens
            self.used_blocks += self.storage_left()

        return res

    def deallocate_memory_for(self, user_request):
        request_id = user_request.user_request_id
        if request_id in self.ongoing_request_ids:
            self.ongoing_request_ids.remove(request_id)

        if request_id in self.request_id2blocks:
            self.request_id2blocks.pop(request_id)
            self.request_id2tokens.pop(request_id)
            self.used_blocks -= self.request_id2blocks[request_id]
            # heap deletion is deferred to the future allocation






# class KV_block_Class:
#     def __init__(self, args):
#         self.args = args
#         self.block_size = args.block_size
#         self.total_storage = args.GPU_KV_cache
#         # KV: {block_id: {request_id1: [token1, ...]}}
#         # each block take only one request's tokens
#         self.cache = {i: [] for i in range(args.KV_block_number)}

#     # how many tokens have occupied this block
#     def space_taken_by_block(self, block_id):
#         return sum([len(v) for v in self.cache[block_id]])
    
#     # how many tokens can stil be stored in this block
#     def space_left_in_block(self, block_id):
#         return self.block_size - sum([len(v) for v in self.cache[block_id]])

#     # the GPU is full or not
#     def is_full(self):
#         # check whether all blocks are full
#         return all([self.space_taken_by_block(k) >= self.block_size for k,v in self.cache.items()])

#     # how many left
#     def storage_left(self):
#         return sum([self.block_size-self.space_taken_by_block(k) for k,v in self.cache.items()])    

#     # KV: {request_id: [token1, token2, ...]}
#     def insert(self, KV):
#         # KV: {request_id: {token1: block_id, token2: block_id, ...}}
#         token_block_id = {}
#         number_of_tokens = sum([len(v) for v in KV.values()])
#         token_inserted = 0
#         while token_inserted < number_of_tokens:
#             for block_id, block in self.cache.items():    
#                 if self.space_left_in_block(block_id) >= number_of_tokens:
#                     self.cache[block_id].extend(KV)
#                     for request_id, tokens in KV.items():
#                         token_block_id[request_id] = {}
#                         for token in tokens:
#                             token_block_id[request_id][token] = block_id
#                     token_inserted += number_of_tokens
#                 else:
#                     space_left = self.space_left_in_block(block_id)
#                     kv_slice_to_insert = {}
#                     for request_id,tokens in KV.items():
#                         token_block_id[request_id] = {}
#                         if len(tokens) <= space_left:
#                             kv_slice_to_insert[request_id] = tokens
#                             space_left -= len(tokens)
#                             for token in tokens:
#                                 token_block_id[request_id][token] = block_id
#                         else:
#                             kv_slice_to_insert[request_id] = tokens[:space_left]
#                             KV[request_id] = tokens[space_left:]
#                             for token_id, token in enumerate(tokens):
#                                 if token_id < space_left:
#                                     token_block_id[request_id][token] = block_id
#                     self.cache[block_id].extend(kv_slice_to_insert)
#                     token_inserted += sum([len(v) for v in kv_slice_to_insert.values()])

#         return token_block_id

#     def remove(self, block_id):
#         self.cache[block_id] = []



def compute_GPU_KV_storage_size(args):
    total_GPU_memory = args.GPU_memory * args.GPU_memory_utilization
    model_size = int(args.setting.split('_')[1][-2])
    activation_memory = total_GPU_memory * args.activation_memory_percentage
    storage_left_for_KV = total_GPU_memory - activation_memory - model_size
    # compute KV size for each token 
    if args.setting.split('_')[1] == '4B':
        # hidden size 2560, attention head 20, layer number 40, bf16 2 bytes
        KV_size = 2560 * 40 * 2 * 2
        # 1 MB = 1,048,576 bytes
        KV_size = KV_size / (1048576 * 1024)
    elif args.setting.split('_')[1] == '7B':
        # hidden size 4096, attention head 32, layer number 40, bf16 2 bytes
        KV_size = 4096 * 32 * 2 * 2
        # 1 MB = 1,048,576 bytes
        KV_size = KV_size / (1048576 * 1024)
    number_of_KV = math.floor(storage_left_for_KV / KV_size)
    KV_block_number = math.floor(number_of_KV / args.block_size)
    number_of_KV = KV_block_number * args.block_size
    return number_of_KV, KV_block_number
