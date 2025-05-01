import math, sys, config
from heapq_utils import *
from utils import bcolors, round_2

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

        heapify(new_heap)
        self.heap = new_heap

    def update_ongoing_requests(self, ongoing_request_list):
        self.ongoing_request_ids = {request.user_request_id for request in ongoing_request_list}

    def usable_tokens_for(self, request_id):
        if request_id in self.request_id2blocks and request_id in self.request_id2tokens:
            return self.request_id2blocks[request_id] * self.block_size - self.request_id2tokens[request_id]
        return 0

    def allocate_memory_for(self, user_request, requested_tokens, full_queue):
        """

        :param user_request:
        :param requested_tokens:
        :return:
        """
        res = {'preempted_requests': [], 'deallocated_requests': []}    # {{preempted_request_id: list of request object}, {deallocated_request_id: list of request object}}
        request_id = user_request.user_request_id

        requested_blocks = math.ceil((requested_tokens - self.usable_tokens_for(request_id)) / self.block_size)

        if self.total_blocks < requested_blocks:
            print(bcolors.FAIL + "Out Of Memory: Not enough memory for the current request" + bcolors.ENDC)
            sys.exit(1)

        if request_id not in self.request_id2blocks:
            self.request_id2blocks[request_id] = 0
            self.request_id2tokens[request_id] = 0
            self.push(user_request, user_request.priority, user_request.predicted_remaining_computation_time[-1] + user_request.prefill_cache_loading_time + user_request.decoding_cache_loading_time)

        if self.storage_left() >= requested_blocks:
            self.request_id2blocks[request_id] += requested_blocks
            self.request_id2tokens[request_id] += requested_tokens
            self.used_blocks += requested_blocks
            return res
        
        else:
            requested_blocks_remaining = requested_blocks
            requested_blocks_remaining -= self.storage_left()

            # deallocate the memory from the lowest priority request
            while requested_blocks_remaining > 0:
                try:
                    lowest_priority_request = self.pop()
                except:
                    print(self.heap)

                lowest_priority_request_id = lowest_priority_request.user_request_id
                if lowest_priority_request_id in self.ongoing_request_ids:
                    self.ongoing_request_ids.remove(lowest_priority_request_id)
                    res['preempted_requests'].append(lowest_priority_request)
                    
                if lowest_priority_request_id == request_id:
                    # if the lowest priority request is the same as the current request
                    # then we just preempt this request, and we don't need to deallocate any memory
                    assert lowest_priority_request in res['preempted_requests']
                    remaining_time = lowest_priority_request.predicted_remaining_computation_time[-1] + lowest_priority_request.prefill_cache_loading_time + lowest_priority_request.decoding_cache_loading_time
                    self.push(lowest_priority_request, lowest_priority_request.priority, remaining_time)

                    return res

                # lazy deletion of the completed request and the preempted request due to the failed memory allocation
                if lowest_priority_request_id not in self.request_id2blocks:
                    continue
                if self.request_id2blocks[lowest_priority_request_id] == 0:
                    self.request_id2blocks.pop(lowest_priority_request_id)
                    self.request_id2tokens.pop(lowest_priority_request_id)
                    continue

                total_tokens_on_cache = self.request_id2tokens[lowest_priority_request.user_request_id] if lowest_priority_request.user_request_id in self.request_id2tokens else 0
                # deleted the impacted request from the full queue
                # in order to updated its position in the queue
                if lowest_priority_request not in res['preempted_requests']:
                    full_queue.two_dim_priority_queue.delete(lowest_priority_request)
                
                # if requested blocks are more than the available blocks after deallocation of this request
                if self.request_id2blocks[lowest_priority_request_id] <= requested_blocks_remaining:
                    # update used blocks
                    self.used_blocks -= self.request_id2blocks[lowest_priority_request_id]
                    # still blocks remaining to deallocate
                    requested_blocks_remaining -= self.request_id2blocks[lowest_priority_request_id]
                    # update request2blocks and request2tokens
                    self.request_id2blocks.pop(lowest_priority_request_id)
                    self.request_id2tokens.pop(lowest_priority_request_id)
                else:
                    # update used blocks
                    self.used_blocks -= requested_blocks_remaining
                    # update request2blocks and request2tokens
                    self.request_id2blocks[lowest_priority_request_id] -= requested_blocks_remaining
                    self.request_id2tokens[lowest_priority_request_id] = self.request_id2blocks[lowest_priority_request_id] * self.block_size
                    # no more blocks to deallocate
                    requested_blocks_remaining = 0

                # update time for lowest_priority_request_id by either swapping memory or deleting memory
                tokens_remaining = self.request_id2tokens[lowest_priority_request_id] if lowest_priority_request_id in self.request_id2tokens else 0
                lowest_priority_request.swap_or_delete_update(remaining_tokens_on_GPU=tokens_remaining, total_tokens_on_cache=total_tokens_on_cache)
                remaining_time = lowest_priority_request.predicted_remaining_computation_time[-1] + lowest_priority_request.prefill_cache_loading_time + lowest_priority_request.decoding_cache_loading_time
                if tokens_remaining > 0:
                    self.push(lowest_priority_request, lowest_priority_request.priority, remaining_time)

                res['deallocated_requests'].append(lowest_priority_request)

            # atomic memory allocation for the request
            self.request_id2blocks[request_id] += requested_blocks
            self.request_id2tokens[request_id] += requested_tokens
            self.used_blocks += requested_blocks

        return res

    def deallocate_memory_for(self, user_request):
        request_id = user_request.user_request_id
        if request_id in self.ongoing_request_ids:
            self.ongoing_request_ids.remove(request_id)

        if request_id in self.request_id2blocks:
            self.used_blocks -= self.request_id2blocks[request_id]
            self.request_id2blocks.pop(request_id)
            self.request_id2tokens.pop(request_id)
            # heap deletion is deferred to the future allocation

        self.heap = [r for r in self.heap if r[2].user_request_id != user_request.user_request_id]
        heapify(self.heap)

    def visualize(self):
        # visualize a heap 
        def print_heap_tree(heap, index=0, indent="", branch="Root: "):
            """
            Recursively prints a heap as a tree.
            
            :param heap: List representing the heap
            :param index: Current index within 'heap'
            :param indent: Current indentation (used internally)
            :param branch: Label of the current branch (Root, L---, R---)
            """
            if index < len(heap):
                # Print the current node
                print(indent + branch + str((heap[index][2].user_request_id, heap[index][2].predicted_priority, heap[index][2].predicted_remaining_computation_time[-1] + heap[index][2].prefill_cache_loading_time + heap[index][2].decoding_cache_loading_time)))
                
                # Prepare for children
                left_index = 2 * index + 1
                right_index = 2 * index + 2
                
                # Increase indentation for children
                child_indent = indent + "    "
                
                # Recursively print left and right subtrees
                print_heap_tree(heap, left_index, child_indent, "L--- ")
                print_heap_tree(heap, right_index, child_indent, "R--- ")

        print(bcolors.OKCYAN + "Memory Management for KV Cache (MaxHeap):" + bcolors.ENDC)
        print_heap_tree(self.heap)


def compute_GPU_KV_storage_size(args):
    total_GPU_memory = args.GPU_memory * args.GPU_memory_utilization

    print('total_GPU_memory', total_GPU_memory)
    model_size = int(args.setting.split('_')[1][-2]) * 4
    activation_memory = total_GPU_memory * args.activation_memory_percentage
    storage_left_for_KV = total_GPU_memory - activation_memory - model_size

    print('storage_left_for_KV', storage_left_for_KV)

    assert storage_left_for_KV > 0, "GPU too small for the current model"
    
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

    print('KV_block_number', KV_block_number)
    print('number_of_KV', number_of_KV)

    return number_of_KV, KV_block_number
