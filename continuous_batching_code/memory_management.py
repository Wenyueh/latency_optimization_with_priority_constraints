import math, sys 
from heapq import heappush, heappop


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
    def __init__(self):
        pass  

    # how many tokens have occupied this block
    def space_taken_by_block(self, block_id):
        pass
    
    # how many tokens can stil be stored in this block
    def space_left_in_block(self, block_id):
        pass

    # the GPU is full or not
    def is_full(self):
        # check whether all blocks are full
        pass

    # how many left
    def storage_left(self):
        pass

    # KV: {request_id: [token1, token2, ...]}
    def insert(self, KV):
        pass

    def remove(self, size):
        pass




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
