################################################################################################################################

# There is probably a way to do the automatic cache cleaning with PyTorch, but I was not able to figure out how to set the relevant settings.

################################################################################################################################
import torch
import numpy as np
################################################################################################################################

RESERVED_MEM_CLEAR_CACHE_THRESHOLD = -1
ALLOC_TO_RESERVED_RATIO_CLEAR_CACHE_THRESHOLD = -1
_MEMORY_UTILS_ENABLED = False

################################################################################################################################

def initialize(
				RESERVED_MEM_CLEAR_CACHE_THRESHOLD_INIT : float = 0.7,
				ALLOC_TO_RESERVED_RATIO_CLEAR_CACHE_THRESHOLD_INIT : float = 0.8
			):
	if (RESERVED_MEM_CLEAR_CACHE_THRESHOLD_INIT > 1) or (RESERVED_MEM_CLEAR_CACHE_THRESHOLD_INIT < 0):
		raise Exception("'RESERVED_MEM_CLEAR_CACHE_THRESHOLD_INIT' should be between 0.0 and 1.0 (inclusive).")
	if (ALLOC_TO_RESERVED_RATIO_CLEAR_CACHE_THRESHOLD_INIT > 1) or (ALLOC_TO_RESERVED_RATIO_CLEAR_CACHE_THRESHOLD_INIT < 0):
		raise Exception("'ALLOC_TO_RESERVED_RATIO_CLEAR_CACHE_THRESHOLD_INIT' should be between 0.0 and 1.0 (inclusive).")
	global RESERVED_MEM_CLEAR_CACHE_THRESHOLD, ALLOC_TO_RESERVED_RATIO_CLEAR_CACHE_THRESHOLD, _MEMORY_UTILS_ENABLED
	RESERVED_MEM_CLEAR_CACHE_THRESHOLD = RESERVED_MEM_CLEAR_CACHE_THRESHOLD_INIT
	ALLOC_TO_RESERVED_RATIO_CLEAR_CACHE_THRESHOLD = ALLOC_TO_RESERVED_RATIO_CLEAR_CACHE_THRESHOLD_INIT
	_MEMORY_UTILS_ENABLED = True

################################################################################################################################

def get_tensor_size_bytes(tensorInput : torch.Tensor):
	return tensorInput.nelement() * tensorInput.element_size()


def print_cuda_memory_usage(device : torch.device, printShort = False):
	gpu_mem_allocated = torch.cuda.memory_allocated(device)
	gpu_mem_reserved = torch.cuda.memory_reserved(device)
	gpu_mem_total = torch.cuda.get_device_properties(device).total_memory
	if not printShort:
		gpu_info_printout_lines = torch.cuda.memory_summary(device=device,abbreviated=True).split('\n')
		gpu_info_printout_str = '\n'.join([gpu_info_printout_lines[i] for i in [0,1,4,5,6,7,11,17,26]])
		print(gpu_info_printout_str)
		print('  Memory Usage (Reserved): %.2f GB / %.2f GB  -  %.2f%%' % (gpu_mem_reserved/(1024**3), gpu_mem_total/(1024**3), (gpu_mem_reserved/gpu_mem_total)*100))
	else:
		print('  Allocated: %.2f GB\tReserved: %.2f GB\tTotal: %.2f GB' % (gpu_mem_allocated/(1024**3), gpu_mem_reserved/(1024**3), gpu_mem_total/(1024**3)))


# Clears the cache when MEMORY_UTILS_ENABLED is set to True and ONE of the following conditions is met:
#	1. A sufficiently large portion of the total GPU memory is reverved (>= RESERVED_MEM_CLEAR_CACHE_THRESHOLD)
#	2. Allocated memory comprises a sufficiently small portion of reserved memory (<= ALLOC_TO_RESERVED_RATIO_CLEAR_CACHE_THRESHOLD)
def clean_unused_reserved_cuda_memory(device : torch.device):
	if (not _MEMORY_UTILS_ENABLED) or (device.type != 'cuda'):
		return
	
	gpu_mem_allocated = torch.cuda.memory_allocated(device)
	gpu_mem_reserved = torch.cuda.memory_reserved(device)
	gpu_mem_total = torch.cuda.get_device_properties(device).total_memory

	if (gpu_mem_reserved == 0) or (gpu_mem_total == 0):
		# Handling potential cases of dividing by zero.
		return
	
	allocatedFraction = gpu_mem_allocated / gpu_mem_reserved
	reservedFraction = gpu_mem_reserved / gpu_mem_total
	if (reservedFraction >= RESERVED_MEM_CLEAR_CACHE_THRESHOLD) or (allocatedFraction <= ALLOC_TO_RESERVED_RATIO_CLEAR_CACHE_THRESHOLD):
		torch.cuda.empty_cache()