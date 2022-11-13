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

def get_cuda_memory_stats(device : torch.device):
	allocated = torch.cuda.memory_allocated(device)
	reserved = torch.cuda.memory_reserved(device)
	allocated_max = torch.cuda.max_memory_allocated(device)
	reserved_max = torch.cuda.max_memory_reserved(device)
	total = torch.cuda.get_device_properties(device).total_memory

	allocated_gb = allocated/(1024**3)
	reserved_gb = reserved/(1024**3)
	allocated_max_gb = allocated_max/(1024**3)
	reserved_max_gb = reserved_max/(1024**3)
	total_gb = total/(1024**3)

	allocated_percent = (allocated / total) * 100
	reserved_percent = (reserved / total) * 100
	allocated_max_percent = (allocated_max / total) * 100
	reserved_max_percent = (reserved_max / total) * 100

	statsDict =	{
					'allocated'				: allocated,
					'allocated_gb'			: allocated_gb,
					'allocated_percent'		: allocated_percent,
					'allocated_max'			: allocated_max,
					'allocated_max_gb'		: allocated_max_gb,
					'allocated_max_percent'	: allocated_max_percent,
					'reserved'				: reserved,
					'reserved_gb'			: reserved_gb,
					'reserved_percent'		: reserved_percent,
					'reserved_max'			: reserved_max,
					'reserved_max_gb'		: reserved_max_gb,
					'reserved_max_percent'	: reserved_max_percent,
					'total'					: total,
					'total_gb'				: total_gb
				}

	return statsDict

def print_cuda_memory_usage(device : torch.device, printType = 0):
	statsDict = get_cuda_memory_stats(device)
	if printType == 0:
		gpu_info_printout_lines = torch.cuda.memory_summary(device=device,abbreviated=True).split('\n')
		gpu_info_printout_str = '\n'.join([gpu_info_printout_lines[i] for i in [0,1,4,5,6,7,11,17,26]])
		print(gpu_info_printout_str)
		print('  Memory Usage (Reserved): %.2f GB / %.2f GB  -  %.2f%%' % (statsDict['reserved_gb'], statsDict['total_gb'], statsDict['reserved_percent']))
	elif printType == 1:
		print('  Allocated: %.2f GB\tReserved: %.2f GB\tTotal: %.2f GB' % (statsDict['allocated_gb'], statsDict['reserved_gb'], statsDict['total_gb']))
	elif printType == 2:
		gpu_info_printout_vals =	(
										statsDict['allocated_gb'], statsDict['allocated_percent'], statsDict['reserved_gb'], statsDict['reserved_percent'],
										statsDict['allocated_max_gb'], statsDict['allocated_max_percent'], statsDict['reserved_max_gb'], statsDict['reserved_max_percent'],
										statsDict['total_gb']
									)
		gpu_info_printout_str = "  | Cur Alloc | %.2f GB\t%.2f%%\t|\t| Cur Res | %.2f GB\t%.2f%%\t|\t| Peak Alloc | %.2f GB\t%.2f%%\t|\t| Peak Res | %.2f GB\t%.2f%%\t|\t| Total VRAM: %.2f GB |"
		print(gpu_info_printout_str % gpu_info_printout_vals)
	else:
		raise Exception("Invalid 'printType' argument.  Should 'printType' should equal 0, 1, or 2.")
		


# Clears the CUDA cache when MEMORY_UTILS_ENABLED is set to True and ONE of the following conditions is met:
#	1. A sufficiently large portion of the total GPU memory is reverved (>= RESERVED_MEM_CLEAR_CACHE_THRESHOLD)
#	2. Allocated memory comprises a sufficiently small portion of reserved memory (<= ALLOC_TO_RESERVED_RATIO_CLEAR_CACHE_THRESHOLD)
# Returns True when the CUDA cache is cleared and False when it is not.
def clean_unused_reserved_cuda_memory(device : torch.device, force_clean : bool = False):
	if (not _MEMORY_UTILS_ENABLED) or (device.type != 'cuda'):
		return False
	
	gpu_mem_allocated = torch.cuda.memory_allocated(device)
	gpu_mem_reserved = torch.cuda.memory_reserved(device)
	gpu_mem_total = torch.cuda.get_device_properties(device).total_memory

	if (gpu_mem_reserved == 0) or (gpu_mem_total == 0):
		# Handling potential cases of dividing by zero.
		return False
	
	allocatedFraction = gpu_mem_allocated / gpu_mem_reserved
	reservedFraction = gpu_mem_reserved / gpu_mem_total
	if (force_clean) or (reservedFraction >= RESERVED_MEM_CLEAR_CACHE_THRESHOLD) or (allocatedFraction <= ALLOC_TO_RESERVED_RATIO_CLEAR_CACHE_THRESHOLD):
		torch.cuda.empty_cache()
		return True

	return False