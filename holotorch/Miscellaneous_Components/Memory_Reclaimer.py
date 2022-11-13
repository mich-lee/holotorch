import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt

from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.Optical_Components.CGH_Component import CGH_Component

import holotorch.utils.Memory_Utils as Memory_Utils
import gc

class Memory_Reclaimer(CGH_Component):
	def __init__(	self,
					device : torch.device = None,
					clear_cuda_cache : bool = True,
					collect_garbage : bool = True,
					print_cleaning_actions	: bool = False,
					print_memory_status	: bool = False,
					print_memory_status_printType	: int = 2
				) -> None:
		super().__init__()
		self.device = device
		self.clear_cuda_cache = clear_cuda_cache
		self.collect_garbage = collect_garbage
		self.print_cleaning_actions = print_cleaning_actions
		self.print_memory_status = print_memory_status
		self.print_memory_status_printType = print_memory_status_printType

		if not Memory_Utils._MEMORY_UTILS_ENABLED:
			Memory_Utils.initialize()

		if not (clear_cuda_cache or collect_garbage or print_memory_status or print_cleaning_actions):
			warnings.warn("Unneccessary Memory_Reclaimer component")


	def forward(self, field : ElectricField) -> ElectricField:
		if self.device is not None:
			workingDevice = self.device
		else:
			workingDevice = field.data.device

		if self.print_memory_status:
			print('-->', end='')
			Memory_Utils.print_cuda_memory_usage(device=workingDevice, printType=self.print_memory_status_printType)

		performedActionFlag = False

		if self.clear_cuda_cache:
			if self.print_cleaning_actions:
				print("Emptying CUDA cache...", end='')
			cudaCacheEmptied = Memory_Utils.clean_unused_reserved_cuda_memory(workingDevice)
			if cudaCacheEmptied:
				performedActionFlag = True
			if self.print_cleaning_actions:
				if cudaCacheEmptied:
					print("Done.", end='')
				else:
					print("Skipped.", end='')
				if self.collect_garbage:
					print("  ", end='')
				else:
					print()

		if self.collect_garbage:
			if self.print_cleaning_actions:
				print("Collecting garbage...", end='')
			gc.collect()
			performedActionFlag = True
			if self.print_cleaning_actions:
				print("Done.")

		if self.print_memory_status and performedActionFlag:
			print('   ', end='')
			Memory_Utils.print_cuda_memory_usage(device=workingDevice, printType=self.print_memory_status_printType)
		
		return field