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
					collect_garbage : bool = True
				) -> None:
		super().__init__()
		self.device = device
		self.clear_cuda_cache = clear_cuda_cache
		self.collect_garbage = collect_garbage

		if not Memory_Utils._MEMORY_UTILS_ENABLED:
			Memory_Utils.initialize()

		if not (clear_cuda_cache or collect_garbage):
			warnings.warn("Unneccessary Memory_Reclaimer component")


	def forward(self, field : ElectricField) -> ElectricField:
		if self.device is not None:
			workingDevice = self.device
		else:
			workingDevice = field.data.device

		if self.clear_cuda_cache:
			Memory_Utils.clean_unused_reserved_cuda_memory(workingDevice)

		if self.collect_garbage:
			gc.collect()
		
		return field