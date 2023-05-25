########################################################
# Copyright (c) 2022 Meta Platforms, Inc. and affiliates
#
# Holotorch is an optimization framework for differentiable wave-propagation written in PyTorch
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
#
# Contact:
# florianschiffers (at) gmail.com
# ocossairt ( at ) fb.com
#
########################################################

import torch
from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.Optical_Components.CGH_Component import CGH_Component

class Field_Mirrorer(CGH_Component):
	def __init__(self,
			mirror_horizontal : bool = True,
			mirror_vertical : bool = False
			) -> None:
		super().__init__()

		self.mirror_horizontal = mirror_horizontal
		self.mirror_vertical = mirror_vertical


	def forward(self, field : ElectricField) -> ElectricField:
		new_field = field.data
		if self.mirror_horizontal:
			new_field = new_field.flip(-1)
		if self.mirror_vertical:
			new_field = new_field.flip(-2)

		field = ElectricField(
			data = new_field,
			spacing=field.spacing,
			wavelengths=field.wavelengths
		)

		return field
