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
from holotorch.Optical_Components.Optical_Aperture import Optical_Aperture
from holotorch.utils.Helper_Functions import generateGrid_MultiRes

class Radial_Optical_Aperture(Optical_Aperture):
	def __init__(self,
			aperture_radius : float,
			off_x : float = 0.0,
			off_y : float = 0.0,
			cutoff_slope = 1e4
			) -> None:
		super().__init__()

		self.add_attribute('aperture_radius')
		self.add_attribute('off_x')
		self.add_attribute('off_y')

		self.aperture_radius_opt    = False
		self.off_x_opt              = False
		self.off_y_opt              = False

		self.aperture_radius        = torch.tensor(aperture_radius)
		self.off_x                  = torch.tensor(off_x)
		self.off_y                  = torch.tensor(off_y)
		self.cutoff_slope			= torch.tensor(cutoff_slope)


	def forward(self, field : ElectricField) -> ElectricField:
		# spacing = field.spacing
		# x = torch.linspace(-0.5,0.5, field.height).to(field.data.device)
		# y = torch.linspace(-0.5,0.5, field.width).to(field.data.device)
		# X,Y = torch.meshgrid(x,y)
		# X = (field.height * spacing.data_tensor[:,:,0] * X[None,None]) - self.off_x
		# Y = (field.width * spacing.data_tensor[:,:,1] * Y[None,None]) - self.off_y

		X, Y = generateGrid_MultiRes((field.height, field.width), field.spacing.data_tensor, device=field.data.device)
		X = X - self.off_x
		Y = Y - self.off_y

		R = torch.sqrt(X**2 + Y**2)

		self.mask = torch.sigmoid(self.cutoff_slope*(self.aperture_radius - R))

		new_field = field.data * self.mask[None,None,:,:,:,:]

		field = ElectricField(
			data = new_field,
			spacing=field.spacing,
			wavelengths=field.wavelengths
		)

		return field
