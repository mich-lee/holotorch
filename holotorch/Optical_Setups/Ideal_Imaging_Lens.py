import numpy as np
import sys
import torch
from torch.nn.functional import grid_sample
import matplotlib.pyplot as plt

from holotorch.Spectra.SpacingContainer import SpacingContainer
from holotorch.Optical_Components.CGH_Component import CGH_Component
import holotorch.utils.Dimensions as Dimension
from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.Optical_Components.Field_Resampler import Field_Resampler

from holotorch.utils.Helper_Functions import generateGrid


# Note that this does not model diffraction effects nor does it model the constant phase
# factors that the image would pick up (the latter is usually ignored anyways, however).
class Ideal_Imaging_Lens(CGH_Component):
	def __init__(	self,
					focal_length			: float,
					object_dist				: float,
					interpolationMode		: str = 'bicubic',
					rescaleCoords			: bool = False,
					rescaleFactor			: float = None,
					device					: torch.device = None,
					gpu_no					: int = 0,
					use_cuda				: bool = False
				) -> None:

		if (rescaleFactor is not None) and (rescaleFactor <= 0):
			raise Exception("Invalid value for 'rescaleFactor'.  Should be a positive real number.")
		
		super().__init__()

		if (device != None):
			self.device = device
		else:
			self.device = torch.device("cuda:"+str(gpu_no) if use_cuda else "cpu")
			self.gpu_no = gpu_no
			self.use_cuda = use_cuda

		self.focal_length = focal_length
		self.object_dist = object_dist
		self.interpolationMode = interpolationMode
		self.rescaleCoords = rescaleCoords
		self.rescaleFactor = rescaleFactor
		self._fieldResampler = None

		self._calculateGeometricOpticsParameters()


	def _calculateGeometricOpticsParameters(self):
		f = self.focal_length
		d_o = self.object_dist

		d_img = f * d_o / (d_o - f)
		M = -f / (d_o - f)

		self.image_dist = d_img
		self.magnification = M

	
	def forward(self, field : ElectricField) -> ElectricField:
		inputHeight, inputWidth, inputPixel_dx, inputPixel_dy = Field_Resampler._getFieldResamplerParams(targetField=field)
		if self.rescaleCoords:
			if self.rescaleFactor is None:
				outputPixel_dx = inputPixel_dx * abs(self.magnification)
				outputPixel_dy = inputPixel_dy * abs(self.magnification)
			else:
				outputPixel_dx = inputPixel_dx * self.rescaleFactor
				outputPixel_dy = inputPixel_dy * self.rescaleFactor
		else:
			outputPixel_dx = inputPixel_dx
			outputPixel_dy = inputPixel_dy
		if self._fieldResampler is not None:
			self._fieldResampler.updateTargetOutput(inputHeight, inputWidth, outputPixel_dx, outputPixel_dy)
		else:
			self._fieldResampler = Field_Resampler	(
														outputHeight=inputHeight, outputWidth=inputWidth,
														outputPixel_dx=outputPixel_dx, outputPixel_dy=outputPixel_dy,
														magnification=self.magnification,
														amplitudeScaling=1/abs(self.magnification),
														interpolationMode=self.interpolationMode
													)

		# temp_spacing_data = torch.clone(field.spacing.data_tensor)
		# temp_spacing_data = temp_spacing_data * abs(self.magnification)
		# tempSpacing = SpacingContainer(spacing=temp_spacing_data, tensor_dimension=field.spacing.tensor_dimension)
		# tempSpacing.set_spacing_center_wavelengths(tempSpacing.data_tensor)
		# E_temp = ElectricField(
		# 				data = field.data,
		# 				wavelengths = field.wavelengths,
		# 				spacing = tempSpacing
		# 			)
		# E_out = self._fieldResampler(E_temp)
		
		E_out = self._fieldResampler(field)
		return E_out