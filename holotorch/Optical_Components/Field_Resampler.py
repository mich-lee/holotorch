import numpy as np
import sys
import torch
from torch.nn.functional import grid_sample
import matplotlib.pyplot as plt

from holotorch.Spectra.SpacingContainer import SpacingContainer
from holotorch.Optical_Components.CGH_Component import CGH_Component
import holotorch.utils.Dimensions as Dimension
from holotorch.CGH_Datatypes.ElectricField import ElectricField

from holotorch.utils.Helper_Functions import generateGrid, get_center_hw_inds_wrt_ft


class Field_Resampler(CGH_Component):
	def __init__(	self,
					outputHeight				: int,
					outputWidth					: int,
					outputPixel_dx				: float,
					outputPixel_dy				: float,
					magnification 				: float = None,
					amplitudeScaling			: float = None,
					interpolationMode			: str = 'bicubic',
					reducePickledSize			: bool = True,
					device						: torch.device = None,
					gpu_no						: int = 0,
					use_cuda					: bool = False
				) -> None:
		
		super().__init__()

		if (device != None):
			self.device = device
		else:
			self.device = torch.device("cuda:"+str(gpu_no) if use_cuda else "cpu")
			self.gpu_no = gpu_no
			self.use_cuda = use_cuda

		if (type(outputHeight) is not int) or (outputHeight <= 0):
			raise Exception("Bad argument: 'outputHeight' should be a positive integer.")
		if (type(outputWidth) is not int) or (outputWidth <= 0):
			raise Exception("Bad argument: 'outputWidth' should be a positive integer.")
		if ((type(outputPixel_dx) is not float) and (type(outputPixel_dx) is not int)) or (outputPixel_dx <= 0):
			raise Exception("Bad argument: 'outputPixel_dx' should be a positive real number.")
		if ((type(outputPixel_dy) is not float) and (type(outputPixel_dy) is not int)) or (outputPixel_dy <= 0):
			raise Exception("Bad argument: 'outputPixel_dy' should be a positive real number.")

		self.outputResolution = (outputHeight, outputWidth)
		self.outputSpacing = (outputPixel_dx, outputPixel_dy)
		self.magnification = magnification
		self.amplitudeScaling = amplitudeScaling
		self.interpolationMode = interpolationMode
		self._reducePickledSize = reducePickledSize
		
		self.grid = None
		self.prevFieldSpacing = None
		self.prevFieldSize = None


	def __getstate__(self):
		if self._reducePickledSize:
			return super()._getreducedstate(fieldsSetToNone=['grid','prevFieldSpacing','prevFieldSize'])
		else:
			return super().__getstate__()


	@classmethod
	def _getFieldResamplerParams(cls, targetField : ElectricField):
		spacing = targetField.spacing.data_tensor
		lastDimSize = spacing.shape[-1]
		numSpacingElem = spacing.numel()
		if (lastDimSize != 1) and (lastDimSize != 2):
			raise Exception("Malformed spacing tensor")
		if (numSpacingElem != 1) and (numSpacingElem != 2):
			# Spacing differs across B/T/P/C dimension(s).  Averaging.
			spacing = torch.tensor([spacing[... , 0].mean(), spacing[... , 1].mean()]).expand(1,1,2)
			
		inputPixel_dx = spacing[... , 0].squeeze()
		if numSpacingElem > 1:
			inputPixel_dy = spacing[... , 1].squeeze()
		else:
			inputPixel_dy = inputPixel_dx

		inputHeight = targetField.data.shape[-2]
		inputWidth = targetField.data.shape[-1]

		return int(inputHeight), int(inputWidth), float(inputPixel_dx), float(inputPixel_dy)


	def updateTargetOutput(self, outputHeight : int, outputWidth : int, outputPixel_dx : float, outputPixel_dy : float):
		tempRes = (outputHeight, outputWidth)
		tempSpacing = (outputPixel_dx, outputPixel_dy)
		if (tempRes != self.outputResolution) or (tempSpacing != self.outputSpacing):
			# This will force the grid to be rebuilt the next time forward(...) is called
			self.grid = None
			self.prevFieldSpacing = None
			self.prevFieldSize = None


	def calculateOutputCoordGrid(self):
		grid = torch.zeros(self.outputResolution[0], self.outputResolution[1], 2, device=self.device)

		# Can assume that coordinate (0,0) is in the center due to how generateGrid(...) works
		gridX, gridY = generateGrid(self.outputResolution, self.outputSpacing[0], self.outputSpacing[1], centerGrids=True, centerCoordsAroundZero=True)
		# gridX = gridX.to(device=self.device)
		# gridY = gridY.to(device=self.device)

		# Stuff is ordered this way because torch.nn.functiona.grid_sample(...) has x as the coordinate in the width direction
		# and y as the coordinate in the height dimension.  This is the opposite of the convention used by Holotorch.
		grid[:,:,0] = gridY
		grid[:,:,1] = gridX

		# self.gridPrototype = grid.to(device=self.device)
		return grid


	@classmethod
	def analyzeSpacingContainer(cls, spacing : SpacingContainer):
		spacing_data = spacing.data_tensor.view(spacing.tensor_dimension.get_new_shape(new_dim=Dimension.BTPCHW))

		singletonSpacingDims = (torch.tensor(spacing_data.shape[:-2]) == 1)
		otherSpacingDims = ~singletonSpacingDims

		# Getting index numbers and computing number of singleton/non-singleton dimensions.
		# Converting to long is done to deal with cases where there are no dims
		#	(no dims --> returned tensor empty when indexing --> tensor defaults to float32 --> cannot use that datatype to index later on; thus need to cast to long.)
		singletonSpacingDims = torch.arange(4)[singletonSpacingDims].to(dtype=torch.long)
		otherSpacingDims = torch.arange(4)[otherSpacingDims].to(dtype=torch.long)
		numSingletonSpacingDims = int(torch.tensor(spacing_data.shape)[singletonSpacingDims].prod())
		numOtherSpacingDims = int(torch.tensor(spacing_data.shape)[otherSpacingDims].prod())

		return spacing_data, singletonSpacingDims, otherSpacingDims, numSingletonSpacingDims, numOtherSpacingDims
	
	
	def forward(self, field : ElectricField):
		# spacing_data = field.spacing.data_tensor.view(field.spacing.tensor_dimension.get_new_shape(new_dim=Dimension.BTPCHW))
		spacing_data, singletonSpacingDims, otherSpacingDims, numSingletonSpacingDims, numOtherSpacingDims = Field_Resampler.analyzeSpacingContainer(field.spacing)

		# Computing permutation indices
		permutationInds = otherSpacingDims.tolist() + singletonSpacingDims.tolist() + [4, 5]
		permutationIndsInverse = torch.zeros(6, dtype=torch.long)
		permutationIndsInverse[permutationInds] = torch.arange(6, dtype=torch.long)
		permutationIndsInverse = permutationIndsInverse.tolist()

		# Moving non-singleton B, T, P, and/or C spacing dimensions to the beginning.
		spacing_data = spacing_data.permute(permutationInds)

		# Combining singleton dimensions into a single dimension (dimension 1).  Combining non-singleton dimensions into a single dimension (dimension 0).
		# Result is a 4D tensor.
		spacing_data = spacing_data.view(numOtherSpacingDims, numSingletonSpacingDims, spacing_data.shape[-2], spacing_data.shape[-1])


		# Rearranging the field data tensor in a similar manner as the spacing_data tensor.
		field_data = field.data
		numSingletonFieldDims = int(torch.tensor(field.data.shape)[singletonSpacingDims].prod())
		numOtherFieldDims = int(torch.tensor(field.data.shape)[otherSpacingDims].prod())
		field_data = field_data.permute(permutationInds)
		if not field_data.is_contiguous():
			field_data = field_data.contiguous()
		field_data = field_data.view(numOtherFieldDims, numSingletonFieldDims, field_data.shape[-2], field_data.shape[-1])


		# Storing field data dimension sizes as variables
		Bf,Tf,Pf,Cf,Hf,Wf = field.data.shape


		# # Convert spacing tensor to 4D
		# Bs,Ts,Ps,Cs,Hs,Ws = spacing_data.shape
		# spacing_data = spacing_data.view(Bs*Ts*Ps,Cs,Hs,Ws)	# Shape to 4D

		# # convert field to 4D tensor for batch processing
		# Bf,Tf,Pf,Cf,Hf,Wf = field.data.shape
		# field_data = field.data.view(Bf*Tf*Pf,Cf,Hf,Wf) # Shape to 4D


		buildGridFlag = False
		if (self.grid is None):
			# No grid was ever made or the grid was cleared, so must make one
			buildGridFlag = True
		elif (self.prevFieldSpacing is None): # This 'elif' is redundant as (self.prevFieldSpacing is None) if and only if (self.grid is None)
			buildGridFlag = True
		elif (self.prevFieldSize is None): # This 'elif' is redundant as (self.prevFieldSize is None) if and only if (self.grid is None)
			buildGridFlag = True
		elif not (torch.equal(self.prevFieldSpacing, field.spacing.data_tensor)):
			buildGridFlag = True
		elif not (torch.equal(torch.tensor(self.prevFieldSize), torch.tensor(field.data.shape))):
			buildGridFlag = True

		if (buildGridFlag):
			# Calculating stuff for normalizing the output coordinates to the input coordinates
			xNorm = spacing_data[:,:,0,:] * ((Hf - 1) // 2)
			xNorm = xNorm[:,:,None,:].squeeze(-1)
			yNorm = spacing_data[:,:,1,:] * ((Wf - 1) // 2)
			yNorm = yNorm[:,:,None,:].squeeze(-1)

			# self.grid = self.gridPrototype.repeat(Bf*Tf*Pf,1,1,1)
			self.grid = self.calculateOutputCoordGrid().to(device=field.data.device)
			self.grid = self.grid.expand(torch.Size([numOtherSpacingDims]) + self.grid.shape[-3:]).clone()	# The clone() is necessary as we are doing in-place operations on 'grid'.  See the documentation on torch.expand(...) for more details.

			# Stuff is ordered this way because torch.nn.functiona.grid_sample(...) has x as the coordinate in the width direction
			# and y as the coordinate in the height dimension.  This is the opposite of the convention used by Holotorch.
			self.grid[... , 0] = self.grid[... , 0] / yNorm
			self.grid[... , 1] = self.grid[... , 1] / xNorm

			# self.grid = self.grid.expand(torch.Size(torch.cat((torch.tensor([Bf*Tf*Pf]), torch.tensor(self.grid.shape)))))
			self.grid = self.grid.view(torch.Size([numOtherSpacingDims]) + self.grid.shape[-3:])

			if (self.magnification is not None):
				fieldOriginXInd, fieldOriginYInd = get_center_hw_inds_wrt_ft(field_data)
				centerXNormed = (fieldOriginXInd / ((field.data.shape[-2] - 1) / 2)) - 1
				centerYNormed = (fieldOriginYInd / ((field.data.shape[-1] - 1) / 2)) - 1
				
				expandedSize = torch.Size(torch.cat((torch.ones(len(self.grid.shape) - 1, dtype=torch.int), torch.tensor([2]))))
				centerCoord = torch.tensor([centerYNormed, centerXNormed], device=self.grid.device).view(expandedSize)

				self.grid = ((self.grid - centerCoord) / self.magnification) + centerCoord


		self.prevFieldSpacing = field.spacing.data_tensor
		self.prevFieldSize = field.data.shape

		
		new_data = grid_sample(field_data.real, self.grid, mode=self.interpolationMode, padding_mode='zeros', align_corners=True)
		new_data = new_data + (1j * grid_sample(field_data.imag, self.grid, mode=self.interpolationMode, padding_mode='zeros', align_corners=True))

		tempDims = torch.Size(torch.tensor(field.data.shape)[otherSpacingDims]) + torch.Size(torch.tensor(field.data.shape)[singletonSpacingDims]) + (self.outputResolution[0], self.outputResolution[1])
		new_data = new_data.view(tempDims).permute(permutationIndsInverse)

		# new_data = new_data.view(Bf,Tf,Pf,Cf,self.outputResolution[0],self.outputResolution[1]) # Reshape to 6D

		if (self.amplitudeScaling is not None):
			new_data = new_data * self.amplitudeScaling

		# Assumes that the last dimension of the input field's spacing data tensor contains the x- and y-spacings
		new_spacing_data = torch.clone(field.spacing.data_tensor)	# Apparently, before clone() was used, new_spacing_data shared a pointer with field.spacing.data_tensor
																	# This caused unexpected behavior.
		# new_spacing_data[... , 0] = self.outputSpacing[0]
		# new_spacing_data[... , 1] = self.outputSpacing[1]
		# spacing = SpacingContainer(spacing=new_spacing_data, tensor_dimension=field.spacing.tensor_dimension)
		# spacing.set_spacing_center_wavelengths(spacing.data_tensor)

		new_spacing_data = torch.tensor([self.outputSpacing[0], self.outputSpacing[1]], device=field.spacing.data_tensor.device).expand(1,1,2)
		spacing = SpacingContainer(spacing=new_spacing_data, tensor_dimension=Dimension.TCD(n_time=1, n_channel=1, height=2))

		Eout = 	ElectricField(
					data = new_data,
					wavelengths = field.wavelengths,
					spacing = spacing
				)

		return Eout