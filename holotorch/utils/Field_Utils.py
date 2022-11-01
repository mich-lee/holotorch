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



from __future__ import print_function

import numpy as np
import torch
import matplotlib.pyplot as plt

import holotorch.utils.Dimensions as Dimensions
from holotorch.utils.Dimensions import *
from holotorch.Spectra.SpacingContainer import SpacingContainer
from holotorch.Spectra.WavelengthContainer import WavelengthContainer
from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.CGH_Datatypes.Light import Light

from holotorch.utils.Helper_Functions import *


# Explanation:
#	Given TensorDimensions old_dim and new_dim, this function returns a list of dimension indices from old_dim and a
# list of their corresponding dimension indices in new_dim.
#
# Example:
#	- Suppose old_dim had dimensions ['T','P','C'] and new_dim had dimensions ['B','T','C','H','W'].  Then, this function will return:
#		- old_indices = tensor([0, 2])
#		- new_indices = tensor([1, 2])
def get_dimension_inds_remapping(old_dim : TensorDimension, new_dim : TensorDimension):
	"""[summary]

	Args:
		dim (TensorDimension): [description]
		new_dim (TensorDimension): [description]
	"""
	new_indices_bool = np.isin(new_dim.id, old_dim.id)
	old_indices_bool = np.isin(old_dim.id, new_dim.id)

	new_indices = torch.tensor(range(len(new_dim.id)))[new_indices_bool]
	old_indices = torch.tensor(range(len(old_dim.id)))[old_indices_bool]
	
	return old_indices, new_indices


# Not bothering to check for matching dimensions here
def applyFilterToElectricField(h : torch.tensor, field : ElectricField):
	filteredField = ElectricField(
						data = applyFilterSpaceDomain(h, field.data),
						wavelengths = field.wavelengths,
						spacing = field.spacing
					)
	return filteredField


def getWavelengthAndSpacingDataAsTCHW(field : ElectricField):
	# extract dx, dy spacing into TxC tensors
	spacing = field.spacing.data_tensor
	dx      = spacing[:,:,0]
	if spacing.shape[2] > 1:
		dy = spacing[:,:,1]
	else:
		dy = dx

	# Get wavelengths as TxCxHxW tensors
	wavelengths = field.wavelengths
	new_shape       = wavelengths.tensor_dimension.get_new_shape(new_dim=Dimensions.TC) # Get TxC shape
	wavelengths_TC  = wavelengths.data_tensor.view(new_shape) # Reshape to TxC tensor
	wavelengths_TC  = wavelengths_TC[:,:,None,None] # Expand wavelengths for H and W dimension
	
	# Get dx and dy as TxCxHxW tensors
	dx_TC   = dx.expand(new_shape) # Reshape to TxC tensor
	dx_TC   = dx_TC[:,:,None,None] # Expand to H and W
	dy_TC   = dy.expand(new_shape) # Reshape to TxC tensor
	dy_TC   = dy_TC[:,:,None,None] # Expand to H and W
	
	return wavelengths_TC, dx_TC, dy_TC


# This function uses some code from ASM_Prop in the HoloTorch library
def computeSpatialFrequencyGrids(field : ElectricField):
	_, dx_TC, dy_TC = getWavelengthAndSpacingDataAsTCHW(field)

	# Initializing return values
	Kx = None
	Ky = None

	# Want values on the half-open interval [-1/2, 1/2).  Want to exclude +1/2 as it is redundant with -1/2
	Kx_vals_normed = torch.linspace(-np.floor(field.height / 2), np.floor((field.height - 1) / 2), field.height) / field.height
	Ky_vals_normed = torch.linspace(-np.floor(field.width / 2), np.floor((field.width - 1) / 2), field.width) / field.width
	
	# Making grid and ensuring that it is on the correct device
	Kx_Grid, Ky_Grid = torch.meshgrid(Kx_vals_normed, Ky_vals_normed)
	Kx_Grid = Kx_Grid.to(dx_TC.device)
	Ky_Grid = Ky_Grid.to(dy_TC.device)

	# Expand the frequency grid for T and C dimension
	Kx = 2*np.pi * Kx_Grid[None,None,:,:] / dx_TC
	Ky = 2*np.pi * Ky_Grid[None,None,:,:] / dy_TC

	return Kx, Ky


# Returns a field object that is a subset of the input field.
# Can use this to pick out certain batches, channels, regions of 2D space, etc
def get_field_slice(field : Light,
					batch_inds_range : int = None,
					time_inds_range : int = None,
					pupil_inds_range : int = None,
					channel_inds_range : int = None,
					height_inds_range : int = None,
					width_inds_range : int = None,
					field_data_tensor_dimension : TensorDimension = None, 	# For specifying the field's data tensor dimension in case it cannot
																			# be inferred from field.wavelengths.
					cloneTensors : bool = True,		# Could probably get away with this being false for many cases.
													# However, setting this to true will help assure one that data in the input argument 'field' will not get modified by this method.
													# I am not 100% sure though whether such is possible in this method.
					assume_6D_BTPCHW_dims : bool = True,
					device : torch.device = None
				):

	def checkIfIndicesValid(inds, dimSize):
		if (dimSize == 0):
			return False
		if (isinstance(inds,tuple)):
			if (len(inds) != 2):
				if not (isinstance(inds[0], int) and isinstance(inds[1], int)):
					return False
				if (inds[0] >= dimSize) or (inds[0] < 0):
					return False
				if (inds[1] > dimSize) or (inds[1] <= 0):
					return False
				if (inds[0] >= inds[1]):
					return False
		elif (isinstance(inds, int)):
			ind = inds
			if (ind >= dimSize) or (ind < 0):
				return False
		elif (inds is None):
			return True
		else:
			return False
		return True

	def getDimIndicesList(inds, curDimSize, maxDimSize):
		if not (checkIfIndicesValid(inds, maxDimSize)):
			raise Exception("Invalid indices given when trying to slice a field object!  Indices should be given in the form of an integer or a 2-tuple.")
		if (inds is None):
			return torch.tensor(range(0, min(curDimSize, maxDimSize)))
		if (curDimSize == 1):
			return torch.tensor([0])
		if (isinstance(inds,int)):
			return torch.tensor([inds])
		return torch.tensor(range(inds[0], inds[1]))
			

	if (cloneTensors):
		field_data = torch.clone(field.data)
	else:
		field_data = field.data
	wavelengths = field.wavelengths
	spacing = field.spacing

	if (device is None):
		device = field.data.device


	# I believe the intent of the people who wrote this library was that the field data tensor would always be 6D (except when the Extra dimension is added).
	# If that is the case, then the 'else' block is unnecessary.
	if assume_6D_BTPCHW_dims:
		if (len(field_data.shape) != 6):
			raise Exception("The 'assume_6D_BTPCHW_dims' argument is set to True, but the field data tensor is not 6D.  Cannot assume BTPCHW dimensions.  (NOTE: Set 'assume_6D_BTPCHW_dims' to False to automatically infer dimensions.)")
		fieldBTPCHW_shape = field_data.shape
	else:
		fieldBTPCHW_shape = torch.ones(6,dtype=int)
		if (field_data_tensor_dimension is None):
			# Trying to infer dimension labels on field_data using the wavelength container
			old_indices, new_indices = get_dimension_inds_remapping(old_dim=wavelengths.tensor_dimension, new_dim=Dimensions.BTPCHW)
			if ((len(field_data.shape) - 2) != len(wavelengths.tensor_dimension.id)):
				raise Exception("ERROR: Could not infer field data dimension labels from wavelength container.  Please manually specify with the 'field_data_tensor_dimension' argument.")
			fieldBTPCHW_shape[new_indices.tolist()] = torch.tensor(field_data.shape)[old_indices.tolist()]
			# Assuming that the wavelengths container does not have height and width dimensions...
			fieldBTPCHW_shape[4:6] = torch.tensor([field_data.shape[-2], field_data.shape[-1]])
		else:
			old_indices, new_indices = get_dimension_inds_remapping(old_dim=field_data_tensor_dimension, new_dim=Dimensions.BTPCHW)
			fieldBTPCHW_shape[new_indices.tolist()] = torch.tensor(field_data.shape)[old_indices.tolist()]
		fieldBTPCHW_shape = torch.Size(fieldBTPCHW_shape)

	wavelengthsBTPCHW_shape = wavelengths.tensor_dimension.get_new_shape(new_dim=Dimensions.BTPCHW)

	# The last dimension for spacing containers should be height, which is used to hold x and y spacing.
	# Want to split up the last dimension into two so it works with the other tensors more nicely
	if (cloneTensors):
		spacingDataX = torch.clone(spacing.data_tensor)[... , 0].unsqueeze(-1)
		spacingDataY = torch.clone(spacing.data_tensor)[... , 1].unsqueeze(-1)
	else:
		spacingDataX = spacing.data_tensor[... , 0].unsqueeze(-1)
		spacingDataY = spacing.data_tensor[... , 1].unsqueeze(-1)
	
	# Halving the height component because we want to work with one spacing coordinate at a time for now
	spacingBTPCHW_shape = spacing.tensor_dimension.get_new_shape(new_dim=Dimensions.BTPCHW)	# As of 9/7/2022, get_new_shape(...) should not modify any of its arguments.
	spacingBTPCHW_shape = torch.tensor(spacingBTPCHW_shape)	# torch.tensor(...) copies data
	spacingBTPCHW_shape[-2] = 1
	spacingBTPCHW_shape = torch.Size(spacingBTPCHW_shape)

	
	maxDim_shape = torch.Size(torch.maximum(
												torch.maximum(
																torch.tensor(fieldBTPCHW_shape),
																torch.tensor(wavelengthsBTPCHW_shape)	# This adds dimensions for height and width at the end
																														# It's being assumed that the wavelengths dimensions do not include height and width
															),
												torch.tensor(spacingBTPCHW_shape)
											)
							)

	# NOTE: Have not tested this for fields with BTCHW_E dimensions
	if (cloneTensors):
		fieldBTPCHW = torch.clone(field_data).view(fieldBTPCHW_shape).to(device)
		wavelengthsBTPCHW = torch.clone(wavelengths.data_tensor).view(wavelengthsBTPCHW_shape).to(device)
	else:
		fieldBTPCHW = field_data.view(fieldBTPCHW_shape).to(device)
		wavelengthsBTPCHW = wavelengths.data_tensor.view(wavelengthsBTPCHW_shape).to(device)
	spacingX_BTPCHW = spacingDataX.view(spacingBTPCHW_shape).to(device)		# 'spacingDataX' would have already been cloned if cloneTensors == True.  No need to clone again.
	spacingY_BTPCHW = spacingDataY.view(spacingBTPCHW_shape).to(device)		# 		Same for spacingDataY.


	# I'm assuming that broadcastability is an equivalence relation.  If that is the case, then one can infer whether or not all
	# these tensors mutually broadcastable with only three checks.
	if not (
				(check_tensors_broadcastable(fieldBTPCHW, wavelengthsBTPCHW)) and
				(check_tensors_broadcastable(fieldBTPCHW, spacingX_BTPCHW)) and
				(check_tensors_broadcastable(fieldBTPCHW, spacingY_BTPCHW))
			):
		raise Exception("Incompatible dimensions encountered.")


	for dimInd in range(6):
		if (dimInd == 0):
			inds = batch_inds_range
		elif (dimInd == 1):
			inds = time_inds_range
		elif (dimInd == 2):
			inds = pupil_inds_range
		elif (dimInd == 3):
			inds = channel_inds_range
		elif (dimInd == 4):
			inds = height_inds_range
		elif (dimInd == 5):
			inds = width_inds_range

		curMaxDimSize = maxDim_shape[dimInd]
		curDimSizeField = fieldBTPCHW_shape[dimInd]
		curDimSizeWavelengths = wavelengthsBTPCHW_shape[dimInd]
		curDimSizeSpacing = spacingBTPCHW_shape[dimInd]

		inds_field = getDimIndicesList(inds, curDimSizeField, curMaxDimSize).to(device=device)
		inds_wavelengths = getDimIndicesList(inds, curDimSizeWavelengths, curMaxDimSize).to(device=device)
		inds_spacing = getDimIndicesList(inds, curDimSizeSpacing, curMaxDimSize).to(device=device)

		fieldBTPCHW = torch.index_select(fieldBTPCHW, dimInd, inds_field)
		wavelengthsBTPCHW = torch.index_select(wavelengthsBTPCHW, dimInd, inds_wavelengths)
		spacingX_BTPCHW = torch.index_select(spacingX_BTPCHW, dimInd, inds_spacing)
		spacingY_BTPCHW = torch.index_select(spacingY_BTPCHW, dimInd, inds_spacing)


	# Putting the spacing data tensor back together
	spacingBTPCHW = torch.cat((spacingX_BTPCHW, spacingY_BTPCHW), 4)

	# Converting the spacing from BTPCHW dimensions to TCD dimensions
	#	This is being done because it seems like parts of the Holotorch library code---e.g. parts of ASM_Prop.py---assume TCD dimensions
	#	for the SpacingContainer dimensions.
	spacingTCD_dim = Dimensions.TCD(n_time=spacingBTPCHW.shape[1], n_channel=spacingBTPCHW.shape[3], height=spacingBTPCHW.shape[4])
	spacingTCD = spacingBTPCHW.view(spacingTCD_dim.shape)
	

	new_wavelength_container =	WavelengthContainer(	wavelengths = wavelengthsBTPCHW,
														tensor_dimension = Dimensions.BTPCHW(	n_batch = wavelengthsBTPCHW.shape[0],
																								n_time = wavelengthsBTPCHW.shape[1],
																								n_pupil = wavelengthsBTPCHW.shape[2],
																								n_channel = wavelengthsBTPCHW.shape[3],
																								height = wavelengthsBTPCHW.shape[4],
																								width = wavelengthsBTPCHW.shape[5],
																							)
													)
	new_wavelength_container = new_wavelength_container.to(device=device)

	new_spacing_container =	SpacingContainer(spacing = spacingTCD, tensor_dimension = spacingTCD_dim)
	new_spacing_container = new_spacing_container.to(device=device)
	new_spacing_container.set_spacing_center_wavelengths(new_spacing_container.data_tensor)

	newField = 	ElectricField(
					data = fieldBTPCHW,
					wavelengths = new_wavelength_container,
					spacing = new_spacing_container
				)

	return newField