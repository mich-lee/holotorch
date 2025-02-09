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

import numpy as np
import torch
import torchvision
from torch.nn.functional import pad
import matplotlib.pyplot as plt

import warnings
import copy

from holotorch.Optical_Components.CGH_Component import CGH_Component
from holotorch.Optical_Propagators.Propagator import Propagator
import holotorch.utils.Dimensions as Dimensions
from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.utils.Helper_Functions import ft2, ift2, fft2_inplace, ifft2_inplace
from holotorch.utils.Enumerators import *
import holotorch.utils.Memory_Utils as Memory_Utils


class ASM_Prop(Propagator):

	def __init__(	self,
					init_distance						: float = 0.0,
					z_opt								: bool = False,
					prop_kernel_type					: ENUM_PROP_KERNEL_TYPE = ENUM_PROP_KERNEL_TYPE.FULL_KERNEL,
					prop_computation_type				: str = 'TF',
					utilize_cpu							: bool = False,
					do_padding							: bool = True,
					do_unpad_after_pad					: bool = True,
					padding_scale						: float or torch.Tensor = None,
					memoize_prop_kernel					: bool = True,
					do_ffts_inplace						: bool = False,
					sign_convention						: ENUM_PHASE_SIGN_CONVENTION = ENUM_PHASE_SIGN_CONVENTION.TIME_PHASORS_ROTATE_CLOCKWISE,
					bandlimit_kernel					: bool = True,
					bandlimit_type						: str = 'exact',
					bandlimit_kernel_fudge_factor_x		: float = 1.0,
					bandlimit_kernel_fudge_factor_y		: float = 1.0,
					reducePickledSize					: bool = True
				):
		"""
		Angular Spectrum method with bandlimited ASM from Digital Holographic Microscopy
		Principles, Techniques, and Applications by K. Kim
		Eq. 4.22 (page 50)

		Args:
			init_distance (float, optional): initial propagation distance. Defaults to 0.0.

			z_opt (bool, optional): is the distance parameter optimizable or not. Defaults to False

			prop_kernel_type (str, optional):	ENUM_PROP_KERNEL_TYPE.FULL_KERNEL		-	Uses the full ASM kernel
												ENUM_PROP_KERNEL_TYPE.PARAXIAL_KERNEL	-	Uses the paraxial/Fresnel approximation for the kernel
												Defaults to ENUM_PROP_KERNEL_TYPE.FULL_KERNEL.

			prop_computation_type (str, optional):	Default -->	'TF'	- Computes the propagated fields using a 'transfer function' approach.
																			- Can think of this as applying a transfer function in the frequency domain/using the frequency domain kernel.
																			- Procedure:
																				- Zero-pad input field to have dimensions B x T x P x C x H_pad x W_pad
																					- Note: input field (without padding) has dimensions B x T x P x C x H x W
																				- Generate Kx and Ky grids of size T x C x H_pad x W_pad
																				- Compute kernel in the frequency domain using the Kx and Ky grids
																				- Take the FFT of the padded input field and multiply it with the kernel
																				- Take the inverse FFT of the result of the multiplication
																				- Unpad the result so it has dimensions B x T x P x C x H x W
																'IR'	- Computes the propagated fields using an 'impulse response' approach.
																			- Can think of this as convolving the input field with a propagation kernel in the space domain.
																			- Procedure:
																				- Zero-pad input field to have dimensions B x T x P x C x H_pad x W_pad
																					- Note: input field (without padding) has dimensions B x T x P x C x H x W
																				- Compute the propagation kernel in the space domain.  The dimensions should be T x C x H x W.
																					- Note that for the case (prop_computation_type='IR', prop_kernel_type=ENUM_PROP_KERNEL_TYPE.PARAXIAL_KERNEL),
																						the this class does not have an explicit formula for the space domain kernel.  Thus, the code will create
																						the kernel in the frequency domain and use an inverse FFT to get the kernel in the space domain.
																							- This feels a little strange, but it seems to work alright in some cases.
																				- Zero-pad the propagation kernel to have dimensions T x C x H_pad x W_pad.
																				- Take the FFTs of the padded input field and the propagation kernel, and then multiply those FFTs together
																				- Take the inverse FFT of the result of the multiplication
																				- Unpad the result so it has dimensions B x T x P x C x H x W

												Reference Note:	See Chapter 5 in "Computational Fourier Optics: A MATLAB Tutorial" by David Voelz for more information on
																transfer function (TF) versus impulse response (IR) propagators.

			utilize_cpu (bool, optional):	When set to true, the process of applying the propagation kernel to the input field will be done on the CPU.
											This can help with insufficient GPU memory issues as RAM will be used instead of GPU RAM, although using the CPU will likely be slower.
											Defaults to False.

			do_padding (bool, optional):	Determines whether or not to pad the input field data before doing calculations.
											Padding can help reduce convolution edge artifacts, but will increase the size of the data processed.
											Defaults to True.

			do_unpad_after_pad (bool, optional):	This determines whether or not to unpad the field data before returning an ElectricField object.
													If 'do_padding' is set to False, 'do_unpad_after_pad' has no effect
													Otherwise:
														- If 'do_unpad_after_pad' is set to True, then the field data is unpadded to its original size, i.e. the size of the input field's data.
														- If 'do_unpad_after_pad' is set to False, then no unpadding is done.  The field data returned will be of the padded size.
													Defaults to True.

			padding_scale (float, tuple, tensor; optional):		Determines how much padding to apply to the input field.
																Padding is applied symmetrically so the data is centered in the height and width dimensions.
																'padding_scale' must be a non-negative real-valued number, a 2-tuple containing non-negative real-valued numbers, or a tensor containing two non-negative real-valued numbers.

																Examples:
																	Example 1:
																		- Input field dimensions: height=50, width=100
																		- padding_scale = 1
																		- Padded field dimensions: height=100, width=200	<--- (50 + 1*50, 100 + 1*100)
																	Example 1:
																		- Input field dimensions: height=50, width=100
																		- padding_scale = torch.tensor([1,2])
																		- Padded field dimensions: height=100, width=300	<--- (50 + 1*50, 100 + 2*100)

			memoize_prop_kernel (bool, optional):	Determines whether or not to save the propagation kernel in-between forward passes.
													This should generally be set to True as it reduces redundant computations.
													However, setting this to False can possibly help memory consumption as the propagation kernels are freed up to be collected by the garbage collector, rather than saved.
													Defaults to True.

			do_ffts_inplace (bool, optional):	Determines whether or not to do the 2D FFTs/IFFTs in-place.
												Setting this to True can help prevent one from running out of memory.
												Note that the out-of-place FFTs/IFFTs are faster.
												Defaults to False.

			sign_convention (ENUM_PHASE_SIGN_CONVENTION):	Determines what sign convention to assume for time domain phasors.
															Cases:
																If sign_convention == ENUM_PHASE_SIGN_CONVENTION.TIME_PHASORS_ROTATE_CLOCKWISE:
																	- Assuming phasors have a time dependence of the form exp(-j 2\pi \omega t)
																	- This is the convention adopted by Goodman's Fourier optics book (3rd edition) and Digital Holographic Microscopy by Myung K. Kim.
																If sign_convention == ENUM_PHASE_SIGN_CONVENTION.TIME_PHASORS_ROTATE_CLOCKWISE:
																	- Assuming phasors have a time dependence of the form exp(-j 2\pi \omega t)
																	- This is the convention often used in electrical engineering.
															Defaults to ENUM_PHASE_SIGN_CONVENTION.TIME_PHASORS_ROTATE_CLOCKWISE

			bandlimit_kernel (bool, optional):	Determines whether or not to apply the bandlimiting described in Band-Limited ASM (Matsushima et al, 2009) to the ASM kernel
													- bandlimit_kernel = True will apply the bandlimiting, bandlimit_kernel = False will not apply the bandlimiting
												Note that evanescent wave components will be filtered out regardless of what this is set to.
												Defaults to True

			bandlimit_type (str, optional):		If bandlimit_kernel is set to False, then this option does nothing.
												If bandlimit_kernel is set to True, then:
													'approx' - Bandlimits the propagation kernel based on Equations 21 and 22 in Band-Limited ASM (Matsushima et al, 2009)
													'exact' - Bandlimits the propagation kernel based on Equations 18 and 19 in Band-Limited ASM (Matsushima et al, 2009)
												Note that for aperture sizes that are small compared to the propagation distance, 'approx' and 'exact' will more-or-less the same results.
												Defaults to 'exact'
			
			bandlimit_kernel_fudge_factor_x (float, optional):	See source code.  Defaults to 1.0.
			
			bandlimit_kernel_fudge_factor_y (float, optional):	See source code.  Defaults to 1.0.

			reducePickledSize (bool, optional):		When this is set to true, certain temporary data is not saved when pickling.
													Specifically, the following fields are saved with their value set to None:
														- prop_kernel
														- prop_kernel_spacing
														- prop_kernel_wavelengths
														- prop_kernel_field_shape
													Defaults to True
		"""

		if (sign_convention != ENUM_PHASE_SIGN_CONVENTION.TIME_PHASORS_ROTATE_CLOCKWISE):
			raise Exception("Need to investigate whether or not the Fourier transform sign convention changes from e^{-j\omega t} for the forward transform if time phasors rotate counterclockwise.")
		if (prop_computation_type != 'TF') and (prop_computation_type != 'IR'):
			raise Exception("Invalid input for argument 'prop_computation_type'.  Should be either 'TF' (transfer function) or 'IR' (impulse response).")
		if ((bandlimit_type != 'approx') and (bandlimit_type != 'exact')):
			raise Exception("Invalid input for argument 'bandlimit_type'.  Should be either 'approx' (approximate) or 'exact' (exact).")

		super().__init__()

		self.add_attribute(attr_name="z")

		DEFAULT_PADDING_SCALE = torch.tensor([1,1])
		if do_padding:
			paddingScaleErrorFlag = False
			if not torch.is_tensor(padding_scale):
				if padding_scale == None:
					padding_scale = DEFAULT_PADDING_SCALE
				elif np.isscalar(padding_scale):
					padding_scale = torch.tensor([padding_scale, padding_scale])
				else:
					padding_scale = torch.tensor(padding_scale)
					if padding_scale.numel() != 2:
						paddingScaleErrorFlag = True
			elif padding_scale.numel() == 1:
				padding_scale = padding_scale.squeeze()
				padding_scale = torch.tensor([padding_scale, padding_scale])
			elif padding_scale.numel() == 2:
				padding_scale = padding_scale.squeeze()
			else:
				paddingScaleErrorFlag = True
			
			if (paddingScaleErrorFlag):
				raise Exception("Invalid value for argument 'padding_scale'.  Should be a real-valued non-negative scalar number or a two-element tuple/tensor containing real-valued non-negative scalar numbers.")
		else:
			padding_scale = None

		# store the input params
		self.sign_convention 					= sign_convention
		self.prop_kernel_type					= prop_kernel_type
		self.prop_computation_type				= prop_computation_type
		self.utilize_cpu						= utilize_cpu
		self.do_padding							= do_padding
		self.do_unpad_after_pad					= do_unpad_after_pad
		self.padding_scale						= padding_scale
		self.memoize_prop_kernel				= memoize_prop_kernel
		self.do_ffts_inplace					= do_ffts_inplace
		self.bandlimit_kernel					= bandlimit_kernel
		self.bandlimit_type						= bandlimit_type
		self.bandlimit_kernel_fudge_factor_x	= bandlimit_kernel_fudge_factor_x
		self.bandlimit_kernel_fudge_factor_y	= bandlimit_kernel_fudge_factor_y
		self._reducePickledSize					= reducePickledSize
		self.z_opt								= z_opt
		self.z									= init_distance

		self.prop_kernel = None
		self.prop_kernel_spacing = None
		self.prop_kernel_wavelengths = None
		self.prop_kernel_field_shape = None

		self.prevParameters = self.createParameterDict()


	def __getstate__(self):
		if self._reducePickledSize:
			return super()._getreducedstate(fieldsSetToNone=['prop_kernel','prop_kernel_spacing','prop_kernel_wavelengths', 'prop_kernel_field_shape'])
		else:
			return super().__getstate__()


	def createParameterDict(self):
		parameterDict =	{
							'sign_convention'					: self.sign_convention,
							'prop_kernel_type'					: self.prop_kernel_type,
							'prop_computation_type'				: self.prop_computation_type,
							'do_padding'						: self.do_padding,
							'do_unpad_after_pad'				: self.do_unpad_after_pad,
							'memoize_prop_kernel'				: self.memoize_prop_kernel,
							'do_ffts_inplace'					: self.do_ffts_inplace,
							'padding_scale'						: self.padding_scale,
							'bandlimit_kernel'					: self.bandlimit_kernel,
							'bandlimit_kernel_fudge_factor_x'	: self.bandlimit_kernel_fudge_factor_x,
							'bandlimit_kernel_fudge_factor_y'	: self.bandlimit_kernel_fudge_factor_y,
							'bandlimit_type'					: self.bandlimit_type,
							'z'									: self.z
						}
		return parameterDict


	def compute_padding(self, H, W, return_size_of_padding = False):
		# Get the shape for processing
		if not self.do_padding:
			paddingH = 0
			paddingW = 0
			paddedH = int(H)
			paddedW = int(W)
		else:
			paddingH = int(np.floor((float(self.padding_scale[0]) * H) / 2))
			paddingW = int(np.floor((float(self.padding_scale[1]) * W) / 2))
			paddedH = H + 2*paddingH
			paddedW = W + 2*paddingW

		if not return_size_of_padding:
			return paddedH, paddedW
		else:
			return paddingH, paddingW


	def visualize_kernel(self,
			field : ElectricField,
		):
		kernel = self.update_kernel(field = field)

		plt.subplot(121)
		plt.imshow(kernel.abs().cpu().squeeze(),vmin=0)
		plt.title("Amplitude")
		plt.subplot(122)
		plt.imshow(kernel.angle().cpu().squeeze())
		plt.title("Phase")
		plt.tight_layout()


	def update_kernel(self, field : ElectricField, forceRebuild: bool = False):
		def checkContainersEqual(c1, c2):
			if not torch.equal(c1.data_tensor, c2.data_tensor):
				return False
			if c1.tensor_dimension.id != c2.tensor_dimension.id:
				return False
			return True

		rebuildFlag = False
		if (forceRebuild == True):
			rebuildFlag = True
		elif self.prop_kernel is None:
			rebuildFlag = True
		elif not torch.equal(torch.tensor(field.data.shape), torch.tensor(self.prop_kernel_field_shape)):
			rebuildFlag = True
		elif not checkContainersEqual(field.wavelengths, self.prop_kernel_wavelengths):
			rebuildFlag = True
		elif not checkContainersEqual(field.spacing, self.prop_kernel_spacing):
			rebuildFlag = True
		
		# This is its own 'if' because changing parameters means that the kernel needs to be rebuilt, regardless of what the input field is.
		if self.prevParameters != self.createParameterDict():
			# Not the most efficient to keep checking like this, but this is probably trivial compared to the main computation.
			rebuildFlag = True
			self.prevParameters = self.createParameterDict()

		if (rebuildFlag == False):
			return

		self.generate_propagation_kernel(field)
		self.prop_kernel = self.prop_kernel.to(device=field.data.device)
		self.prop_kernel_field_shape = field.data.shape
		self.prop_kernel_wavelengths = copy.deepcopy(field.wavelengths)
		self.prop_kernel_spacing = copy.deepcopy(field.spacing)
		

	def generate_propagation_kernel(self, field : ElectricField):
		def create_normalized_grid(H, W, device):
			# precompute frequency grid for ASM defocus kernel
			with torch.no_grad():
				# Creates the frequency coordinate grid in x and y direction
				kx = (torch.linspace(0, H - 1, H) - (H // 2)) / H
				ky = (torch.linspace(0, W - 1, W) - (W // 2)) / W
				Kx, Ky = torch.meshgrid(kx, ky)
				return Kx.to(device=device), Ky.to(device=device)


		if (self.prop_computation_type == 'TF'):
			# The Kx and Ky grids will be the same size as the padded field data
			tempShape = torch.tensor(field.shape)
			tempShapeH, tempShapeW = self.compute_padding(tempShape[-2], tempShape[-1])
			tempShape[-2] = tempShapeH
			tempShape[-1] = tempShapeW
			tempShape = torch.Size(tempShape)
		elif (self.prop_computation_type == 'IR'):
			# The Kx and Ky grids will be the same size as the unpadded field data
			tempShape = field.shape
		else:
			raise Exception("Invalid option for 'prop_computation_type'.")

		_,_,_,_,H_new,W_new = tempShape




		# get the wavelengths data as a TxC tensor
		new_shape       = field.wavelengths.tensor_dimension.get_new_shape(new_dim=Dimensions.TC)
		wavelengths_TC  = field.wavelengths.data_tensor.view(new_shape) # T x C
		wavelengths_TC  = wavelengths_TC[:,:,None,None]		# Expand wavelengths for H and W dimension

		# extract dx, dy spacing into T x C tensors
		spacing = field.spacing.data_tensor
		dx      = spacing[:,:,0]
		if spacing.shape[2] > 1:
			dy = spacing[:,:,1]
		else:
			dy = dx

		# get the spacing data as a TxC tensor
		dx_TC   = dx.expand(new_shape)
		dx_TC   = dx_TC[:,:,None,None] # Expand to H and W
		dy_TC   = dy.expand(new_shape)
		dy_TC   = dy_TC[:,:,None,None] # Expand to H and W

		# compute wavenumbers
		K_lambda = 2*np.pi /  wavelengths_TC	# T x C x H x W
		K_lambda_2 = K_lambda**2 				# T x C x H x W




		#################################################################
		# Propagation Using Impulse Response
		#################################################################
		# This is the impulse response propagator case (see Chapter 5 of "Computational Fourier Optics: A MATLAB Tutorial" by David Voelz)
		# This kernel will be space domain convolved with the field to be propagated
		#################################################################
		if (self.prop_computation_type == 'IR'):
			if (self.prop_kernel_type is ENUM_PROP_KERNEL_TYPE.PARAXIAL_KERNEL):
				if (self.z == 0):
					raise Exception("Cannot have the propagation distance be zero when using a paraxial kernel with prop_computation_type == 'IR'.")

				# Get normalized grids.  Values are on the interval [-0.5,0.5).
				gridX, gridY = create_normalized_grid(H_new, W_new, field.data.device)
				gridX = gridX[None,None,:,:]
				gridY = gridY[None,None,:,:]

				# Scale the grid so it is centered around (0,0) and the grid coordinates are spaced one apart
				gridX = gridX * gridX.shape[-2]
				gridY = gridY * gridY.shape[-1]
				if ((gridX.shape[-2] % 2) == 0): # Even length in dimension so need to shift over by 0.5
					gridX = gridX + 0.5
				if ((gridY.shape[-1] % 2) == 0): # Even length in dimension so need to shift over by 0.5
					gridY = gridY + 0.5

				# Scale the grids to the correct size
				gridX = gridX * dx_TC
				gridY = gridY * dy_TC

				kernelOut = (1 / (1j * wavelengths_TC * self.z)) * torch.exp(1j * K_lambda * self.z) * torch.exp((1j*K_lambda/(2*self.z)) * ((gridX**2) + (gridY**2)))
				kernelOut = kernelOut * (dx_TC * dy_TC)		# Space domain kernel

				pad_x, pad_y = self.compute_padding(field.data.shape[-2], field.data.shape[-1], return_size_of_padding=True)
				kernelOut = pad(kernelOut, (pad_y, pad_y, pad_x, pad_x), mode='constant', value=0)

				if self.do_ffts_inplace:
					fft2_inplace(kernelOut, norm='backward')
				else:
					kernelOut = ft2(kernelOut, norm='backward')
				
				self.prop_kernel = kernelOut
				return
			elif (self.prop_kernel_type is ENUM_PROP_KERNEL_TYPE.FULL_KERNEL):
				raise Exception("Cannot use an impulse response propagator for this case.  A closed-form expression for the space-domain kernel is not available.")
			else:
				raise Exception("Unknown kernel type.")
		#################################################################




		#################################################################
		# Propagation using transfer functions
		#################################################################
		# MORE on ASM Kernels here in these sources:
		#	Digital Holographic Microscopy - Principles, Techniques, and Applications
		#	Also some information in Goodman's Fourier optics book (3rd edition)
		#################################################################
		# create the frequency grid for each T x C wavelength/spacing combo
		Kx, Ky = create_normalized_grid(H_new, W_new, field.data.device)
		Kx = 2*np.pi * Kx[None,None,:,:] / dx_TC
		Ky = 2*np.pi * Ky[None,None,:,:] / dy_TC
		K2 = Kx**2 + Ky**2

		if self.prop_kernel_type is ENUM_PROP_KERNEL_TYPE.PARAXIAL_KERNEL:
			# NOTES:
			#	- The expression "ang = self.z * K_lambda - self.z/(2*K_lambda)*K2" should hopefully be the correct expression
			#		- Sources:
			#			- Equation 4.23 in "Digital Holographic Microscopy: Principles, Techniques, and Applications" by Myung K. Kim
			#			- Equation 4-20 and 4-22 in "Introduction to Fourier Optics" (3rd Edition) by Joseph W. Goodman
			#	- SIDE NOTES: The Fresnel/paraxial approximation allows one to avoid the square root operation.  This is good because that operation is expensive.
			ang = self.z * K_lambda - self.z/(2*K_lambda)*K2	# T x C x H x W
		elif self.prop_kernel_type is ENUM_PROP_KERNEL_TYPE.FULL_KERNEL:
			ang = self.z * torch.sqrt(K_lambda_2 - K2) # T x C x H x W
		else:
			raise Exception("Unknown kernel type.")

		# Adjust angle to match sign convention
		#	For more information, see Section 4.2.1 in "Introduction to Fourier Optics" (3rd Edition) by Joseph W. Goodman
		if (self.sign_convention == ENUM_PHASE_SIGN_CONVENTION.TIME_PHASORS_ROTATE_CLOCKWISE):
			# Assuming phasors have a time dependence of the form exp(-j 2\pi \omega t)
			pass # Do nothing---the calculations for 'ang' should have assumed this convention.
		elif (self.sign_convention == ENUM_PHASE_SIGN_CONVENTION.TIME_PHASORS_ROTATE_COUNTERCLOCKWISE):
			# Assuming phasors have a time dependence of the form exp(+j 2\pi \omega t)
			ang = -ang
		else:
			raise Exception("Invalid value for 'sign_convention'.")

		kernelOut =  torch.exp(1j * ang)		# Compute the kernel without bandlimiting
		kernelOut[(K_lambda_2 - K2) < 0] = 0	# Remove evanescent components

		if (self.bandlimit_kernel):
			#################################################################
			# Bandlimit the kernel
			# see band-limited ASM - Matsushima et al. (2009)
			# K. Matsushima and T. Shimobaba,
			# "Band-Limited Angular Spectrum Method for Numerical Simulation of Free-Space Propagation in Far and Near Fields,"
			#  Opt. Express  17, 19662-19673 (2009).
			#################################################################
			# Some equations:
			#	delta_kx = (2*pi / dx) / nHeight,	delta_ky = (2*pi / dy) / nWidth
			#	delta_u = delta_kx / (2*pi)		,	delta_v = delta_ky / (2*pi)
			#	u_limit = 1 / [lambda * sqrt( (2*deltaU*z)^2 + 1)]						(Equation 13)
			#	v_limit = 1 / [lambda * sqrt( (2*deltaV*z)^2 + 1)]						(Equation 20)
			delta_u = ((2*np.pi / dx_TC) / field.height) / (2*np.pi)
			delta_v = ((2*np.pi / dy_TC) / field.width) / (2*np.pi)
			u_limit = 1 / torch.sqrt( ((2*delta_u*self.z)**2) + 1 ) / wavelengths_TC
			v_limit = 1 / torch.sqrt( ((2*delta_v*self.z)**2) + 1 ) / wavelengths_TC
			if (self.bandlimit_type == 'exact'):
				# Precise constraints on frequency:
				#	constraint1 = Kx^2 / (2*pi*u_lim)^2 + Ky^2 / k^2 <= 1			(From Equation 18, substituting in Equation 13, and making the substitutions Kx = 2*pi*u, Ky = 2*pi*v, and k = 2*pi/lambda)
				#	constraint2 = Kx^2 / k^2 + Ky^2 / (2*pi*v_lim)^2 <= 1			(From Equation 19, substituting in Equation 20, and making substitutions Kx = 2*pi*u, Ky = 2*pi*v, and k = 2*pi/lambda)
				constraint1 = (((Kx**2) / ((2*np.pi*u_limit)**2)) + ((Ky**2) / (K_lambda**2))) <= (1 * self.bandlimit_kernel_fudge_factor_x)
				constraint2 = (((Kx**2) / (K_lambda**2)) + ((Ky**2) / ((2*np.pi*v_limit)**2))) <= (1 * self.bandlimit_kernel_fudge_factor_y)
				combinedConstraints = constraint1 & constraint2
				kernelOut[~combinedConstraints] = 0		# <--- Better for memory usage as one is not allocating an entire tensor for a bandlimiting filter
			elif (self.bandlimit_type == 'approx'):
				# Approximate constraints on frequency:
				#	k_x_max_approx = 2*pi * [1 / sqrt((2*deltaU*z)^2 + 1)] * (1 / lambda)			(From Equation 21, substituting in Equation 13, and making the substitutions Kx_max = 2*pi*u_limit and deltaU = 1/length_x = 1/(dx*nHeight))
				#	k_y_max_approx = 2*pi * [1 / sqrt((2*deltaV*z)^2 + 1)] * (1 / lambda)			(From Equation 22, substituting in Equation 20, and making the substitutions Ky_max = 2*pi*v_limit and deltaV = 1/length_y = 1/(dy*nWidth))
				length_x = field.height * dx_TC
				length_y = field.width  * dy_TC
				k_x_max_approx = 2*np.pi / torch.sqrt( ((2*(1/length_x)*self.z)**2) + 1 ) / wavelengths_TC
				k_y_max_approx = 2*np.pi / torch.sqrt( ((2*(1/length_y)*self.z)**2) + 1 ) / wavelengths_TC
				k_x_max_approx = k_x_max_approx * self.bandlimit_kernel_fudge_factor_x
				k_y_max_approx = k_y_max_approx * self.bandlimit_kernel_fudge_factor_y
				kernelOut[ ( torch.abs(Kx) > k_x_max_approx) | (torch.abs(Ky) > k_y_max_approx) ] = 0		# <--- Better for memory usage as one is not allocating an entire tensor for a bandlimiting filter
			else:
				raise Exception("Invalid option for 'bandlimit_type'.")
		
		self.prop_kernel = kernelOut





	def forward(self,
			field : ElectricField,
			) -> ElectricField:
		"""
		Takes in optical field and propagates it to the instantiated distance using ASM from KIM
		Eq. 4.22 (page 50)

		Args:
			field (ElectricField): Complex field 6D tensor object

		Returns:
			ElectricField: The electric field after the rotate field propagation model
		"""

		# Update the propagation kernel (if necessary)
		self.update_kernel(field)

		# extract the data tensor from the field
		wavelengths = field.wavelengths
		field_data  = field.data

		# convert field to 4D tensor for batch processing
		B,T,P,C,H,W = field_data.shape
		field_data = field_data.view(B*T*P,C,H,W)

		# Pad 'field_data' avoid convolution wrap-around effects
		if (self.do_padding):
			pad_x, pad_y = self.compute_padding(H, W, return_size_of_padding=True)
			field_data = pad(field_data, (pad_y, pad_y, pad_x, pad_x), mode='constant', value=0)
		else:
			field_data = field_data.clone()

		_, _, H_pad,W_pad = field_data.shape

		originalPropKernelDevice = self.prop_kernel.device
		if (self.utilize_cpu):
			self.prop_kernel = self.prop_kernel.to('cpu')
			field_data = field_data.to('cpu')

		# Convert to angular spectrum/frequency domain
		if self.do_ffts_inplace:
			fft2_inplace(field_data)
		else:
			field_data = ft2(field_data)
		field_data = field_data.view(B,T,P,C,H_pad,W_pad)	# Convert 4D into 6D so that 6D propagation kernel can be applied

		# Do convolution in frequency
		field_data = field_data * self.prop_kernel[None,:,None,:,:,:]

		# Go back to the space domain
		field_data = field_data.view(B*T*P,C,H_pad,W_pad)	# Convert from 6D to 4D so IFFT can be applied
		if self.do_ffts_inplace:
			ifft2_inplace(field_data)
		else:
			field_data = ift2(field_data)

		# Unpad the field after convolution, if necessary
		if (self.do_padding and self.do_unpad_after_pad):
			center_crop = torchvision.transforms.CenterCrop([H,W])
			field_data = center_crop(field_data)

		_, _, H_out, W_out = field_data.shape


		# convert field back to 6D tensor
		field_data = field_data.view(B,T,P,C,H_out,W_out)

		# Move field_data back to original device
		if (self.utilize_cpu):
			field_data = field_data.to(field.data.device)
			self.prop_kernel = self.prop_kernel.to(originalPropKernelDevice)

		field.spacing.set_spacing_center_wavelengths(field.spacing.data_tensor)

		Eout = ElectricField(
				data=field_data,
				wavelengths=wavelengths,
				spacing=field.spacing
				)

		if not self.memoize_prop_kernel:
			self.prop_kernel = None
			self.prop_kernel_field_shape = None
			self.prop_kernel_wavelengths = None
			self.prop_kernel_spacing = None

		return Eout