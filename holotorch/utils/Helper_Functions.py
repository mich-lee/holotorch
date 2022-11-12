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

import copy
import re

# Image wranglers
from PIL import Image

import warnings
from typing import Union

import holotorch.utils.transformer_6d_4d as transformer_6d_4d

import holotorch.utils.Memory_Utils as Memory_Utils

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 


def replace_bkwd(fwd: torch.Tensor, bkwd: torch.Tensor):
    new         = bkwd.clone()  # contains backwardFn from bkwd
    new.data    = fwd.data      # copies data from fwd  
    return new

def replace_fwd(fwd : torch.Tensor, bkwd : torch.Tensor): 
    bkwd.data = fwd; 
    return bkwd

def set_default_device(device: Union[str, torch.device]):
    if not isinstance(device, torch.device):
        device = torch.device(device)
        
    print(device)

    if device.type == 'cuda':
        print("TEST")
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.cuda.set_device(device.index)
        print("CUDA1")
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
        print("CUDA2")

def total_variation(input: torch.tensor):
    '''
    compute centerered finite difference derivative for input along dimensions dim
    zero pad the boundary

    input: 4D torch tensor with dimension 2,3 being the spatial difference
    returns: scalar value |dx|_1 + |dy|_1  
    '''
    # reshape if its a 6D tensor
    if input.ndim == 6:
        B,T,P,C,H,W = input.shape
        input = input.view(B*T*P,C,H,W)

    dx, dy = center_difference(input)
    return dx.abs().mean() + dy.abs().mean()

def center_difference(input: torch.tensor):
    '''
    compute centerered finite difference derivative for input along dimensions dim
    zero pad the boundary

    input: 4D torch tensor with dimension 2,3 being the spatial difference
    returns: dx, dy - 4D tensors same size as input 
    '''
    # create a new tensor of zeros for zeropadding
    dx = torch.zeros_like(input)
    dy = torch.zeros_like(input)
    _, _, H, W = input.shape
    dx[:,:,:,1:-1] = W/4*(-input[:,:,:,0:-2] + 2*input[:,:,:,1:-1] - input[:,:,:,2:])
    dy[:,:,1:-1,:] = H/4*(-input[:,:,0:-2,:] + 2*input[:,:,1:-1,:] - input[:,:,2:,:])
    return dx, dy

def tt(x):
    return torch.tensor(x)

def regular_grid4D(M,N,H,W, range=tt([[-1,1],[-1,1],[-1,1],[-1,1]]), device=torch.device("cpu")):
    '''
    Create a regular grid 4D tensor with dims M x N x H x W specified within a range 
    '''
    #Coordinates                 
    x = torch.linspace(range[0,0], range[0,1], M, device=device)  
    y = torch.linspace(range[1,0], range[1,1], N, device=device)  
    u = torch.linspace(range[2,0], range[2,1], H, device=device)  
    v = torch.linspace(range[3,0], range[3,1], W, device=device)  

    #Generate Coordinate Mesh and store it in the model
    return torch.meshgrid(x,y,u,v)    

def regular_grid2D(H,W, range=tt([[-1,1],[-1,1]]), device=torch.device("cpu")):
    '''
    Create a regular grid 2D tensor with dims H x W specified within a range 
    '''
    #XCoordinates                 
    x_c = torch.linspace(range[0,0], range[0,1], W, device=device)  
    #YCoordinates 
    y_c = torch.linspace(range[1,0], range[1,1], H, device=device)  
    #Generate Coordinate Mesh and store it in the model
    return torch.meshgrid(x_c,y_c)    

def ft2(input, delta=1, norm = 'ortho', pad = False):
    """
    Helper function computes a shifted fourier transform with optional scaling
    """
    return perform_ft(
        input=input,
        delta = delta,
        norm = norm,
        pad = pad,
        flag_ifft=False
    )

def ift2(input, delta=1, norm = 'ortho', pad = False):
    
    return perform_ft(
        input=input,
        delta = delta,
        norm = norm,
        pad = pad,
        flag_ifft=True
    )

def perform_ft(input, delta=1, norm = 'ortho', pad = False, flag_ifft : bool = False):
    
    # Get the initial shape (used later for transforming 6D to 4D)
    tmp_shape = input.shape

    # Save Size for later crop
    Nx_old = int(input.shape[-2])
    Ny_old = int(input.shape[-1])
        
    # Pad the image for avoiding convolution artifacts
    if pad == True:
        
        pad_scale = 1
        
        pad_nx = int(pad_scale * Nx_old / 2)
        pad_ny = int(pad_scale * Ny_old / 2)
        
        input = torch.nn.functional.pad(input, (pad_nx,pad_nx,pad_ny,pad_ny), mode='constant', value=0)
    
    if flag_ifft == False:
        myfft = torch.fft.fft2
        my_fftshift = torch.fft.fftshift
    else:
        myfft = torch.fft.ifft2
        my_fftshift = torch.fft.ifftshift
    
    # Compute the Fourier Transform
    out = (delta**2)* my_fftshift( myfft (  my_fftshift (input, dim=(-2,-1))  , dim=(-2,-1), norm=norm)  , dim=(-2,-1))
    
    if pad == True:
        input_size = [Nx_old, Ny_old]
        pool = torch.nn.AdaptiveAvgPool2d(input_size)
        out = transformer_6d_4d.transform_6D_to_4D(tensor_in=out)
        
        if out.is_complex():
            out = pool(out.real) + 1j * pool(out.imag)
        else:
            out = pool(out)
        new_shape = (*tmp_shape[:4],*out.shape[-2:])

        out = transformer_6d_4d.transform_4D_to_6D(tensor_in=out, newshape=new_shape)

    return out


################################################################################################################################
# Some in-place 2D FFT implementations.  Useful for conserving GPU VRAM/memory since the out-of-place FFTs allocate new memory.
################################################################################################################################

def fft2_inplace(x : torch.tensor, centerOrigins : bool = True, norm : str = 'ortho'):
	return _fft2_inplace_helper(x=x, centerOrigins=centerOrigins, norm=norm, inverse_fft_flag=False)

def ifft2_inplace(x : torch.tensor, centerOrigins : bool = True, norm : str = 'ortho'):
	return _fft2_inplace_helper(x=x, centerOrigins=centerOrigins, norm=norm, inverse_fft_flag=True)

def _fft2_inplace_helper(x : torch.tensor, centerOrigins : bool = True, norm : str = 'ortho', inverse_fft_flag : bool = False):
	if not inverse_fft_flag:
		fft_func = torch.fft.fft
		fftshift_func = torch.fft.fftshift
	else:
		fft_func = torch.fft.ifft
		fftshift_func = torch.fft.ifftshift

	for r in range(x.shape[-2]):
		if centerOrigins:
			x[...,r,:] = fftshift_func(fft_func(fftshift_func(x[...,r,:], dim=-1), dim=-1, norm=norm), dim=-1)
		else:
			x[...,r,:] = fft_func(x[...,r,:], dim=-1, norm=norm)

	for c in range(x.shape[-1]):
		if centerOrigins:
			x[...,:,c] = fftshift_func(fft_func(fftshift_func(x[...,:,c], dim=-1), dim=-1, norm=norm), dim=-1)
		else:
			x[...,:,c] = fft_func(x[...,:,c], dim=-1, norm=norm)

	return x

################################################################################################################################


def generateGrid(res, deltaX, deltaY, centerGrids = True, centerAroundZero = True, device=None):
	if (torch.is_tensor(deltaX)):
		deltaX = copy.deepcopy(deltaX).squeeze().to(device=device)
	if (torch.is_tensor(deltaY)):
		deltaY = copy.deepcopy(deltaY).squeeze().to(device=device)

	if (centerGrids):
		if (centerAroundZero):
			xCoords = torch.linspace(-((res[0] - 1) // 2), (res[0] - 1) // 2, res[0]).to(device=device) * deltaX
			yCoords = torch.linspace(-((res[1] - 1) // 2), (res[1] - 1) // 2, res[1]).to(device=device) * deltaY
		else:
			xCoords = (torch.linspace(0, res[0] - 1, res[0]) - (res[0] // 2)).to(device=device) * deltaX
			yCoords = (torch.linspace(0, res[1] - 1, res[1]) - (res[1] // 2)).to(device=device) * deltaY
	else:
		xCoords = torch.linspace(0, res[0] - 1, res[0]).to(device=device) * deltaX
		yCoords = torch.linspace(0, res[1] - 1, res[1]).to(device=device) * deltaY

	xGrid, yGrid = torch.meshgrid(xCoords, yCoords)

	return xGrid, yGrid


# Resizes image while keeping its aspect ratio.  Will make the resized image as big as possible without
# exceeding the resolution set by 'targetResolution'.
#	- For example, if the target resolution is 600x400 pixels, a 200x200 pixel input image will be resized to 400x400 pixels.
def fit_image_to_resolution(inputImage, targetResolution):
	def getNumChannels(shape):
		if (len(shape) == 2):
			return 1
		elif (len(shape) == 3):
			return shape[2]
		else:
			raise Exception("Unrecognized image data shape.")

	inputImageAspectRatio = inputImage.size[0] / inputImage.size[1]
	targetAspectRatio = targetResolution[1] / targetResolution[0]

	if (targetAspectRatio == inputImageAspectRatio):
		outputImage = inputImage.resize((targetResolution[1], targetResolution[0]))
	elif (inputImageAspectRatio < targetAspectRatio):
		# Width relatively undersized so should resize to match height
		imageMag = targetResolution[0] / inputImage.size[1]																		# = (Target resolution height) / (Input image height)
		imageMagWidth = np.int(np.floor(inputImage.size[0] * imageMag))															# = floor((Input image width) * imageMag)
		resizedImageData = np.asarray(inputImage.resize((imageMagWidth, targetResolution[0])))									# Resize input image to match target resolution's height, then convert image to array
		paddingOffset = (targetResolution[1] - imageMagWidth) // 2																# Calculate how much more width the target resolution has relative to the input image, divide that number by 2, and round down
		numChannels = getNumChannels(resizedImageData.shape)
		paddedImageData = np.zeros([targetResolution[0], targetResolution[1], numChannels])										# Initialize new array for image data (array indices represent height, width, and color/alpha channels respectively)
		if (numChannels == 1):
			resizedImageData = resizedImageData[:,:,None]
		paddedImageData[:,paddingOffset:(paddingOffset+imageMagWidth),:] = resizedImageData										# Put resizedImageData array into paddedImageData, with the resizedImageData array being centered in the width dimension
		if (numChannels == 1):
			paddedImageData = np.squeeze(paddedImageData)
			outputImage = Image.fromarray(paddedImageData.astype(np.uint8), mode='L')
		else:
			outputImage = Image.fromarray(paddedImageData.astype(np.uint8))														# Convert paddedImageData to an image object
	else:
		# Height relatively undersized so should resize to match width
		imageMag = targetResolution[1] / inputImage.size[0]																		# = (Target resolution width) / (Input image width)
		imageMagHeight = np.int(np.floor(inputImage.size[1] * imageMag))														# = floor((Input image height) * imageMag)
		resizedImageData = np.asarray(inputImage.resize((targetResolution[1], imageMagHeight)))									# Resize input image to match target resolution's width, then convert image to array
		paddingOffset = (targetResolution[0] - imageMagHeight) // 2																# Calculate how much more height the target resolution has relative to the input image, divide that number by 2, and round down
		numChannels = getNumChannels(resizedImageData.shape)
		paddedImageData = np.zeros([targetResolution[0], targetResolution[1], numChannels])										# Initialize new array for image data (array indices represent height, width, and color/alpha channels respectively)
		if (numChannels == 1):
			resizedImageData = resizedImageData[:,:,None]
		paddedImageData[paddingOffset:(paddingOffset+imageMagHeight),:,:] = resizedImageData									# Put resizedImageData array into paddedImageData, with the resizedImageData array being centered in the height dimension
		if (numChannels == 1):
			paddedImageData = np.squeeze(paddedImageData)
			outputImage = Image.fromarray(paddedImageData.astype(np.uint8), mode='L')
		else:
			outputImage = Image.fromarray(paddedImageData.astype(np.uint8))														# Convert paddedImageData to an image object

	return outputImage


def parseNumberAndUnitsString(str):
	unitsMatches = re.findall('(nm)|(um)|(mm)|(cm)|(ms)|(us)|(ns)|(m)|(s)', str)
	unitStrings = ['nm', 'um', 'mm', 'cm', 'ms', 'us', 'ns', 'm', 's']
	unitTypes = ['spatial', 'spatial', 'spatial', 'spatial', 'time', 'time', 'time', 'spatial', 'time']
	unitsMultipliers = [1e-9, 1e-6, 1e-3, 1e-2, 1e-3, 1e-6, 1e-9, 1, 1]
	if (len(unitsMatches) > 1):
		raise Exception("Invalid number string.")
	elif (len(unitsMatches) == 0):
		multiplier = 1
		numStr = str
		unitStr = ''
		unitTypeStr = ''
	else: # len(unitsMatches) == 1
		unitStr = ''.join(list(unitsMatches[-1]))
		unitIndex = unitStrings.index(unitStr)
		multiplier = unitsMultipliers[unitIndex]
		unitTypeStr = unitTypes[unitIndex]
		numStr = str[0:-len(unitStr)]
	try:
		return float(numStr) * multiplier, unitStr, unitTypeStr
	except:
		raise Exception("Invalid number string.")


def conv(a : torch.tensor, b : torch.tensor):
	if (a.shape[-2:] != b.shape[-2:]):
		raise Exception("Mismatched tensor dimensions!  Last two dimensions must be the same size.")

	# Assumes h and x have the same size
	Nx_old = int(b.shape[-2])
	Ny_old = int(b.shape[-1])

	pad_nx = int(Nx_old / 2)
	pad_ny = int(Ny_old / 2)

	# As of 9/2/2022, cannot rely on padding in ft2(...) function
	# That function has the pad_nx and pad_ny arguments reversed in its call to torch.nn.functional.pad(...)
	# Because of that, the x dimension (height) gets y's padding amount and vice-versa.
	# Therefore, doing the padding here.
	aPadded = torch.nn.functional.pad(a, (pad_ny,pad_ny,pad_nx,pad_nx), mode='constant', value=0)
	bPadded = torch.nn.functional.pad(b, (pad_ny,pad_ny,pad_nx,pad_nx), mode='constant', value=0)
	A = ft2(aPadded, pad=False)
	B = ft2(bPadded, pad=False)

	# The normalization on the FFTs of hPadded and xPadded get multiplied together.  This throws off the scaling.
	rescaleFactor = np.sqrt(aPadded.shape[-2] * aPadded.shape[-1])

	Y = (A * B) * rescaleFactor
	y = ift2(Y)
	y = y[..., pad_nx:(pad_nx+Nx_old), pad_ny:(pad_ny+Ny_old)]

	return y


def applyFilterSpaceDomain(h : torch.tensor, x : torch.tensor):
	return conv(h, x)


def computeBandlimitingFilterSpaceDomain(f_x_max, f_y_max, Kx, Ky):
	# Should be 4D T x C x H x W, thus ift2(...) can be used.
	bandlimiting_Filter_Freq = torch.zeros_like(Kx)
	bandlimiting_Filter_Freq[ ( torch.abs(Kx) < f_x_max) & (torch.abs(Ky) < f_y_max) ] = 1
	bandlimiting_Filter_Space = ift2(bandlimiting_Filter_Freq)
	bandlimiting_Filter_Space = bandlimiting_Filter_Space[None,:,None,:,:,:] # Expand from 4D to 6D (TCHW dimensions --> BTPCHW dimensions)
	return bandlimiting_Filter_Space


def check_tensors_broadcastable(a : torch.Tensor, b : torch.Tensor):
	if not (isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor)):
		raise Exception("Error: Need tensor inputs for check_tensors_broadcastable(...).")
	lenDiff = abs(len(a.shape) - len(b.shape))
	for i in range(min(len(a.shape), len(b.shape)) - 1, -1, -1):
		aInd = i + lenDiff*(len(a.shape) > len(b.shape))
		bInd = i + lenDiff*(len(a.shape) < len(b.shape))
		if not ((a.shape[aInd] == 1) or (b.shape[bInd] == 1) or (a.shape[aInd] == b.shape[bInd])):
			return False
	return True