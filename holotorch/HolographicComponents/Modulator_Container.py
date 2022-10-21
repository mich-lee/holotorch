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
import os
import glob
import re
from pathlib import Path
import matplotlib.pyplot as plt
import shutil
import warnings

from holotorch.HolographicComponents.ValueContainer import ValueContainer
from holotorch.utils.Dimensions import TensorDimension
from holotorch.Optical_Components.CGH_Component import CGH_Component
from holotorch.HolographicComponents.SLM_Upsampler import SLM_Upsampler 
from holotorch.utils.string_processor import convert_integer_into_string
from holotorch.utils.Visualization_Helper import add_colorbar
from holotorch.utils.units import *
from holotorch.utils.Enumerators import *
import holotorch.utils.Dimensions as Dimensions
import holotorch.utils.pjji as piji

class Modulator_Container(CGH_Component):

    _save_flag = True

    def __init__(self,  
                        tensor_dimension : TensorDimension,
                        feature_size    : float,
                        n_slm_batches               = 1,
                        replicas :int               = 1,
                        pixel_fill_ratio: float     = 1.0,                  
                        pixel_fill_ratio_opt: bool  = False,  
                        init_type       : ENUM_SLM_INIT = None,
                        init_variance : float = 0,
                        flag_complex : bool         = False,
                        slm_directory : str         = ".slm",
                        slm_id : int                = 0,
                        store_on_gpu                = False,
                        ) -> None:
        """create an container of multiple slm for optimizing.

        the SLM-Input is a 5D (4D) 
        
        B x T x C x H X W
        B: batch
        T: time
        C: color
        H: image height
        W: image width

        B // batches_per_slm SLM_Models are created. Each SLM_Model is assigned its own optimizer, so
        batches_per_slm SLM images are optimized jointly  

        NOTE: batches_per_slm is assumed to be a multiple of batchdim B!!

        Args:
            slm_sz ([type]): [5D tuple of tensor dimensions]
            slm_type (ENUM_SLM_TYPE.phase_only): phase only / amplitude / complex
            init_type (ENUM_SLM_INIT): initialize method for slm (e.g. random values or zeros)
            material (CGH_Material): the material of the SLM (for wavelength dependency)
            n_batches (int, optional): How many batches of SLM objects do we need. Defaults to 1.
            device ([type], optional): Device of the tensor objects. Defaults to torch.device('cpu').
        """
        
        # create the models
        super().__init__()

        if ((tensor_dimension.batch % n_slm_batches) != 0):
            raise Exception("Error: 'n_slm_batches' does not evenly divide the number of batches.")

        if not (hasattr(self, 'static_slm')):
            raise Exception("A subclass failed to set the 'static_slm' attribute.")
        if not (hasattr(self, 'static_slm_data_path')):
            raise Exception("A subclass failed to set the 'static_slm_data_path' attribute.")

        if (self.static_slm == False):
            if (self.static_slm_data_path is not None):
                raise Exception("static_slm_data_path' is set, but static_slm is False.")
        else:
            if (self.static_slm_data_path is None):
                raise Exception("'static_slm' is set to True, but 'static_slm_data_path' is not set.")
            elif (not issubclass(type(self.static_slm_data_path), Path)):
                raise Exception("'static_slm_data_path' must be a pathlib.Path object.")
            elif not (self.static_slm_data_path.exists()):
                raise Exception("The pathlib.Path object provided for 'static_slm_data_path' does not point to a valid path.")

        self.input_arg_init_variance = init_variance
        self.input_arg_init_type = init_type
        self.input_arg_flag_complex = flag_complex

        # create the upsampler (defaults to identity operation)
        self.upsampler = SLM_Upsampler(replicas=replicas, 
                                        pixel_fill_ratio=pixel_fill_ratio,
                                        pixel_fill_ratio_opt=pixel_fill_ratio_opt)
        
                   
        # Number of independent images in the full dataset
        self.dataset_size         = tensor_dimension.batch
        self.feature_size = feature_size

        # Computes how many Images are in one batch
        # NOTE: Assume that dataset is always the same size       
        images_per_batch = self.dataset_size // n_slm_batches
    
        self.store_on_gpu = store_on_gpu

        # # Set the parameters for each Batch-SLM        
        # #batch_tensor_dimension = copy.deepcopy(tensor_dimension)
        
        batch_tensor_dimension =  Dimensions.BTCHW(
                n_batch         = images_per_batch,
                n_time          = tensor_dimension.time,
                n_channel       = tensor_dimension.channel,
                height          = tensor_dimension.height,
                width           = tensor_dimension.width
            )
        
        if isinstance(tensor_dimension, Dimensions.BTCHW_E):
            batch_tensor_dimension = Dimensions.BTCHW_E.from_BTCHW(
                    btchw =   batch_tensor_dimension,
                    extra_dim= tensor_dimension.extra
                )
        
        if store_on_gpu == True:
            init_function = self.init_on_gpu
        else:
            init_function = self.init_on_disk

        init_function(
                    batch_tensor_dimension = batch_tensor_dimension,
                    init_variance    = init_variance,
                    init_type        = init_type,
                    flag_complex     = flag_complex,
                    slm_directory            = slm_directory,
                    slm_id              = slm_id,
                    images_per_batch    = images_per_batch,
                    n_slm_batches       = n_slm_batches
            )

        if (self.static_slm):
            self.load_all_slms_from_folder(self.static_slm_data_path)
        
        
    def init_on_gpu(
                self,
                batch_tensor_dimension  : TensorDimension,
                init_variance           : float,
                init_type               : ENUM_SLM_INIT,
                flag_complex            : bool,
                slm_directory,
                slm_id,
                images_per_batch,
                n_slm_batches
        ):

        for k in range(n_slm_batches):
            tmp_values : ValueContainer = ValueContainer(
                tensor_dimension = batch_tensor_dimension,
                init_variance    = init_variance,
                init_type        = init_type,
                flag_complex     = flag_complex,
            )
            if not (self.device is None):
                tmp_values.data_tensor = tmp_values.data_tensor.to(self.device)
                tmp_values.scale = tmp_values.scale.to(self.device)
            setattr(self,"slm" + str(k),tmp_values)

        self.n_slm_batches = n_slm_batches
        self.current_batch_idx = 0


    def init_on_disk(self,
        batch_tensor_dimension  : TensorDimension,
        init_variance           : float,
        init_type               : ENUM_SLM_INIT,
        flag_complex            : bool,
        slm_directory,
        slm_id,
        images_per_batch,
        n_slm_batches
                    ):

        # Create a directory where temporary SLM data is stored.  Initializes self.tmp_dir
        self.create_tmp_directory(tmp_name=slm_directory, slm_id = slm_id)

        # Clear out files in the temporary SLM directory (that end in .pt)
        self.clear_temp_slm_dir()

        # Initializing a values container (needs to be initialized for self.set_images_per_batch(...) to work)
        self.values : ValueContainer = ValueContainer(
            tensor_dimension = batch_tensor_dimension,
            init_variance    = init_variance,
            init_type        = init_type,
            flag_complex     = flag_complex,
        )

        # Initializing field
        self.n_slm_batches = n_slm_batches

        for k in range(n_slm_batches - 1, -1, -1):
            with torch.no_grad():
                # Updating batch-related fields in values container and generating new SLM data (which is contained in self.values.data_tensor).
                self.values.set_images_per_batch(number_images_per_batch=images_per_batch)

                # NOTE that the self.values.data_tensor and self.values.scale tensors would have been computed on the CPU and will reside on the CPU at this point.
                #	(Unless something changes after 9/8/2022)

                # Saving the SLM data
                self.save_single_slm(batch_idx=k)

        # Moving tensors to a different device if necessary
        if not (self.device is None):
            self.values.to(self.device)

        # Forcing the currently loaded SLM to be idx=0
        self.current_batch_idx = 0
        self.load_single_slm(batch_idx=0)	# This line is technically not necessary because 'k' ends up as 0 in the loop 'for k in range(n_slm_batches - 1, -1, -1)'

    def move_tmp_save_folder(self,
             slm_id : int or str,
             tmp_base_dir = None,
             move_or_copy : ENUM_COPY_MOVE = ENUM_COPY_MOVE.MOVE
              ):

        if tmp_base_dir is None:
            tmp_base_dir = self.tmp_base_dir

        src = self.tmp_dir
        
        self.create_tmp_directory(tmp_name=tmp_base_dir, slm_id=slm_id)
        # # Source path 
        
        # # Destination path 
        dest = self.tmp_dir

        if src == dest:
            return
            
        shutil.rmtree(dest)	

        # Copy the content of 
        # source to destination 
        if move_or_copy is ENUM_COPY_MOVE.COPY:
            shutil.copytree(src.resolve(), dest.resolve(), dirs_exist_ok = True) 
        elif ENUM_COPY_MOVE.MOVE:
            shutil.move(str(src), str(dest)) 

    @property
    def save_slm_flag(self) -> bool:
        return self._save_flag
    
    @save_slm_flag.setter
    def save_slm_flag(self, value : bool):
        self._save_flag = value
    
    @property
    def shape(self):
        
        if self.store_on_gpu:
            value : ValueContainer = getattr(self,"slm0")
            return value.data_tensor.shape
        else:
            return self.values.data_tensor.shape


    @property
    def images_per_batch(self) -> int:
        """Returns the number images stored per batch

        Returns:
            int: _description_
        """        
        return self.values.images_per_batch
    
    @property 
    def batch_tensor_dimension(self):
        if self.store_on_gpu:
            value : ValueContainer = getattr(self,"slm0")
            return value.tensor_dimension
        else:
            return self.values.tensor_dimension

    def set_images_per_batch(self, number_images_per_batch : int, number_slm_batches = None):
        """Sets the Number of images per batch

        WARNING: This involves changing the pointer and lightning objects
        will no be longer be able to optimize for the old ones
        
        WARNING: This will delete 
        
        HENCE: Be careful when calling this
        
        Args:
            number (int): _description_
        """        
        if number_slm_batches is None:
            number_slm_batches = self.n_slm_batches
        
        if self.n_slm_batches == None:
            self.n_slm_batches = number_slm_batches
        elif number_slm_batches == self.n_slm_batches:
            if number_images_per_batch == self.images_per_batch:
                return # Nothing to do
        else:
               self.n_slm_batches = number_slm_batches 

        # First we will remove any existing stored data
        test = os.listdir(self.tmp_dir)
        for item in test:
            if item.endswith(".pt"):
                os.remove(os.path.join(self.tmp_dir, item))
        
         
        with torch.no_grad():
            # Set the new batch dimension
            self.values.set_images_per_batch(number_images_per_batch=number_images_per_batch)
            
            # Now we need to save each batch as an indivudal file
            for k in range(self.n_slm_batches):
                self.save_single_slm(batch_idx=k)
        
    
    def reset_values(self, noise_variance = 0.1):
        self.set_new_values(self.values.data_tensor * 0 + noise_variance*torch.rand_like(self.values.data_tensor))
    
    def get_param(self):
        return self.values.data_tensor
    
    def set_new_values(self,
                       new_values   : torch.Tensor,
                       scale        : torch.Tensor = None,
                       batch_idx    : int = None,
                       sub_batch_idx = None,
                       save_flag    : bool = None
                       ):
        """Loads new values into the tensor without changing the pointer

        Args:
            new_values (torch.Tensor): _description_
        """        
        
        # If not provided take the one that's saved in the SLM state
        if save_flag is None:
            save_flag = self.save_slm_flag
        
        if (new_values.ndim > 5 and not isinstance(self.values.tensor_dimension,Dimensions.BTCHW_E)):
            raise ValueContainer("SLM cannot have than more than 5 dimensions")
        
        # If we change the values of a different batch we need to save the current one
        if batch_idx != self.current_batch_idx and batch_idx is not None:
            if save_flag == True:
                self.save_single_slm()
            self.current_batch_idx = batch_idx
        
        if scale is None:
            scale = torch.ones([1])*1.0
        
        # NOTE: If we only want to change a sub-batch we need to slightly different funnction calls
        with torch.no_grad():
            if sub_batch_idx is None:
                
                tmp = self.values.data_tensor
                tmp_scale = self.values.scale
                scale = scale.expand(tmp_scale.shape)
                
                new_values = new_values.view(tmp.shape)
                
                self.values.data_tensor.copy_(new_values)
                self.values.scale.copy_(scale)
                
            else:
                tmp = self.values.data_tensor[sub_batch_idx]
                tmp_scale = self.values.scale[sub_batch_idx]
                scale = scale.expand(tmp_scale.shape)
                
                new_values = new_values.view(tmp.shape)
                self.values.data_tensor[sub_batch_idx].copy_(new_values)
                self.values.scale[sub_batch_idx].copy_(scale)
                
            
    def create_tmp_directory(self, tmp_name : str, slm_id : int or str):
        if tmp_name == None:
            raise ValueError
        # set a place to save the temporary files
        tmp_dir = Path(tmp_name)
        self.tmp_base_dir = tmp_dir

        if isinstance(slm_id, int):
            self.tmp_dir = tmp_dir / convert_integer_into_string(slm_id, depth=2)
        else:
            self.tmp_dir = tmp_dir / slm_id

        self.tmp_dir.mkdir(parents=True, exist_ok=True)   

    @staticmethod
    def create_filename(batch_idx, folder):
        # create the filename
        if batch_idx is None:
            print("This should never be none")
            return
        slmmodel_filename = Path("SLM_" + str(batch_idx).zfill(4) + ".pt")
        slm_path = Path(folder) / slmmodel_filename
        return slm_path
    
    

    def _create_file_path(self,
            folder : str = None,
            filename : str = None,
            batch_idx :int  = 0
                        ):
        if filename is None:
            if folder is None:
                folder = self.tmp_dir
            slm_path = Modulator_Container.create_filename(batch_idx = batch_idx, folder = folder)
        else:
            slm_path = Path(folder) / Path(filename)
            
        slm_path.parents[0].mkdir(exist_ok=True)
        
        return slm_path
    

    
    def save_single_slm(self,
                        batch_idx :int  = None,
                        folder : str = None,
                        filename : str = None
                        ):
        if (hasattr(self, 'current_batch_idx')):
            if batch_idx == None:
                batch_idx = self.current_batch_idx
            if ((batch_idx != self.current_batch_idx) and (not self.store_on_gpu)):
                self.load_single_slm(batch_idx=batch_idx)
        # else:
        # 	# Reaching this point means class is still initializing
        # 	pass

        slm_path = self._create_file_path(batch_idx = batch_idx, folder = folder, filename=filename)

        if (self.store_on_gpu):
            tempValues = self.load_single_slm(batch_idx=batch_idx)
            state_dict = tempValues.state_dict()
        else:
            # If 'current_batch_idx' was not set, then the class is still initializing.  This means that whatever is stored in self.values are the initialized values, which means that they should just be saved.
            # If 'current_batch_idx' was set, then either the correct values were already loaded before this method was called, or the correct values would have been loaded earlier in this method.
            # Therefore, one can just have this single line.
            state_dict = self.values.state_dict()

        # If not static_slm, save.  If folder is not none, then assume that the user is explicitly trying to save an SLM, rather than the code trying to save a temporary SLM file.
        if (not self.static_slm) or (folder is not None):
            torch.save(state_dict, slm_path)


    def load_single_slm(	self,
                            batch_idx,
                            folder : str = None,
                            filename : str = None,
                            flag_save_current = True,
                        ):
        if ((filename is not None) or (folder is not None)):
            raise Exception("Use case not covered.")
        else:
            if self.store_on_gpu:
                load_function = self.load_single_slm_gpu		# Nothing different needs to be done
            elif not self.static_slm:
                load_function = self.load_single_slm_disk
            else:
                folder = self.static_slm_data_path.resolve()		# The call to resolve() is technically not needed.  Even though self.static_slm_data_path should be a pathlib.Path object and not a string, the functions that argument gets passed to should be able to handle a pathlib.Path object.
                load_function = self.load_single_slm_disk_static

            slm = load_function(
                                    batch_idx = batch_idx,
                                    folder=folder,
                                    filename=filename,
                                    flag_save_current = flag_save_current
                                )


            return slm.to(self.device)


    def load_single_slm_gpu(	self,
                        batch_idx :int = 0,
                        folder : str = None,
                        filename : str = None,
                        flag_save_current = True,
                         ):
    
        slm = getattr(self, "slm" + str(batch_idx))
        return slm
    

     # Loads an SLM without saving anything to disk
    def load_single_slm_disk_static(self,
                                    batch_idx,
                                    folder : str = None,
                                    filename : str = None,
                                    flag_save_current = False	# This argument does nothing in this function and only serves to make it compatible with other function calls
                                    ):
        if batch_idx == self.current_batch_idx:
            # We don't need to do anything if the current batch is already loaded
            return self.values

        self.current_batch_idx = batch_idx
        slm_path = self._create_file_path(batch_idx = batch_idx, folder = folder, filename=filename)

        # there is only one slm object
        new_state_dict = torch.load(slm_path)
        self.values.load_state_dict(new_state_dict)

        return self.values

    
    def load_single_slm_disk(self,
                        batch_idx :int = 0,
                        folder : str = None,
                        filename : str = None,
                        flag_save_current = True,
                         ):
        
        if batch_idx == self.current_batch_idx and flag_save_current == True:
            # We don't need to do anything if the current batch is already loaded
            return self.values
        
        if flag_save_current:
            self.save_single_slm()
        
        self.current_batch_idx = batch_idx

        slm_path = self._create_file_path(batch_idx = batch_idx, folder = folder, filename=filename)

        # there is only one slm object  
        new_state_dict = torch.load(slm_path)
        self.values.load_state_dict(new_state_dict)

        if flag_save_current == False:
            self.save_single_slm()

        return self.values

    def save_all_slms_into_folder(self, folder):
        """
        save slms to a specific folder

        Args:
            folder: the folder where all SLM state_dicts are stored
        
        """
        self.clear_save_dir(folder)
        os.makedirs(folder,exist_ok=True)

        for k in range(self.n_slm_batches):
            slmmodel_filename = "SLM_" + str(k).zfill(4) + ".pt"
            slm_path = folder / slmmodel_filename  
            values = self.load_single_slm(batch_idx=k)
            state_dict = values.state_dict()
            torch.save(state_dict, slm_path)


    def load_all_slms_from_folder(self, folder):
        """
        load slm state_dicts from a specific folder

        Args:
            slmmodel_folder: the folder where all SLM state_dicts are stored
        """

        # Clear out files in the temporary SLM directory (that end in .pt)
        self.clear_temp_slm_dir()

        files = glob.glob(str(folder)+"\\*.pt")
        
        n_batches = len(files)
        self.n_slm_batches = n_batches

        batch_size = -1
        n_batches = -1

        for k in range(len(files)):
            file = Path(files[k])
            filename = str(file.stem)
            # Finds the index
            idx_slm = [int(x) for x in re.findall('\d+', filename)][0]
            slm_state_dict = torch.load(file, map_location=self.device)

            self.current_batch_idx = idx_slm
            slmName = "slm" + str(idx_slm)
            
            newContainerDimShape = slm_state_dict['_data_tensor'].shape
            if (len(newContainerDimShape) == 5):
                newContainerDim = Dimensions.BTCHW(
                    n_batch     = newContainerDimShape[0], # Total number of images for Modualator_Container
                    n_time      = newContainerDimShape[1],
                    n_channel   = newContainerDimShape[2],
                    height      = newContainerDimShape[3],
                    width       = newContainerDimShape[4]
                )
            else:
                raise Exception("SLM data tensors should have dimensions BTCHW.")

            if k == 0:
                batch_size = slm_state_dict['_data_tensor'].shape[0]

                with torch.no_grad():
                    if not (self.store_on_gpu):
                        self.values : ValueContainer = ValueContainer(	tensor_dimension = newContainerDim,
                                                                        init_variance    = self.input_arg_init_variance,
                                                                        init_type        = self.input_arg_init_type,
                                                                        flag_complex     = self.input_arg_flag_complex,
                                                                    )

            if not (self.store_on_gpu):
                self.values.load_state_dict(slm_state_dict)
                if not (self.static_slm):
                    self.save_single_slm(batch_idx=idx_slm)
            else:
                tmp_values : ValueContainer = ValueContainer(
                    tensor_dimension = newContainerDim,
                    init_variance    = self.input_arg_init_variance,
                    init_type        = self.input_arg_init_type,
                    flag_complex     = self.input_arg_flag_complex,
                )
                setattr(self, slmName, tmp_values)

                with torch.no_grad():
                    getattr(self, slmName).set_images_per_batch(number_images_per_batch=batch_size)
                getattr(self, slmName).load_state_dict(slm_state_dict)

                # Moving data to proper device if necessary
                if not (self.device is None):
                    getattr(self, slmName).data_tensor = getattr(self, slmName).data_tensor.to(self.device)
                    getattr(self, slmName).scale = getattr(self, slmName).scale.to(self.device)


    def clear_temp_slm_dir(self):
        # Removes temporary stored data
        if not (hasattr(self, 'tmp_dir')):
            warnings.warn("Tried to clear the temporary SLM data directory, but the corresponding field 'tmp_dir' was uninitialized.")
            return
        elif (self.tmp_dir is None):
            warnings.warn("Tried to clear the temporary SLM data directory, but the corresponding field 'tmp_dir' was set to None.")
            return

        test = os.listdir(self.tmp_dir)
        for item in test:
            if item.endswith(".pt"):
                os.remove(os.path.join(self.tmp_dir, item))


    def clear_save_dir(self, folder):
        # Removes saved data
        if (Path(folder).exists()):
            test = os.listdir(folder)
            for item in test:
                if item.endswith(".pt"):
                    os.remove(os.path.join(folder, item))


    def __str__(self, ):
        """
        Creates an output for the SLM Container class.
        """
        mystr = super().__str__()
        mystr += "\n-------------------------------------------------------------\n"
        mystr += "number batches: " + str(self.n_slm_batches) + "\n"
        mystr += "Shape: " + str(self.shape) + "\n"


        return mystr
    
    def __repr__(self):
        return self.__str__()

    @staticmethod        
    def quantized_voltage(voltage_map,
                          bit_depth=8,
                          ):
        """
        just quantize voltages into 8bit and return a tensor with the same dtype
        :param phasemap:
        :return:
        """


        # Convert into integer, round, and divide by max digital number (e.g. 255 for bit_depth=8) 
        # range is still [0, 1]
        voltage_map = torch.round((2.**bit_depth-1) * voltage_map) / (2.**bit_depth-1)

        return voltage_map    
    

    def forward(self,
                batch_idx       : int               = None,
                bit_depth       : int or None       = None,
                flag_quantized = False,
                ) -> torch.Tensor:
        """ Applies the SLM to the input field given the SLM index

        Args:
            input ([type]): [description]
            batch_idx (int, optional): [description]. Defaults to 0.

        Returns:
            [type]: [description]
        """        
        if batch_idx is None:
            batch_idx = self.current_batch_idx

        value_container = self.load_single_slm(batch_idx=batch_idx)

        values = value_container.data_tensor
        
        if flag_quantized:
            values = Modulator_Container.quantized_voltage(voltage_map=values)

        return values, value_container.scale

    def show_piji(self, batch_idx : int = 0):
        
        self.load_single_slm(batch_idx=batch_idx)
        piji.show(self.values.data_tensor.detach().squeeze().cpu(), title="SLM_Pattern" + str(batch_idx))
    
    def visualize_slm(self, figsize=(10,10),
                      batch_idx = 0,
                      sub_batch_idx = 0,
                    x0 = None,
            y0 = None,
            width = None,
            height = None,
            vmax = None,
            vmin = None,
            title1 = "",
            wavelengths = None,
            ):
        
        self.load_single_slm(batch_idx=batch_idx)
        
        plt.figure(figsize=figsize)
        if height == None:
            x1 = None
        else:
            x1 = x0 + height

        if width == None:
            y1 = None
        else:
            y1 = y0 + width
        
        img1 = self.values.data_tensor[sub_batch_idx,:,:,x0:x1,y0:y1].squeeze()
        
        if torch.is_tensor(img1):
            img1 = img1.detach().cpu()
            
        if img1.ndim == 2:

            _im = plt.imshow(img1, vmax = vmax, vmin=vmin, cmap = 'gray')
            add_colorbar(_im)
            plt.title(title1)
            
        elif img1.ndim == 3:
            
            for k in range(img1.shape[0]):
                plt.subplot(1,img1.shape[0],k+1)
                _im = plt.imshow(img1[k], vmax = vmax, vmin=vmin, cmap = 'gray')
                add_colorbar(_im)
                tmp_title = title1
                if wavelengths is not None:
                    tmp_title += str(int(wavelengths.data_tensor.squeeze()[k]/nm)) + "nm"
                plt.title(tmp_title)
            plt.tight_layout()
            