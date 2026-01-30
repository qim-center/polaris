import os
import numpy as np
from glob import glob
from tqdm.notebook import tqdm
import tifffile
import json
from cil.framework import AcquisitionGeometry, AcquisitionData

class PolarisDataReader(object):
    
    def __init__(self, folder_name=None, roi=None):
        """
        folder_name: Should be the parent folder where the folders
        num_proj_subset: number of projections to use starting from the first projection.
            TODO - Extend this to a slice object.
        """
        self.folder_name = folder_name
        self._roi = roi
        
        if folder_name is not None:
            self.set_up(folder_name=folder_name)
        

    def set_up(self, folder_name=None):
        with open(os.path.join(folder_name,'/02-tomo/scan_information.json'), 'r') as f:
          scan_information = json.load(f)
        with open(os.path.join(folder_name,'/02-tomo/Input/command.json'), 'r') as f:
          tomo_command = json.load(f)
        with open(os.path.join(folder_name,'/01-ff/Input/command.json'), 'r') as f:
          flat_command = json.load(f)
        
        # Parsing of parameters from file
        voxel_size_mm = scan_information['pixel_size_um']/1000
        self.propagation_distance_mm = scan_information['propagation_distance_mm']
        source_detector_mm = tomo_command['stage_position_mm']['camera_beam']
        source_object_mm = tomo_command['stage_position_mm']['object_beam'] 
        roi = tomo_command['camera']['roi_px']
        num_pixels = [roi['bot']-roi['top']+1, roi['right']-roi['left']+1]
        self.total_angle_deg = tomo_command['acquisition']['total_angle_deg']
        self.step_size_deg = tomo_command['acquisition']['step_size_deg']
        
        self.roi_flat = roi
        # The Detektor pixel size is not saved in the metadata.
        if tomo_command['camera']['name'] == 'camera-photonicscience-gsense4040xl-221094':
            px_size = 0.016
        else:
            print('Camera Name is unknown')
            px_size = 1

        # Tomo Files
        self.projections_path = os.path.join(folder_name,"02-tomo/Output/Binaries")
        # Flat Files
        self.flats_path = os.path.join(folder_name,"01-ff/Output/Binaries")

        # Update angles and num_pixels based on roi
        angles = np.arange(0,self.total_angle_deg+self.step_size_deg,self.step_size_deg)
        if self._roi['angle'] != -1:
            slicing = self._roi['angle']
            angles = angles[slicing[0]:slicing[1]:slicing[2]]
        if self._roi['vertical'] != -1:
            start = self._roi['vertical'][0] if self._roi['vertical'][0] != None else 0
            end = self._roi['vertical'][1] if self._roi['vertical'][1] != None else num_pixels[0]
            step = self._roi['vertical'][2]
            # Update num_pixel in the vertical direction
            num_pixels[0] = (end - start)//step
        if self._roi['horizontal'] != -1:
            start = self._roi['horizontal'][0] if self._roi['horizontal'][0] != None else 0
            end = self._roi['horizontal'][1] if self._roi['horizontal'][1] != None else num_pixels[1]
            step = self._roi['horizontal'][2]
            # Update num_pixel in the horizontal direction
            num_pixels[1] = (end - start)//step

        # Translating roi dict for TiffStackReader
        self.tiff_roi = {'axis_0':self._roi['angle'],
                   'axis_1':self._roi['vertical'],
                   'axis_2':self._roi['horizontal']}
        
        self._ag = AcquisitionGeometry.create_Cone3D(
            source_position=[0,-source_object_mm,0],
            detector_position=[0,source_detector_mm-source_object_mm,0],
            units = 'mm'
        ).set_panel(
            num_pixels=[num_pixels[1],num_pixels[0]], pixel_size=(px_size, px_size)
        ).set_angles(angles=angles)

        self._ig = self._ag.get_ImageGeometry()
        self._ig.voxel_size_x = voxel_size_mm
        self._ig.voxel_size_y = voxel_size_mm
        
        self.num_pixels = num_pixels
        self.angles = angles

    def get_geometry(self):
        """
        Return AcquisitionGeometry object.
        """
        return self._ag
        
    def get_image_geometry(self):
        """
        Return ImageGeometry object.
        """
        return self._ig
    
    def read(self):
        """
        Reads projections and return AcquisitionData container
        """
        
        files_tomo = glob(f"tomo*.tif",root_dir=self.projections_path)
        files_tomo = [file for file in files_tomo if "_" not in file]
        files_tomo.sort()
        files_tomo = np.array(files_tomo)
        
        angle_slicing = self._roi['angle']
        if angle_slicing != -1:
            files_tomo = files_tomo[angle_slicing[0]:angle_slicing[1]:angle_slicing[2]]

        vertical_slicing = self._roi['vertical']
        horizontal_slicing = self._roi['horizontal']
        # Preallocate numpy array
        self._arr = np.empty((len(files_tomo),*self.num_pixels))
        for i, file in enumerate(tqdm(files_tomo, desc="Importing Projections", unit="images")):
            img = tifffile.imread(os.path.join(self.projections_path,file))
            if vertical_slicing != -1 and horizontal_slicing != -1:
                img = img[vertical_slicing[0]:vertical_slicing[1]:vertical_slicing[2],
                    horizontal_slicing[0]:horizontal_slicing[1]:horizontal_slicing[2]]
            elif vertical_slicing != -1:
                img = img[vertical_slicing[0]:vertical_slicing[1]:vertical_slicing[2],:]
            elif horizontal_slicing != -1:
                img = img[:,horizontal_slicing[0]:horizontal_slicing[1]:horizontal_slicing[2]]
            self._arr[i] = img

        files_flat = glob(f"ff*.tif",root_dir=self.flats_path)
        files_flat.sort()
        self._flats = np.empty((len(files_flat),*self.num_pixels))
        for i,file in enumerate(tqdm(files_flat,desc="Importing Flat fields",unit="images")):
            img = tifffile.imread(os.path.join(self.flats_path,file))[self.roi_flat['top']:self.roi_flat['bot']+1,self.roi_flat['left']:self.roi_flat['right']+1]
            if vertical_slicing != -1 and horizontal_slicing != -1:
                img = img[vertical_slicing[0]:vertical_slicing[1]:vertical_slicing[2],
                    horizontal_slicing[0]:horizontal_slicing[1]:horizontal_slicing[2]]
            elif vertical_slicing != -1:
                img = img[vertical_slicing[0]:vertical_slicing[1]:vertical_slicing[2],:]
            elif horizontal_slicing != -1:
                img = img[:,horizontal_slicing[0]:horizontal_slicing[1]:horizontal_slicing[2]]
            self._flats[i] = img
        
        corrected = self._arr / np.average(self._flats, axis = 0)
        self._ad = AcquisitionData(array=corrected, geometry=self._ag)
        return self._ad