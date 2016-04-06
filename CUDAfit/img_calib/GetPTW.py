# -*- coding: utf8 -*-
"""
    GetPTW : this program is python module that allows the CEDIP (FLIR ATS) cameras images
    and physicale data associated tyo the image
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2 of the License.
    
    Copyright (C) 2012 Jean-François Witz

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
"""

from numpy import dtype , frombuffer,reshape 
import struct

class GetPTW:
  """
  GetPTW is a simple class that allows the loading of Infra Red Files of Jade and Titanium cameras.
  It loads significant parameters such as the different temperature of the camera (housing, sensor), 
  physical info on the images (pixel pitch & size, max and min values of the film)
  This class has been developped using the DL002U Altair Reference guide pdf document.
  Members: 
      __init__
      get_frame
      close_file
  Attributes:
      Global: 
	ambiant_temperature: the ambiant temperature measured bu camera
	aperture: the aperture of the camera (fixed for a given camera)
	bit_resolution: the dynamic of the camera
	external_trigger: boolean that tell if the camera is trigged
	focal_length: ??? how does it know the lens mounted on ???
	frame_rate: step between two images
	horizontal_flip: boolean that tell if there is a horizontal flip on the image
	initial_internal_housing_temperature: global temperature of the camera housing
	intergration_time: intergration time of the camera (! is drived by trigger time on the jade)
	max_lambda: cut off wavelength
	max_lut: max DL in film
	min_lambda: cun on wavelength
	max_lut: min DL in film
	number_of_cols: number of columns 
	number_of_frames: number of frames
	number_of_rows: number of rows
	pixel_pitch: space between two pixels
	pixel_size: physical size of pixel in µm
	vertical_flip: boolean that tell if there is a horizontal flip on the image
	wheel_index: index of the filter wheel 
    Frame: 
	frame_data: value of the pixel returned as a numpy array
	frame_housing_temperature: housing temperature for a frame
	frame_sensor_temperature: frame sensor temperature for a frame
	frame_time_IRIG: Frame time using IRIG norm, if 0.0 then not available for this camera
	frame_timestamp: timestamp of a frame in µs if 0.0 then not available for this camera
  """
  def __init__(self,input_file):
    """ initialisation method: take the file path as only argument and return Global attributes (see GetPTW.__doc__ for attributes)"""
    self._openfile = open(input_file,'rb') # Open file
    self._openfile.seek(11,0) # skip the first 11 bits
    self._main_header_size = struct.unpack('i', self._openfile.read(4))[0] # get the size of the main hearder
    self._frame_header_size = struct.unpack('i', self._openfile.read(4))[0] # get the size of the frame hearder
    self._openfile.seek(27, 0);
    self.number_of_frames=struct.unpack('i', self._openfile.read(4))[0] 
    self._openfile.seek(145, 0);
    self.ambiant_temperature=struct.unpack('f', self._openfile.read(4))[0]
    self._openfile.seek(188, 0);
    self.min_lambda=struct.unpack('f', self._openfile.read(4))[0]
    self.max_lambda=struct.unpack('f', self._openfile.read(4))[0]
    self.pixel_size=struct.unpack('f', self._openfile.read(4))[0]
    self.pixel_pitch=struct.unpack('f', self._openfile.read(4))[0]
    self.aperture=struct.unpack('f', self._openfile.read(4))[0]
    self.focal_length=struct.unpack('f', self._openfile.read(4))[0]
    self.initial_internal_housing_temperature=struct.unpack('f', self._openfile.read(4))[0]
    #self.internal_housing_temperature_2=struct.unpack('f', self._openfile.read(4))[0] # null in our case !
    self._openfile.seek(245, 0);
    self.min_lut=struct.unpack('h', self._openfile.read(2))[0] # get the
    self.max_lut=struct.unpack('h', self._openfile.read(2))[0]
    self._openfile.seek(377, 0);
    self.number_of_cols=struct.unpack('H', self._openfile.read(2))[0]
    self.number_of_rows=struct.unpack('H', self._openfile.read(2))[0]
    self.bit_resolution=struct.unpack('H', self._openfile.read(2))[0]
    self._openfile.seek(397, 0); 
    self.external_trigger=struct.unpack('b', self._openfile.read(1))[0]
    self._openfile.seek(403, 0)
    self.frame_rate=struct.unpack('f', self._openfile.read(4))[0]
    self.intergration_time=struct.unpack('f', self._openfile.read(4))[0]
    self._frame_size=self._frame_header_size+self.number_of_cols*self.number_of_rows*2
    self._openfile.seek(2408,0)
    self.horizontal_flip=struct.unpack('b', self._openfile.read(1))[0]
    self._openfile.seek(2409,0)    
    self.vertical_flip=struct.unpack('b', self._openfile.read(1))[0]    
    self._openfile.seek(2414,0)    
    self.wheel_index=struct.unpack('b', self._openfile.read(1))[0]
  
  def get_frame(self,index_of_frame):
    """ frame grabber: take the index of the image as only argument and return Frame attributes (see GetPTW.__doc__ for attributes)"""
    current_frame_index=index_of_frame*self._frame_size+self._main_header_size
    self._openfile.seek(current_frame_index+80,0)
    # Jade don't have timestamp so this method can be useful
    minute=struct.unpack('b', self._openfile.read(1))[0] 
    hour=struct.unpack('b', self._openfile.read(1))[0]
    cent=struct.unpack('b', self._openfile.read(1))[0]
    second=struct.unpack('b', self._openfile.read(1))[0]
    self._openfile.seek(current_frame_index+160,0)
    milli=struct.unpack('b', self._openfile.read(1))[0]
    micro=struct.unpack('H', self._openfile.read(2))[0]
    self.frame_time=3600.*hour+60.*minute+second+0.01*cent+0.001*milli+1e-6*micro
    self._openfile.seek(current_frame_index+228,0)
    self.frame_sensor_temperature=struct.unpack('f', self._openfile.read(4))[0]
    self.frame_housing_temperature=struct.unpack('f', self._openfile.read(4))[0]
    self._openfile.seek(current_frame_index+301,0)
    self.frame_timestamp=struct.unpack('q', self._openfile.read(8))[0]
    self._openfile.seek(current_frame_index+312,0)
    IRIG_hour=struct.unpack('b', self._openfile.read(1))[0]
    IRIG_minute=struct.unpack('b', self._openfile.read(1))[0]
    IRIG_second=struct.unpack('i', self._openfile.read(4))[0]
    IRIG_micro_second=struct.unpack('b', self._openfile.read(1))[0]
    self.frame_time_IRIG=3600.*IRIG_hour+60.*IRIG_minute+IRIG_second+1e-6*IRIG_micro_second
    frame_data_index=index_of_frame*self._frame_size+self._main_header_size + self._frame_header_size
    self._openfile.seek(frame_data_index,0)
    image_buffer=self._openfile.read(self.number_of_cols*self.number_of_rows*2)
    self.frame_data=frombuffer(image_buffer, dtype='H').reshape(self.number_of_rows,self.number_of_cols)

  def close_file(self):
    """ Close file: special member that allows one to close the open file in case of multiple read files """
    self._openfile.close()
    del self._openfile
  

