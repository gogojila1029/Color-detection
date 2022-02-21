import warnings
import string
import random
import numpy as np
import keras
import keras.preprocessing.image
import glob
import argparse
import tensorflow as tf 
from PIL import Image
from tqdm.auto import tqdm
from struct import unpack
import argparse
import os
import shutil
import time

def arg_parse():
  """
  parse the argument.
  """
  parser=argparse.ArgumentParser(description="any image parsing")
  parser.add_argument("--c_address", help="the address of the directory the images you want to augument are located in", type=str)
  parser.add_argument("--m_address", help="the address of a directory where you want to save the augumented images", type=str)
  parser.add_argument("--s_address", help="the source of the data directory", type=str)
  parser.add_argument("--b_size", help="batch size of the images", type=int, default=1)
  parser.add_argument("--img_size", help="size of the image", type=str, default="416, 416")
  parser.add_argument("--cls", type=str)
  #parser.print_help()
  args=parser.parse_args()
  return args

marker_mapping = {
    0xffd8: "Start of Image",
    0xffe0: "Application Default Header",
    0xffdb: "Quantization Table",
    0xffc0: "Start of Frame",
    0xffc4: "Define Huffman Table",
    0xffda: "Start of Scan",
    0xffd9: "End of Image"
}

class JPEG:
    def __init__(self, image_file):
        with open(image_file, 'rb') as f:
            self.img_data = f.read()
    
    def decode(self):
        data = self.img_data
        while(True):
            marker, = unpack(">H", data[0:2])
            # print(marker_mapping.get(marker))
            if marker == 0xffd8:
                data = data[2:]
            elif marker == 0xffd9:
                return
            elif marker == 0xffda:
                data = data[-2:]
            else:
                lenchunk, = unpack(">H", data[2:4])
                data = data[2+lenchunk:]            
            if len(data)==0:
                break    
                  
class agumentation:
  classes={}
  def __init__(self, 
               current_address="", 
               moving_address="", 
               batch_size=1,
               v_split=None,
               labels="inferred",
               image_size=(416,416),
               shuffle=True, 
               seed=None,
               crop_size=None,
               max_delta=None):
    self.current_address=current_address
    self.moving_address=moving_address
    self.batch_size=batch_size
    self.v_split=v_split,
    self.labels=labels
    self.image_size=image_size
    self.shuffle=shuffle
    self.seed=seed
    self.crop_size=crop_size
    self.max_delta=max_delta
    
  #from_dir function has the association of ImageDataGenerator class  
  def aug(self, image):
    
    image=(image/255.0)
    if type(self.crop_size) is list:
      image= tf.image.resize_with_crop_or_pad(image, self.crop_size)
      
    if type(self.max_delta) is float:
      image = tf.image.random_brightness(image, self.max_delta)
      
    return image

  def from_dir(self):
    a=tf.keras.utils.image_dataset_from_directory(self.current_address,
                                                  self.labels,
                                                  batch_size=self.batch_size,
                                                  image_size=self.image_size,
                                                  shuffle=self.shuffle,
                                                  seed=self.seed,
                                                  validation_split=None)
    
    return a

  #iter function is to iterate from_dir function

  def itering(self):
    a=agumentation(self.current_address, 
                      self.moving_address, batch_size=self.batch_size)
    data=a.from_dir()
    iter_n=0
    add=0
    b_size=tf.constant(self.batch_size)
  
    #for i in os.listdir(self.current_address):
      #for _, _, files in os.walk(os.path.join(self.current_address, i)):
        #iter_n=iter_n+len(files)

    try:
          #prog_bar=tqdm(desc="Image preprocessing")
          data=iter(data)
          while True:
            data_itr=next(data, (1.1, 1.2))
            if data_itr is (1.1, 1.2):
              break

            data_itr=tf.data.Dataset.from_tensor_slices(data_itr)
            data_itr=data_itr.map(lambda x, y: (a.aug(x), y))
            data_final=list(data_itr.as_numpy_iterator())  

            #print(data_final)
            for i in tqdm(range(self.batch_size), desc=f"Batch of the Image processing{add}"):
              img=data_final[:][i][0]
              label=data_final[:][i][1]
              img=tf.keras.utils.array_to_img(img)
              for k_value in agumentation.classes.items():
                if label==k_value[1]:
                  generator=a.id_generator()
                  tf.keras.utils.save_img(os.path.join(self.moving_address, k_value[0]+"/"+k_value[0]+"_"+generator+".jpg"), img)
              #time.sleep(0.001)    
            add+=1      
            #prog_bar.update(add)
    except StopIteration:
      pass

  def id_generator(self, size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

class deal_with_file:
  classes=[]
  ext=('jpg', 'jpeg', 'bmp', 'png', 'ppm', 'tif','tiff')
  
  def __init__(self, current_address, moving_address):
    self.c_addr=current_address
    self.m_addr=moving_address

  def c_dir_create(self):
    try:
      for i in deal_with_file.classes:
        if os.path.isfile(os.path.join(self.c_addr, i))==True:
          continue
        else:
          dir=os.makedirs(os.path.join(self.c_addr, i))
          auged_dir=os.makedirs(os.path.join(self.m_addr, i))

    except FileExistsError:
      pass

  def rm_dir(self, addr):
    if type(addr) is list:
      for name in addr:
        shutil.rmtree(name)

    elif type(addr) is str:
      shutil.rmtree(addr)

  def rm_file(self, addr):
    #try:
    ext=deal_with_file.ext
    ext = tuple('.%s' % e for e in ((ext,) if isinstance(ext, str) else ext))
    if type(addr) is list:
      for name in addr:
        list_f=[os.path.join(root, f) for root, _, files in os.walk(name) for f in files if f.lower().endswith(ext)]
        for j in list_f:
          os.remove(j)

    elif type(addr) is str:
      list_f=[os.path.join(root, f) for root, _, files in os.walk(addr) for f in files if f.lower().endswith(ext)]
      for j in list_f:
        os.remove(j)  

    #except OSError:
      #print(f"All the JPG files in {addr} are removed!")
      #pass

  def isEmpty(self, addr):
    if os.path.exists(addr) and not os.path.isfile(addr):

        # Checking if the directory is empty or not
        if not os.listdir(addr):
          a=deal_with_file(self.c_addr, self.m_addr)
          a.rm_dir(addr)            
        else:
            print("Not empty directory")
            pass
    else:
        print("The path is either for a file or not valid")

  def tf_files(self, arg):
    try:
      ext=deal_with_file.ext
      for ft in tqdm(os.listdir(arg)):
        if ft.endswith(ext):
            for i in deal_with_file.classes:
              if os.path.basename(arg)==i or os.path.basename(arg)=="./":
                wrong_img=JPEG(os.path.join(arg, ft))
                try:
                  wrong_img=wrong_img.decode()
                  shutil.move(os.path.join(arg, ft), os.path.join(self.c_addr, i))
                except:
                  print(f"problem at {ft} file")
                  pass
        #a=deal_with_file(self.c_addr, self.m_addr)
        #a.isEmpty(arg)
    except OSError:
      print("process is done")        


if __name__ == '__main__':  
  args=arg_parse()
  c_address=str(args.c_address)
  m_address=str(args.m_address)
  s_address=str(args.s_address)
  b_size=int(args.b_size)
  clss=str(args.cls)
  img_size=str(args.img_size)

  clss_dic={}
  clss_list=clss.casefold().split(", ")
  clss_list.sort()  
  clss_list=tuple(clss_list)
  
  for i, j in enumerate(clss_list): 
    clss_dic[j]=i 
    deal_with_file.classes.append(j)
  agumentation.classes=clss_dic  
  a=deal_with_file(c_address, m_address) 
  a.c_dir_create() 
  #print(clss_list)
  
  for i in clss_dic.keys():
    a.tf_files(os.path.join(s_address, i))
    print(os.path.join(s_address, i))

  img_size=img_size.split(", ")
  for i in img_size:
    img_size.append(int(i))
    img_size.pop(0) 
  img_size=tuple(img_size)   
  b=agumentation(current_address=c_address, moving_address=m_address, shuffle=True, seed=1, batch_size=b_size, image_size=img_size)
  b.itering()  