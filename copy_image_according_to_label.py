import os 
import shutil 
import xml.etree.ElementTree as ET 


def copy_image(path_label_in, path_label_out, path_image_in, path_image_out):
    list_label = os.listdir(path_label_in) 
    for label in list_label:
        tree = ET.parse(path_label_in + label)
        objects = tree.findall('object')
        if len(objects):
            shutil.copy(path_image_in + label[:-3] + 'jpg', path_image_out)
            shutil.copy(path_label_in + label, path_label_out)



if __name__ == "__main__":
    ###
    name_in = 'ji_dan_enhanced'
    name_out = 'ji_dan998'
    ###
    path_label_in = 'D:/data/' + name_in + '/data/Annotations/' 
    path_image_in = 'D:/data/' + name_in + '/data/images/' 
    path_label_out = 'D:/data/' + name_out + '/data/Annotations/'
    path_image_out = 'D:/data/' + name_out + '/data/images/'  

    copy_image(path_label_in, path_label_out, path_image_in, path_image_out)