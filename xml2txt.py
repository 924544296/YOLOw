import xml.etree.ElementTree as ET 
import os 
##################################


def xml2txt(path_in, path_out, name2class):
    dirs = os.listdir(path_in)
    for dir_ in dirs:
        tree = ET.parse(path_in + dir_)
        W = float(tree.find('size').find('width').text)
        H = float(tree.find('size').find('height').text)
        objects = tree.findall('object')
        name_file = dir_.split('.')[0]
        with open(path_out + name_file + '.txt', 'w') as f:  
            for object_ in objects:
                name_class = object_.find('name').text
                name_class = name2class[name_class]
                x1 = float(object_.find('bndbox').find('xmin').text) / W
                y1 = float(object_.find('bndbox').find('ymin').text) / H
                x2 = float(object_.find('bndbox').find('xmax').text) / W
                y2 = float(object_.find('bndbox').find('ymax').text) / H
                w = x2 - x1
                h = y2 - y1
                xc = x1 + w / 2
                yc = y1 + h / 2
                f.write(str(name_class) + ' ' + str(xc) + ' ' + str(yc) + ' ' + str(w) + ' ' + str(h) + '\r')
        
  
########################################   
if __name__ == "__main__":
    ###
    name = 'ji_dan998'
    ###
    path_in = 'D:/data/' + name + '/data/Annotations/'
    path_out = 'D:/data/' + name + '/data/labels/'
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    # name2class = {'毛发': 0, '色点': 1, '色块': 2, '乳胶颗粒': 3, '纤维': 4, '不锈钢金属屑': 5}
    # name2class = {'ship': 0}
    name2class = {'foreign': 0}
    xml2txt(path_in=path_in, path_out=path_out, name2class=name2class)

