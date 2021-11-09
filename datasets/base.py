import os
import math
import torch
from PIL import Image
import numpy as np
from xml.etree import cElementTree as ElementTree

class Set(torch.utils.data.Dataset):
    def __init__(self, root, path, transform = None,
                 label_map = {"o1": 0, "o2": 1, "o3": 2, "o4": 3}):
                 
        self.root = root
        self.path = path
        self.transform = transform
        self.label_map = label_map
        self.import_records()

    def import_records(self):
        xml_files = [elm for elm in os.listdir(self.path) if ".xml" in elm]
        self.record_paths = [os.path.join(self.root,self.path, xml_file) for xml_file in xml_files]

    def nb_classes(self):
        return len(self.objects)

    def __len__(self):
        return len(self.record_paths)
    
    def img_load(self, path):
            im = Image.open(path)
            if len(list(im.split())) == 1:
                im = im.convert('RGB')
            return im
        
    def xml_reader(self, path):
        root = ElementTree.parse(path).getroot()
        img_path = root.findtext("path").split("\\")[-1]
        
        annots = []
        for annot_xml in root.findall("object"):
            annot_dict = {}
            annot_dict["class"] = annot_xml.findtext("name")
            
            bbox_xml = annot_xml.findall("bndbox")[0]
            xmin = int(bbox_xml.findtext("xmin"))
            ymin = int(bbox_xml.findtext("ymin"))
            xmax = int(bbox_xml.findtext("xmax"))
            ymax = int(bbox_xml.findtext("ymax"))
            area = (xmax - xmin) * (ymax - ymin)
            # if area<100:
            #     diff = int(math.sqrt(100-area)+1)
            #     xmin -= diff
            #     ymin -= diff
            #     xmax += diff
            #     ymax += diff
            annots.append([xmin,
                           ymin,
                           xmax,
                           ymax,
                           int(self.label_map[annot_xml.findtext("name")]),
                           int(area)])
            
        return img_path, annots
    
    def __getitem__(self, index):
        img_path, annots = self.xml_reader(self.record_paths[index])
        img_path = os.path.join(self.root, self.path, img_path)
        sample = {}
        sample["image"] = self.img_load(img_path)
        sample["annot"] = np.asarray(annots,dtype=np.float)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample