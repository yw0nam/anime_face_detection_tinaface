from tqdm import tqdm
import os
import shutil
import cv2
import pandas as pd
import argparse
import xml.etree.ElementTree as ET
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser(description='Convert animeface csv to Voc style')
    parser.add_argument('--root_path', help= 'The path that contatin images and csv')
    parser.add_argument('--save_path', help= 'The path that save contents',
                        default='./data/anime_face_detector/')
    
    args = parser.parse_args()
    return args

def parse_xml_danbooru(csv, root_path, save_path):
    try:
        os.makedirs(save_path +'/0--illustration/')
    except FileExistsError:
        pass
    for idx in tqdm(csv['index']):
        shutil.copyfile(root_path+'/{0:06d}.jpg'.format(idx), 
                        save_path +'/0--illustration/{0:06d}.jpg'.format(idx))
        shutil.copyfile(root_path+'/{0:06d}.xml'.format(idx), 
                        save_path +'/Annotations/{0:06d}.xml'.format(idx))
        img = cv2.imread(save_path +'/0--illustration/{0:06d}.jpg'.format(idx))
        h, w, _ = img.shape
        replace_xml_folder(save_path +'/Annotations/{0:06d}.xml'.format(idx), 
                           '0--illustration', h, w)
        
def replace_xml_folder(xml_path, folder_name, h, w, img_save_path=None):
    xml = open(xml_path, 'rt', encoding='UTF8')
    tree = ET.parse(xml)
    root = tree.getroot()
    path = root.find('path')
    if path == None:
        pass
    else:
        path.text = img_save_path
    
    is_null = root.find("object")
    label_text = '0' if is_null == None else '1'
    ET.SubElement(root, "label").text = label_text
    
    size = root.find("size") 
    if size == None:
        ET.SubElement(root, "size").text = '1'
        size = root.find('size')
        ET.SubElement(size, 'width').text = str(w)
        ET.SubElement(size, 'height').text = str(h)
    else:
        size.clear()
        ET.SubElement(size, 'width').text = str(w)
        ET.SubElement(size, 'height').text = str(h)

    folder_tag = root.find("folder") 
    folder_tag.text = folder_name 
    indent(root)
    indent(size)
    tree.write(xml_path)

def indent(elem, level=0):
    i = "\n" + level*"    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "    "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
            
def main():
    args = parse_args()
  
    train_path = os.path.join(args.root_path, 'trainval.txt')
    val_path = os.path.join(args.root_path, 'test.txt')

    train = pd.read_csv(train_path, 
                    header=None, names=['index'])
    val = pd.read_csv(val_path, 
                   header=None, names=['index'])
    
    train_dst_path = os.path.join(args.save_path, 'train')
    val_dst_path = os.path.join(args.save_path, 'val')
    dataset_num = len(glob(train_dst_path+'/*--*'))
    folder_name = '{0}--{1}'.format(dataset_num, 'illustration')
    
    parse_xml_danbooru(train, args.root_path, train_dst_path)
    parse_xml_danbooru(val, args.root_path, val_dst_path)
    
if __name__ == '__main__':
    main()
