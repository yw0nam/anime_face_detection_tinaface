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

def parse_xml_anime(csv, root_path, save_path,
                    dataset_num, folder_name,
                   anime_name):
    
    try:
        os.makedirs(os.path.join(save_path, folder_name))
    except FileExistsError:
        pass
    
    for idx in tqdm(csv['index']):
        original_img_path = os.path.join(root_path, '{0}.jpg'.format(idx))
        img_save_path = os.path.join(save_path, folder_name, 
                                     '{0}_{1}.jpg'.format(anime_name, idx))
        xml_save_path = os.path.join(save_path, 'Annotations',
                                     '{0}_{1}.xml'.format(anime_name, idx))
        
        shutil.copyfile(original_img_path, img_save_path)
        shutil.copyfile(os.path.join(root_path,'{0}.xml'.format(idx)),
                       xml_save_path)
        
        img = cv2.imread(img_save_path)
        h, w, _ = img.shape
        
        replace_xml_folder(xml_save_path, folder_name, h, w, img_save_path)
        
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
    
    root_paths = glob(os.path.join(args.root_path, '*'))
    train_dst_path = os.path.join(args.save_path, 'train')
    val_dst_path = os.path.join(args.save_path, 'val')
    
    print(root_paths)
    for root in root_paths:
        anime_name = os.path.basename(root)
        print('Process {0}'.format(anime_name))
        
        csv_path = os.path.join(root, 'ROI.csv')
        csv = pd.read_csv(csv_path)
        csv = csv.rename(columns={'Unnamed: 0':'index' })

        train = csv[:int(len(csv) * 0.8)]
        val = csv[int(len(csv) * 0.8):]

        dataset_num = len(glob(train_dst_path+'/*--*'))
        folder_name = '{0}--{1}'.format(dataset_num, anime_name)

        parse_xml_anime(train, root, train_dst_path,
                        dataset_num, folder_name, anime_name)
        parse_xml_anime(val, root, val_dst_path,
                    dataset_num, folder_name, anime_name)

if __name__ == '__main__':
    main()
