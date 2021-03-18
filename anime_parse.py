import argparse
import pascal_voc_writer as voc
import pandas as pd
from tqdm import tqdm
import os
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser(description='Convert animeface csv to Voc style')
    parser.add_argument('--root_path', help= 'The root path that contatin images and csv')
    parser.add_argument('--width', help='Width of your screen', default=1280)
    parser.add_argument('--height', help='height of your screen', default=720)
    args = parser.parse_args()
    return args

def parse_to_voc(path, width, height):
    csv_path = os.path.join(path, 'ROI.csv')
    csv = pd.read_csv(csv_path)
    for i in tqdm(range(len(csv))):
        img_name = os.path.basename(csv['image_path'][i])
        writer = voc.Writer(os.path.join(path, img_name), width, height)
        xml_save_path = os.path.join(path, '{0}.xml'.format(i))

        for bb_str in csv['ROI'][i][:-1].split('/'):
            bb = bb_str.split('|')
            writer.addObject('face', bb[1], bb[0], bb[3], bb[2])

        writer.save(xml_save_path)

def main():
    args = parse_args()
    root_paths = glob(os.path.join(args.root_path, '*'))
    
    print(root_paths)
    
    for root in root_paths:
        print('Process {0}'.format(os.path.basename(root)))
        parse_to_voc(root, args.width, args.height)
if __name__ == '__main__':
    main()

    
