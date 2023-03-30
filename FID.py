# example of calculating the frechet inception distance in Keras for cifar10
import torch
import random
import numpy as np
from skimage.transform import resize
import argparse
import os
import pandas as pd
import csv
import cv2


def data_prep(data_path,special="none"):
    dim = (299,299)

    files = []
    if special == "none":
        files = os.listdir(data_path)
    elif special == "celebblack":
        print("In CelebBlack")
        path_1 = os.listdir(data_path+"/black_men")
        path_1 = ["black_men/"+i for i in path_1]
        random.shuffle(path_1)
        path_2 = os.listdir(data_path+"/black_women")
        path_2 = ["black_women/"+i for i in path_2]
        random.shuffle(path_2)
        files = path_1[:500]+path_2[:500]   
    else:
        print("In UniFace")
        path_1 = os.listdir(data_path+"/male")
        path_1 = ["male/"+i for i in path_1]
        random.shuffle(path_1)
        path_2 = os.listdir(data_path+"/female")
        path_2 = ["female/"+i for i in path_2]
        random.shuffle(path_2)
        files = path_1[:500]+path_2[:500]
    
    
    random.shuffle(files)
    imgs = []
    for i in range(0, 1000):
        # print("i:",i)
        # print("file:",files[i])
        img = cv2.imread(data_path+"/"+files[i])
        if img is None:
            print("img is none:",data_path+"/"+files[i])
            print("filepath:",files[i])
        try:
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
            imgs.append(img)
        except:
            continue

        
    output = torch.FloatTensor(imgs)
    output = output.permute(0, 3, 1, 2)
    output = output.numpy()
    output = output.astype(np.uint8)
    output = torch.from_numpy(output)
    print(output.shape)
    return output

def read_dict(csv_file):
    df = pd.read_csv(csv_file)
    df.columns = ['Name', 'Data_Path']
    dictt = df.to_dict()
    out_dict = {}
    for i in range(0,len(dictt['Name'].values())):
        data_name = list(dictt['Name'].values())[i]
        data_path = list(dictt['Data_Path'].values())[i]
        out_dict[data_name] = data_path
    return out_dict
    
def main(args):
    found = []
    results = []
    dictt = read_dict(args.dict_csv)
    

    _ = torch.manual_seed(123)
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=64)

    for point1_name, point1_value in dictt.items():
        for point2_name, point2_value in dictt.items():
            det_name = point1_name+"_vs_"+point2_name
            det_name_2 = point2_name+"_vs_"+point1_name
            if point1_name != point2_name and det_name not in found and det_name_2 not in found: 
                found.append(det_name)
                found.append(det_name_2)
                images1 = []
                images2 = []
                print("point1 name: ", point1_name)
                print("point2 name: ", point2_name)
    
                
                if point1_name == "celebblack" and point2_name=="uniface" :
                    images1 = data_prep(point1_value,"celebblack")
                    images2 = data_prep(point2_value, "uniface")    
                elif point1_name == "uniface" and point2_name=="celebblack" :
                    images1 = data_prep(point1_value,"uniface")
                    images2 = data_prep(point2_value, "celebblack")  
                elif point1_name == "celebblack":
                    images1 = data_prep(point1_value,"celebblack")
                    images2 = data_prep(point2_value)
                    
                elif point1_name == "uniface":
                    images1 = data_prep(point1_value,"uniface")
                    images2 = data_prep(point2_value)
                    
                elif point2_name == "celebblack":
                    print('In here 0')
                    images1 = data_prep(point1_value)
                    images2 = data_prep(point2_value,"celebblack")
                    
                elif point2_name == "uniface":
                    print('In here 1')
                    images1 = data_prep(point1_value)
                    images2 = data_prep(point2_value,"uniface")
                    
                else:
                    images1 = data_prep(point1_value)
                    images2 = data_prep(point2_value)
            
                
                fid.update(images1, real=True)
                fid.update(images2, real=False)
                fid_result = fid.compute()
                results.append(point1_name+"_vs_"+point2_name+" : "+ str(fid_result))
             
    df_results = pd.DataFrame(results)
    df_results.to_csv(args.save_path_csv, header=False, index=False)
  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict_csv', type=str,
                        help='A dictionary of the different clip scores for all possible datasets e.g. "{dataset_name:dataset_path, dataset_name_2:dataset_path_2}"')
    parser.add_argument('--save_path_csv', type=str,
                        help='path to the save the results')
    
    args = parser.parse_args()
    main(args)
    
    
