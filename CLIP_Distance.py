import pandas as pd
import numpy as np
from numpy.linalg import norm
import seaborn as sn
import argparse
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import csv
from csv import DictReader
import random

def format_scores(csv_file):
    df = pd.read_csv(csv_file)
    print(csv_file)
    df.columns = ['Name', 'Embedding']
    lis = df[['Embedding']].values.tolist()
    random.shuffle(lis)
    lis = lis[:5000]
    conditional_probs = []
    for val in lis:
        holder = []
        sample = val[0]
        split_sample = sample.split('\n')
        split_sample[0] = split_sample[0].split("tensor([[")[1:][0]
        split_sample[-1] = split_sample[-1].split("]])")[0]
        for i in split_sample:
            split_str = (i.strip()).split(",")
            for j in split_str:
                try:
                    holder.append(float(j))
                except:
                    continue
                    
        conditional_probs.append(holder)
        
    return conditional_probs

    
def main(dict_csv, save_path):
   # read the list of dictionaries from the input csv file
    list_of_dict = []
    with open("dictt.csv", 'r') as f:
        dict_reader = DictReader(f)
        list_of_dict = list(dict_reader)
        print(list_of_dict)
        
    mean_dict = {}
    
    for item in list_of_dict:
        matrix = format_scores(item['data_path'])
        mean_dict[item['data_name']] = np.array(matrix)
        
    found = []
    results = []
    for point1_name, point1_value in mean_dict.items():
        for point2_name, point2_value in mean_dict.items():
            det_name = point1_name+"_vs_"+point2_name
            det_name_2 = point2_name+"_vs_"+point1_name
            if point1_name != point2_name and det_name not in found and det_name_2 not in found:
                found.append(det_name)
                found.append(det_name_2)
                clip_distance = np.sum(point1_value*point2_value, axis=1)/(norm(point1_value, axis=1)*norm(point2_value, axis=1))
                results.append(point1_name+"_vs_"+point2_name+" : "+ str(np.mean(clip_distance)))
                print(clip_distance)
                
    df_results = pd.DataFrame(results)
    df_results.to_csv(save_path, header=False, index=False)

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict_csv', type=str,
                        help='A dictionary of the different clip scores for all possible datasets e.g. "{dataset_name:dataset_path, dataset_name_2:dataset_path_2}"')
    parser.add_argument('--save_path_csv', type=str,
                        help='path to the save the results')
    
    args = parser.parse_args()
    main(args.dict_csv, args.save_path_csv)
    

