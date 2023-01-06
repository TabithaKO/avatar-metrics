import torch
from PIL import Image
import open_clip
import os
import pandas as pd
import logging
import argparse


def execute(output_path, input_list):
    logging.basicConfig(filename="clip.log", level=logging.NOTSET)

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="laion400m_e32")
    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    output_vecs = []
    try:
        text = tokenizer(input_list)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            df = pd.DataFrame(text_features)
            df.to_csv(output_path, header=False, index=False)
        
    except Exception as err:
        logging.warn("error message:",{err})
            

def main(args):
    execute(args.output_path, args.list)
    print("DONE!!!")
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str,
                        help='path to the output csv')
    parser.add_argument('--list', type=list,
                        help='the list of strings you want embedded')
    
    args = parser.parse_args()
    main(args.output_path, args.list)