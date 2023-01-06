import torch
from PIL import Image
import open_clip
import os
import pandas as pd
import logging
import argparse

def execute(dir_path, output_path):
    logging.basicConfig(filename="clip.log", level=logging.NOTSET)

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="laion400m_e32"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")

    output_vecs = []
    images = os.listdir(dir_path)
    for i in range(0, len(images)):
        try:
            img_name = dir_path +"/" + images[i]
            image = preprocess(Image.open(img_name)).unsqueeze(0)

            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            output_vecs.append([img_name, image_features])
            if i % 100 == 0:
                logging.info(f"iter = {i}")
                df = pd.DataFrame(output_vecs)
                df.to_csv(output_path, header=False, index=False)
        except Exception as err:
            logging.warn(f"iter {i}: {err}")
            

def main(args):
    execute(args.dir_name, args.output_path)
    print("DONE!!!")
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str,
                        help='path to the output csv')
    parser.add_argument('--input_path', type=str,
                        help='path to the image directory')
    
    args = parser.parse_args()
    main(args.input_path, args.output_path)