## Overview
This repository houses the code required to calculate the perfromance metrics of face generation models:

- VAEs
- GANs
- Diffusion Models

The different metrics in this repo include:
- Inception Score [(IS)](https://github.com/sbarratt/inception-score-pytorch)
- Frechet Inception Distance [(FID)](https://github.com/mseitzer/pytorch-fid)
- Learned Perceptual Image Patch Similarity [(LPIPS)](https://github.com/richzhang/PerceptualSimilarity)
- Structural Similarity Instance Metric [(SSIM)](https://github.com/Po-Hsun-Su/pytorch-ssim)

## Requirements & Steps
Please ensure you have Python 3.6+ installed on your device.
```
$ git clone https://github.com/TabithaKO/avatar-metrics.git
$ cd avatar-metrics
$ bash run.sh
```

The bash script will clone 2 repositories:
- [open_clip](https://github.com/mlfoundations/open_clip)
- [face-parse-Pytorch](https://github.com/TabithaKO/face-parsing.PyTorch)

While using this repository, I assume that you have a folder full of real face images and a folder full of generated face images. Here are the instructions on how to compute various metrics to compute the differences between the **real** and **generated** images.

**CROP THE FACES**

While the face detection script isn't perfect, if you wish to crop your images around the subject's face you can use the following script:
```
$ python3 face-parsing.PyTorch/face_detect.py --input_dir "input directory" --output_dir "output directory"
```
**FID SCORE:**

The FID compares the distribution of generated images with the distribution of real images. The FID score is defined in the image below as the “distance” between two normal distributions computed using the formula below. These normal distributions are described using mean and covariance. Tr refers to the [trace linear algebra operation](https://en.wikipedia.org/wiki/Trace_(linear_algebra))

```math
 d^2 = ||\mu_1-\mu_2||^2 + Tr(C_1 + C_2 - 2\sqrt{C_1\cdot C_2} 
```

```
$ python3 face-parsing.PyTorch/fid.py --path_to_fake "path to generated images" --path_to_real "path to real images" --outpath "path to csv file"
```

**SKIN TONE EUCLIDEAN DISTANCE SCORE** 

This the euclidean distance between average of the skin pixels of a real image and the skin pixels of the generated image. The face-parsing script below segments the skin pixels on the face of the input image and averages out the values into a 3-dimensional [B,G,R] vecor which is stored in the output csv file.
```
$ python3 face-parsing.PyTorch/face-parse.py --bucket_name "S3 bucket name" --S3_ID "S3 ID" --S3_Key "S3 Key" \
--input_dir "your input images" --output_name "output csv file"
```
If you have a folder full of images that you want to split across the average [B,G,R] skin values threshold, you can use the domain splitting code. Why do I have this code? Well, my research is based on creating generative models that produce images whose skin complexion is more representative of the general population. I wanted to create models that generated faces with darker skin tones (like mine :)).

```
$ python3 face-parsing.PyTorch/domain_splitting.py  --bucket_name "S3 bucket name" --S3_ID "S3 ID" \
--S3_Key "S3 Key"  --input_dir "your input images" --output_dir_domain_1 "path to first output folder" \
--output_dir_domain_2 "path to second output folder" --b_avg "average blue value" \
--g_avg "average green value"  --r_avg "average red value"
```

**CLIP EMBEDDINGS**

Get the image embeddings:
```
$ python3 open_clip/get_embeddings.py --output_path "path to output csv" --input_path "path to input dir"
```
Get text embeddings:
```
$ python3 open_clip/get_embeddings.py --output_path "path to output csv" --list "list of strings to embed"
```
Compute the image labels based on the face attributes from the CelebA annotations
```
$ python3 open_clip/get_embeddings.py --output_path "path to output csv" --input_path "path to input dir"
```

**CLIP Scores**
In order to compute the CLIP Score and the CLIP Fretchet Distance you need to run:
```
# computing CLIP score
thisdict = [{"data_name":"dataset_0","data_path":"path_to_dataset_0_CLIP_embeddings.csv"},
           {"data_name":"dataset_1", "data_path":"path_to_dataset_1_CLIP_embeddings.csv"},
           ...
           {"data_name":"dataset_n", "data_path":"path_to_dataset_n_CLIP_embeddings.csv"}]

with open('dict_name.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = ["data_name","data_path"])
    writer.writeheader()
    writer.writerows(thisdict)
    
$ python3 CLIP_Distance.py --dict_csv dict_name.csv --save_path_csv path_to_save_output.csv
```
```
# computing CLIP Fretchet Distance
thisdict = [{"data_name":"dataset_0","data_path":"path_to_dataset_0_CLIP_embeddings.csv"},
           {"data_name":"dataset_1", "data_path":"path_to_dataset_1_CLIP_embeddings.csv"},
           ...
           {"data_name":"dataset_n", "data_path":"path_to_dataset_n_CLIP_embeddings.csv"}]

with open('dict_name.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = ["data_name","data_path"])
    writer.writeheader()
    writer.writerows(thisdict)
    
$ python3  CLIP_Wasserstein.py --dict_csv dict_name.csv --save_path_csv path_to_save_output.csv
```

TO DO:
- Write a dedicated script for LPIPS calculations on generated images
- Work on computing FID score using generative networks
