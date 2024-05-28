import os
import torch
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from explainability import pipelines as ppl
from torchvision import transforms

from utils import get_model, summary_viz
from consts.paths import Paths, MosaicPaths
from explainability import pipelines as ppl
from consts.consts import DatasetArgs, MosaicArgs, ArchArgs, Split

# Load data
hashes_df = pd.read_csv(Paths.explainability_csv)

# Format metrics
def format_metrics(df):
    metrics_text = ""
    metrics_text += f"Precision: {df['precision']:.3f}\n"
    metrics_text += f"Sensitivity: {df['sensitivity']:.3f}\n"
    metrics_text += f"False Negative Rate: {df['false-negative-rate']:.3f}\n"
    metrics_text += f"False Positive Rate: {df['false-positive-rate']:.3f}\n"
    metrics_text += f"Specificity: {df['specificity']:.3f}\n"
    metrics_text += f"Accuracy: {df['accuracy']:.3f}\n"
    metrics_text += f"F1 Score: {df['f1-score']:.3f}"
    return metrics_text

# Get mosaics directory
def get_mosaics_dir(dataset: MosaicArgs):
    return MosaicPaths.get_from(dataset).images_folder

# Get methods for architecture and dataset        
def get_methods(dataset: DatasetArgs, architecture: ArchArgs):
    return hashes_df[(hashes_df['architecture'] == str(architecture)) &
                    (hashes_df['dataset'] == str(dataset))]['xai_method'].unique()

# Get heatmap array
def get_heatmap(dataset: DatasetArgs, architecture: ArchArgs, method: str, mosaic_name: str):
    hash = hashes_df[(hashes_df['architecture'] == str(architecture)) &
                    (hashes_df['dataset'] == str(dataset)) &
                    (hashes_df['xai_method'] == method)]['hash'].values[0]
                    
    heatmap_file = os.path.join(Paths.explainability_path, hash, mosaic_name.replace('.jpg', '.npy'))
    return np.load(heatmap_file)

# Get metrics dataframe 
def get_metrics_df(dataset: DatasetArgs, architecture: ArchArgs, method: str, idx: int):
    hash = hashes_df[(hashes_df['architecture'] == str(architecture)) &
                    (hashes_df['dataset'] == str(dataset)) &
                    (hashes_df['xai_method'] == method)]['hash'].values[0]
                    
    csv_file = os.path.join(Paths.explainability_path, hash + '.csv')
    
    # Read the CSV file into a DataFrame
    metrics_df = pd.read_csv(csv_file)
    
    # Filter the DataFrame to the specific index 'idx'
    metrics_df = metrics_df.iloc[idx]
    
    return metrics_df

# Save summary image
def save_summary_image(dataset: DatasetArgs, architecture: ArchArgs, mosaic_name: str, info: bool, fig):
    summary_dir = os.path.join(Paths.mosaics_summary_path, str(dataset), str(architecture))
    os.makedirs(summary_dir, exist_ok=True)
    
    summary_path = os.path.join(summary_dir, mosaic_name.replace('.jpg', '_summary_info.jpg') if info else mosaic_name.replace('.jpg', '_summary.jpg'))
    fig.savefig(summary_path, bbox_inches='tight')  # Use bbox_inches='tight' to avoid cropping

# Parse arguments for generating summary images
def parse_summary_images_args():
    parser = argparse.ArgumentParser(description='Generate summary images for all or a specific mosaic')
    parser.add_argument('--dataset', type=DatasetArgs, help='Dataset name')
    parser.add_argument('--architecture', type=ArchArgs, help='Architecture name') 
    parser.add_argument('--mosaic_idx', type=int, help='Mosaic Index (mosaics are expected to be named mosaic_idx.jpg)')
    parser.add_argument('--mosaic_dataset', type=MosaicArgs, help='Mosaic dataset to generate summary info.')
    parser.add_argument('--info', action='store_true', default=False, help='Whether to create visualization with or without further information like metrics and classes.')
    return parser.parse_args()

def generate_low_info_summary(dataset: DatasetArgs, architecture: ArchArgs, mosaics: MosaicArgs, mosaic_idx: int):
    # Load mosaic image
    mosaic_name = str(mosaic_idx) + '.jpg'
    mosaic_img = plt.imread(os.path.join(MosaicPaths.get_from(mosaics).images_folder, mosaic_name))

    # Get methods
    methods = get_methods(dataset, architecture)

    heatmap_dict = {}
                          
    for j, method in enumerate(methods):
        # Get heatmap
        heatmap = get_heatmap(dataset, architecture, method, mosaic_name)
        heatmap_dict[method] = heatmap

    #print(heatmap_dict.keys())
    fig = summary_viz(img = mosaic_img, heatmap_dict=heatmap_dict, rescale=True)

    # Save figure
    save_summary_image(dataset, architecture, mosaic_name, fig=fig, info=False)
    plt.close()

def get_model_prediction(filename: str, model, bcos=False):

    img = Image.open(filename)

    if bcos:
        img_tensor = model.transform(img)
        img_tensor = img_tensor[None]
        preds = model(img_tensor)

    else:
        transformation = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                            ])
        
        img = img.convert('RGB')
        img_tensor = transformation(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)
        preds = model(img_tensor)

    return preds

def generate_high_info_summary(dataset: DatasetArgs, architecture: ArchArgs, mosaic_idx: int, mosaics: MosaicArgs):
    mosaics_df = pd.read_csv(MosaicPaths.get_from(mosaics).csv_path)
    mosaic_filenames = mosaics_df['filename'].tolist()

    # Load mosaic image
    mosaic_img = plt.imread(os.path.join(MosaicPaths.get_from(mosaics).images_folder, f'{mosaic_idx}.jpg'))

    # Get methods
    methods = get_methods(dataset, architecture)

    heatmap_dict, info = {}, {}
    info['architecture'] = architecture
    info['mosaic_idx'] = mosaic_idx

    model = get_model(str(architecture), use_bcos=False)                    # get the model under investigation in eval-mode
    model_bcos = get_model(str(architecture), use_bcos=True)
    
    mosaic_dataset = ppl.DATASETS[mosaics].load_dataset()
    mosaic_filepaths, target_classes, image_filenames, image_labels = mosaic_dataset.get_subset(Split.VAL)

    info['classes'] = image_labels[mosaic_idx]
    info['filenames'] = image_filenames[mosaic_idx]
    info['target_class'] = target_classes[mosaic_idx]

    info['preds'] = torch.topk(get_model_prediction(mosaic_filepaths[mosaic_idx], model=model), k=3)[1]
    info['preds_bcos'] = torch.topk(get_model_prediction(mosaic_filepaths[mosaic_idx], model=model_bcos, bcos=True), k=3)[1]

    for j, method in enumerate(methods):
        # Get heatmap
        heatmap = get_heatmap(dataset, architecture, method, f'{mosaic_idx}.jpg')
        heatmap_dict[method] = heatmap

        # Get the mosaic index in the DataFrame
        mosaic_idx = mosaic_filenames.index(str(mosaic_idx)+'.jpg')
        metrics_df = get_metrics_df(dataset, architecture, method, mosaic_idx)
        metrics_df.drop('filename', inplace=True)
        info[method] = metrics_df
    
    fig = summary_viz(img = mosaic_img, heatmap_dict=heatmap_dict, info = info, rescale=True)

    # Save figure
    save_summary_image(dataset, architecture, f'{mosaic_idx}.jpg', fig=fig, info=True)
    plt.close()

if __name__ == '__main__':
    # create a summary of mosaic 0 via:
    # python sumgen_script.py --dataset carsncats --architecture vgg11_bn --mosaic_idx 0 --mosaic_dataset carsncats_mosaic
    args = parse_summary_images_args()

    # Check which function to execute based on the command-line arguments
    if not args.info:
        generate_low_info_summary(args.dataset, args.architecture, args.mosaic_dataset, args.mosaic_idx)
    elif args.info:
        generate_high_info_summary(args.dataset, args.architecture, mosaic_idx=args.mosaic_idx, mosaics=args.mosaic_dataset)
