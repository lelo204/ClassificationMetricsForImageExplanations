import os
import time
import shutil
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from utils import get_model
from explainability import pipelines as ppl
from explainability.utils import create_dir
from evaluation.compute_metrics import compute_metric
from consts.paths import MosaicPaths, Paths, get_heatmaps_folder
from dataset_manager.multiclass_mosaic_creation import mosaic_creation
from consts.consts import DatasetArgs, MosaicArgs, ArchArgs, XmethodArgs, Split


def make_parser():
    parser = argparse.ArgumentParser('XAI Evaluation.')
    parser.add_argument('--dataset', type=DatasetArgs, choices=list(DatasetArgs), help='Dataset for mosaic generation.')
    parser.add_argument('--mosaics', type=MosaicArgs, choices=list(MosaicArgs), default=None, help='Name of mosaic dataset.')
    parser.add_argument('--mosaics_per_class', type=int, default=None, help='Number of mosaics that will be created per class, if argument is None, existing mosaics will be used.')
    parser.add_argument('--architecture', type=ArchArgs, choices=list(ArchArgs), help='Classification model.')
    parser.add_argument('--xai_method', type=XmethodArgs, choices=list(XmethodArgs), help='XAI method used for generating heatmaps.')
    parser.add_argument('--seed', type=int, default=42)

    return parser


def main(args):

    if args.mosaics_per_class is not None:      # creates new folder with n mosaics and deletes existing ones (including the heatmaps already created for the existing ones)

        # delete old mosaics
        if os.path.exists(MosaicPaths.get_from(args.mosaics).csv_path):
            os.remove(MosaicPaths.get_from(args.mosaics).csv_path)
            shutil.rmtree(MosaicPaths.get_from(args.mosaics).images_folder)     

            try:
                hash_df = pd.read_csv(Paths.explainability_csv)
            except FileNotFoundError:
                pass
            else:
                relevant_hashs = hash_df[hash_df['dataset'] == str(args.dataset)]['hash'].to_list()     # extract hashs for experiments made with the old mosaics of the selected dataset
                relevant_hashs_idx = hash_df[hash_df['dataset'] == str(args.dataset)].index

                # next section deletes heatmaps if they were calculated for obsolete mosaics (which have since been deleted)
                for hash in relevant_hashs:
                    path = os.path.join(Paths.explainability_path, hash)                # path to folder were heatmaps are saved
                    csv_path = os.path.join(Paths.explainability_path, f'{hash}.csv')   # path to csv-file that contains evaluation metrics
                    if os.path.exists(path):
                        shutil.rmtree(path)         # delete old mosaics
                    if os.path.exists(csv_path):
                        os.remove(csv_path)         # delete old csv-file
                hash_df = hash_df.drop(relevant_hashs_idx)      # delete hash_df entries corresponding to old mosaics
                hash_df.to_csv(Paths.explainability_csv, index=False)        # save updated hash_df

        mosaic_creation(dataset=args.dataset, mosaic=args.mosaics, mosaics_per_class=args.mosaics_per_class, seed=args.seed)
    
    use_bcos = True if str(args.xai_method) == 'bcos' else False                    # bcos-flag because a special model architecture has to be used
    model = get_model(str(args.architecture), use_bcos=use_bcos)                    # get the model under investigation in eval-mode
    explainer = ppl.XMETHOD[args.xai_method](model)                                 # initialize xai-method

    _hash, heatmaps_path = get_heatmaps_folder(xmethod=args.xai_method, dataset=args.dataset, architecture=args.architecture, ckpt=None)        # create hash dependend on xai method, dataset and model architecture used in the experiment
    create_dir(heatmaps_path)                                                                                                                   # create a directory that will contain the calculated heatmaps
    try:
        hash_df = pd.read_csv(Paths.explainability_csv)                             # hash_df keeps track of executed experiments and is updated during every run
    except FileNotFoundError:
        hash_df = pd.DataFrame(columns=['hash', 'dataset', 'architecture', 'xai_method'])
    if len(hash_df[hash_df['hash'] == _hash]) == 0:
        new_row = {'hash': _hash, 'dataset': args.dataset, 'architecture': args.architecture, 'xai_method': args.xai_method}
        hash_df = pd.concat([hash_df, pd.DataFrame([new_row])], ignore_index=True)
        hash_df.to_csv(Paths.explainability_csv, index=False)

    try:
        log_df = pd.read_csv(Paths.logfile_csv)                 
    except FileNotFoundError:
        log_df = pd.DataFrame(columns=['timestamp', 'dataset', 'architecture', 'xai_method', 'execution_time_in_s'])
    
    eval_df = pd.DataFrame(columns=['filename', 'precision', 'sensitivity', 'false-negative-rate', 'false-positive-rate', 'specificity', 'accuracy', 'f1-score'])       # dataframe for keeping track of evaluation metrics per experiment
    mosaic_dataset = ppl.DATASETS[args.mosaics].load_dataset()
    num_mosaics = len(mosaic_dataset.get_subset(Split.VAL)[0])                  # get_subset function of mosaic dataset does not use Split.subset argument to actually select images in the subset, therefore all existing mosaics are counted
    
    start = time.time()
    for mosaic_filepath, target_class, images_filenames, image_labels in tqdm(zip(*mosaic_dataset.get_subset(Split.VAL)), total=num_mosaics):           # Split.VAL does not constrain selection of mosaics (cf. get_subset function of MosaicDataset class)
        mosaic_name = os.path.splitext(os.path.basename(mosaic_filepath))[0]
        output_path = os.path.join(heatmaps_path, f'{mosaic_name}.npy')

        if not os.path.exists(output_path):                                         
            mosaic_explanation = explainer.explain(mosaic_filepath, target_class)
            np.save(output_path, mosaic_explanation)
        else:
            mosaic_explanation = np.load(output_path)

        metrics = compute_metric(mosaic_explanation, target_class, image_labels, metric='all')       # image_labels contains label order for mosaics from csv-file
        metrics = list(metrics)
        new_row = {'filename': mosaic_name, 'precision': metrics[0], 'sensitivity': metrics[1], 'false-negative-rate': metrics[2], 'false-positive-rate': metrics[3], 'specificity': metrics[4], 'accuracy': metrics[5], 'f1-score': metrics[6]}

        eval_df = pd.concat([eval_df, pd.DataFrame([new_row])], ignore_index=True)                  # eval_df contains evaluation metrics (columns) of xai method for mosaics (rows)
    end = time.time()

    log_row = {'timestamp': datetime.now().strftime('%Y-%m-%d_%H%M'), 'dataset': args.dataset, 'architecture': args.architecture, 'xai_method': args.xai_method, 'execution_time_in_s': end-start}
    log_df = pd.concat([log_df, pd.DataFrame([log_row])], ignore_index=True)
    log_df.to_csv(Paths.logfile_csv, index=False)

    print(f'\nMedian focus: {eval_df["precision"].median()}\nVariance focus: {eval_df["precision"].var()}')
    median_row = {'filename': 'Median:', 'precision': eval_df["precision"].median(), 'sensitivity': eval_df["sensitivity"].median(), 'false-negative-rate': eval_df["false-negative-rate"].median(), 'false-positive-rate': eval_df["false-positive-rate"].median(), 'specificity': eval_df["specificity"].median(), 'accuracy': eval_df["accuracy"].median(), 'f1-score': eval_df["f1-score"].median()}
    variance_row = {'filename': 'Variance:', 'precision': eval_df["precision"].var(), 'sensitivity': eval_df["sensitivity"].var(), 'false-negative-rate': eval_df["false-negative-rate"].var(), 'false-positive-rate': eval_df["false-positive-rate"].var(), 'specificity': eval_df["specificity"].var(), 'accuracy': eval_df["accuracy"].var(), 'f1-score': eval_df["f1-score"].var()}
    eval_df = pd.concat([eval_df, pd.DataFrame([median_row]), pd.DataFrame([variance_row])], ignore_index=True)     # adds median and variance for every evaluation metric in last two rows of dataframe
    eval_df.to_csv(f'{heatmaps_path}.csv', index=False)


if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args)
