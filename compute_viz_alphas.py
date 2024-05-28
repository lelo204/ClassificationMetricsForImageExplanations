import argparse
import krippendorff as kr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import os

from scipy.stats import rankdata
from consts.paths import get_heatmaps_folder
from consts.consts import MosaicArgs, ArchArgs, XmethodArgs, DatasetArgs, Split
import seaborn as sns

from scipy.stats import spearmanr

from utils.helpers import get_xmethod_name, get_dset_name


def compute_krippendorf_alpha(data, level_of_measurement='ordinal'):
    """Computes Krippendorf's alpha (agreement metric) with interval and nominal metric.

    Args:
        data (numpy.ndarray): Reliability data, defined as one row per coder and one column per unit.
            To measure agreement between xai-methods, coders are xai-methods and columns are values for images.

    Returns:
        float: Krippendorf's alpha for ordinal values.
    """
    if level_of_measurement == 'ordinal':
        max_rank = int(np.max(np.ceil(data)))
        return kr.alpha(data, level_of_measurement='ordinal')#, value_domain=range(1, max_rank+2))
    else:
        return kr.alpha(data, level_of_measurement=level_of_measurement)

def make_parser():
    parser = argparse.ArgumentParser('XAI Evaluation.')
    parser.add_argument('--dataset', type=DatasetArgs, choices=list(DatasetArgs), help='Dataset for mosaic generation.')
    parser.add_argument('--architecture', type=ArchArgs, choices=list(ArchArgs), help='Classification model.')
    parser.add_argument('--xai_methods', type=XmethodArgs, nargs='*')
    parser.add_argument('--negative_FI', type=bool, default=False)

    return parser

def main(args):
    metric = args.metric
    xmethods = args.xai_methods

    idx = 0
    reliability_matrix = None

    for xmethod in xmethods:
        # get path csv-file containing metric scores
        h, path = get_heatmaps_folder(xmethod=xmethod, dataset=args.dataset, architecture=args.architecture, ckpt=None)
        # take csv from path
        try:
            df = pd.read_csv(path + '.csv')
        except FileNotFoundError:
            print('Could not find the data for dataset {}, architecture {} and method {}.'.format(args.dataset, args.architecture, xmethod))
            return

        # if it does not exist, create reliability matrix in the required shape
        if type(reliability_matrix) is not np.ndarray:
            reliability_matrix = np.zeros(shape=(len(xmethods), df.shape[0]-2))

        # add data for xai-method to the reliability matrix
        reliability_matrix[idx,:] = df[metric][:-2].to_numpy().transpose()
        idx += 1
    print(reliability_matrix.shape)

    # compute the agreement between different xai-methods -> inter-rater reliability
    # high agreement means -> xai-methods are consistently ranked in the same order over different images by saliency metric
    ranked_reliability = rankdata(-reliability_matrix, axis=0, nan_policy='omit')
    alpha_ord = compute_krippendorf_alpha(data = ranked_reliability.transpose())
    print('Krippendorffs alpha (ordinal metric) for', str(metric) ,':', alpha_ord)

    # compute the correlation between the metric-results/the agreement between images -> inter-method reliability
    # high correlation means -> images are consistently rated good/bad between different xai-methods
    corr = spearmanr(reliability_matrix, axis=1).statistic #spearman correlation
    xmethod_names = [get_xmethod_name(str(xmethod)) for xmethod in xmethods]

    ax = sns.heatmap(corr, cmap='coolwarm', annot=np.around(corr, decimals=2), vmax=1, vmin=-1, xticklabels=xmethod_names, yticklabels=xmethod_names)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='left')
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    if not os.path.isdir('evaluation/figures'):
        os.makedirs('evaluation/figures')
    plt.savefig('evaluation\\figures\\correlation_'+ str(args.dataset) + '_' + str(args.architecture) + '_' + str(metric) + '.png', bbox_inches='tight', dpi=150)
    plt.tight_layout()
    plt.clf()

    return alpha_ord, corr

def get_metric_means(args):
    metric = args.metric
    xmethods = args.xai_methods

    idx = 0
    medians = []
    vars = []
    means = []

    for xmethod in xmethods:
        # get path csv-file containing metric scores
        h, path = get_heatmaps_folder(xmethod=xmethod, dataset=args.dataset, architecture=args.architecture, ckpt=None)
        print(h, path)
        # take csv from path
        try:
            df = pd.read_csv(path + '.csv')
        except FileNotFoundError:
            print('Could not find the data for dataset {}, architecture {} and method {} at path {}.'.format(args.dataset, args.architecture, xmethod, path))
            return

        # add data for xai-method to the medians and variances
        medians.append(df[metric][len(df)-2])
        vars.append(df[metric][len(df)-1])
        means.append(np.mean(df[metric][:-2]))
        idx += 1

    return medians, vars, means

def create_violin_plot(mean, median, variance, xmethods, metric, title, model):
    data = []

    for xmethod in xmethods:
        # get path csv-file containing metric scores
        h, path = get_heatmaps_folder(xmethod=xmethod, dataset=args.dataset, architecture=args.architecture, ckpt=None)
        # take csv from path
        try:
            df = pd.read_csv(path + '.csv')
        except FileNotFoundError:
            print('Could not find the data for dataset {}, architecture {} and method {}.'.format(args.dataset, args.architecture, xmethod))
            return
        
        # add data for xai-method to the plot data
        data.append(df[metric][:-2])
        #idx += 1
    fig = plt.figure(figsize=(6,4))
    plt.violinplot(data)

    xmethod_names = [get_xmethod_name(str(xmethod)) for xmethod in xmethods]
    
    # Set labels for each violin
    plt.xticks(range(1, len(mean)+1), xmethod_names)
    plt.xticks(rotation=45)
    plt.ylim([-0.05,1.05])

    # Plot mean and median as red dots
    plt.plot(range(1, len(mean)+1), mean, 'ro', label='Mean')
    plt.plot(range(1, len(mean)+1), median, 'r+', label='Median')

    plt.legend()
    plt.ylabel(metric.title())
    newTitle = title.replace(" positive FI", "")
    plt.title(get_dset_name(newTitle))

    if not os.path.isdir('evaluation/figures'):
        os.makedirs('evaluation/figures')
    plt.savefig('evaluation/figures/' + str(title) + '_' + str(model) + '_' + str(metric) + '.png', bbox_inches='tight')
    plt.cla()

def compute_alphas_FI(args):
    if args.negative_FI:
        metrics = ['precision','sensitivity','false-negative-rate','false-positive-rate','specificity','accuracy','f1-score']
    else:
        metrics = ['precision']
    alphas = pd.DataFrame(data=np.zeros((1, len(metrics))), columns=metrics, index=[str(args.dataset)+'_'+str(args.architecture)+'_pFI' if not args.negative_FI else str(args.dataset)+'_'+str(args.architecture)])

    for metric in metrics:
        args.metric = metric
        medians, vars, means = get_metric_means(args)
        df = pd.DataFrame(data=[means, medians, vars], index=['mean', 'median', 'variance'], columns=[args.xai_methods])
        create_violin_plot(means, medians, vars, args.xai_methods, metric=args.metric, title=str(args.dataset)+' positive FI' if not args.negative_FI else str(args.dataset), model=args.architecture)

        alpha, correlation = main(args)
        alphas[metric] = alpha


    try:
        df_alpha = pd.read_csv('evaluation\\alphas.csv', index_col=0)
        try:
            df_alpha = pd.concat([df_alpha, alphas], verify_integrity=True)
        except ValueError:
            pass
        df_alpha.to_csv('evaluation\\alphas.csv')
    except FileNotFoundError:
        alphas.to_csv('evaluation\\alphas.csv')

if __name__ == '__main__':
    # python compute_viz_alphas.py --dataset carsncats --architecture resnet50 --xai_methods bcos gradcam gradcam++ smoothgrad intgrad lime lrp shap
    args = make_parser().parse_args()

    compute_alphas_FI(args=args)
