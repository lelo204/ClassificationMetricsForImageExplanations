import numpy as np
import pandas as pd
import krippendorff as kr

from consts.paths import get_heatmaps_folder
from consts.consts import XmethodArgs


rankings_path = 'evaluation/rankings'

metrics = ['precision','sensitivity','false-negative-rate','false-positive-rate','specificity','accuracy','f1-score']
negFI_xmethods = ['bcos', 'shap', 'intgrad', 'lrp']

architectures = ['resnet50', 'vgg11_bn']
datasets = ['carsncats', 'mountaindogs', 'ilsvrc2012']

alpha_rankings = pd.DataFrame(columns=['metric_architecture', 'alpha_mean', 'alpha_median'])

for architecture in architectures:
    for metric in metrics:
        if metric == 'precision':
            ranking_mean = pd.DataFrame(columns=['dataset', 'bcos', 'gradcam', 'gradcam++', 'intgrad', 'lime', 'lrp', 'shap', 'smoothgrad'])
            ranking_median = pd.DataFrame(columns=['dataset', 'bcos', 'gradcam', 'gradcam++', 'intgrad', 'lime', 'lrp', 'shap', 'smoothgrad'])
            for dataset in datasets:
                means = []
                medians = []
                for xmethod in list(XmethodArgs):
                    _, path = get_heatmaps_folder(xmethod, dataset, architecture, ckpt=None)
                    df = pd.read_csv(f'{path}.csv')
                    means.append((str(xmethod), np.mean(df[metric][:2])))
                    medians.append((str(xmethod), df[metric].iloc[-2]))
                sorted_means = sorted([(name, j+1) for j, (name, _) in enumerate(sorted(means, key=lambda tup: tup[1], reverse=True))], key=lambda tup: tup[0])
                sorted_medians = sorted([(name, j+1) for j, (name, _) in enumerate(sorted(medians, key=lambda tup: tup[1], reverse=True))], key=lambda tup: tup[0])
                new_row_means = {'dataset': dataset, 'bcos': sorted_means[0][1], 'gradcam': sorted_means[1][1], 'gradcam++': sorted_means[2][1], 'intgrad': sorted_means[3][1], 'lime': sorted_means[4][1], 'lrp': sorted_means[5][1], 'shap': sorted_means[6][1], 'smoothgrad': sorted_means[7][1]}
                new_row_medians = {'dataset': dataset, 'bcos': sorted_medians[0][1], 'gradcam': sorted_medians[1][1], 'gradcam++': sorted_medians[2][1], 'intgrad': sorted_medians[3][1], 'lime': sorted_medians[4][1], 'lrp': sorted_medians[5][1], 'shap': sorted_medians[6][1], 'smoothgrad': sorted_medians[7][1]}
                ranking_mean = pd.concat([ranking_mean, pd.DataFrame([new_row_means])], ignore_index=True)
                ranking_median = pd.concat([ranking_median, pd.DataFrame([new_row_medians])], ignore_index=True)
            ranking_mean.to_csv(f'{rankings_path}/{metric}_{architecture}_mean.csv')
            ranking_median.to_csv(f'{rankings_path}/{metric}_{architecture}_median.csv')

            alpha_mean = kr.alpha(ranking_mean.iloc[-4:, -8:].to_numpy(dtype='int64'), level_of_measurement='ordinal')
            alpha_median = kr.alpha(ranking_median.iloc[-4:, -8:].to_numpy(dtype='int64'), level_of_measurement='ordinal')
            new_row_alphas = {'metric_architecture': f'{metric}_{architecture}', 'alpha_mean': alpha_mean, 'alpha_median': alpha_median}
            alpha_rankings = pd.concat([alpha_rankings, pd.DataFrame([new_row_alphas])], ignore_index=True)
                
        else:
            ranking_mean = pd.DataFrame(columns=['dataset', 'bcos', 'intgrad', 'lrp', 'shap'])
            ranking_median = pd.DataFrame(columns=['dataset', 'bcos', 'intgrad', 'lrp', 'shap'])
            for dataset in datasets:
                means = []
                medians = []
                for xmethod in negFI_xmethods:
                    _, path = get_heatmaps_folder(xmethod, dataset, architecture, ckpt=None)
                    df = pd.read_csv(f'{path}.csv')
                    means.append((str(xmethod), np.mean(df[metric][:2])))
                    medians.append((str(xmethod), df[metric].iloc[-2]))
                sorted_means = sorted([(name, j+1) for j, (name, _) in enumerate(sorted(means, key=lambda tup: tup[1], reverse=True))], key=lambda tup: tup[0])
                sorted_medians = sorted([(name, j+1) for j, (name, _) in enumerate(sorted(medians, key=lambda tup: tup[1], reverse=True))], key=lambda tup: tup[0])
                new_row_means = {'dataset': dataset, 'bcos': sorted_means[0][1], 'intgrad': sorted_means[1][1], 'lrp': sorted_means[2][1], 'shap': sorted_means[3][1]}
                new_row_medians = {'dataset': dataset, 'bcos': sorted_medians[0][1], 'intgrad': sorted_medians[1][1], 'lrp': sorted_medians[2][1], 'shap': sorted_medians[3][1]}
                ranking_mean = pd.concat([ranking_mean, pd.DataFrame([new_row_means])], ignore_index=True)
                ranking_median = pd.concat([ranking_median, pd.DataFrame([new_row_medians])], ignore_index=True)
            ranking_mean.to_csv(f'{rankings_path}/{metric}_{architecture}_mean.csv')
            ranking_median.to_csv(f'{rankings_path}/{metric}_{architecture}_median.csv')

            alpha_mean = kr.alpha(ranking_mean.iloc[-4:, -4:].to_numpy(dtype='int64'), level_of_measurement='ordinal')
            alpha_median = kr.alpha(ranking_median.iloc[-4:, -4:].to_numpy(dtype='int64'), level_of_measurement='ordinal')
            new_row_alphas = {'metric_architecture': f'{metric}_{architecture}', 'alpha_mean': alpha_mean, 'alpha_median': alpha_median}
            alpha_rankings = pd.concat([alpha_rankings, pd.DataFrame([new_row_alphas])], ignore_index=True)

alpha_rankings.to_csv(f'{rankings_path}/alphas.csv')
