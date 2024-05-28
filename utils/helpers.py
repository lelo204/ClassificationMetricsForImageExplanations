import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from models import alexnet, resnet18, resnet34, resnet50, resnet101, resnet152, vgg11_bn, vgg13, vgg16, vgg19


def get_model(model_name: str, use_bcos: bool = False):

    if model_name == 'alexnet':
        if not use_bcos:
            model = alexnet(pretrained=True)
        else:
            raise NotImplementedError
    elif model_name == 'resnet18':
        if not use_bcos:
            model = resnet18(pretrained=True)
        else:
            model = torch.hub.load('B-cos/B-cos-v2', 'resnet18', pretrained=True)
    elif model_name == 'resnet34':
        if not use_bcos:
            model = resnet34(pretrained=True)
        else:
            model = torch.hub.load('B-cos/B-cos-v2', 'resnet34', pretrained=True)
    elif model_name == 'resnet50':
        if not use_bcos:
            model = resnet50(pretrained=True)
        else:
            model = torch.hub.load('B-cos/B-cos-v2', 'resnet50', pretrained=True)
    elif model_name == 'resnet101':
        if not use_bcos:
            model = resnet101(pretrained=True)
        else:
            model = torch.hub.load('B-cos/B-cos-v2', 'resnet101', pretrained=True)
    elif model_name == 'resnet152':
        if not use_bcos:
            model = resnet152(pretrained=True)
        else:
            model = torch.hub.load('B-cos/B-cos-v2', 'resnet152', pretrained=True)
    # atm only vgg11_bnu is available through torch hub as a bcos network
    elif model_name == 'vgg11_bn':
        if not use_bcos:
            model = vgg11_bn(pretrained=True)               # we use vgg11_bn here, because bcos also uses a VGG11 with batch normalization
        else:
            model = torch.hub.load('B-cos/B-cos-v2', 'vgg11_bnu', pretrained=True)
    elif model_name == 'vgg13':
        if not use_bcos:
            model = vgg13(pretrained=True)
        else:
            raise RuntimeError(f'Model {model_name} is not available in torch hub')
    elif model_name == 'vgg16':
        if not use_bcos:
            model = vgg16(pretrained=True)
        else:
            raise RuntimeError(f'Model {model_name} is not available in torch hub')
    elif model_name == 'vgg19':
        if not use_bcos:
            model = vgg19(pretrained=True)
        else:
            raise RuntimeError(f'Model {model_name} is not available in torch hub')

    return model.eval()

def get_xmethod_name(xmethod):
    if xmethod == "bcos":
        return "B-cos"
    elif xmethod == 'lime':
        return "LIME"
    elif xmethod == 'shap':
        return "SHAP"
    elif xmethod == 'gradcam':
        return "Grad-CAM"
    elif xmethod == 'gradcam++':
        return "Grad-CAM++"
    elif xmethod == 'smoothgrad':
        return "SmoothGrad"
    elif xmethod == 'intgrad':
        return "IntGrad"
    elif xmethod == 'lrp':
        return "LRP"

def get_dset_name(dataset):
    if dataset == "carsncats":
        return '"tabby" vs. "sports car"'
    elif dataset == "mountaindogs":
        return '"Greater Swiss Mountain Dog" vs. "Bernese Mountain Dog"'
    elif dataset == "ilsvrc2012":
        return "ImageNet"

def summary_viz(img: np.ndarray, heatmap_dict: dict, info: dict = None, rescale: bool = False):

    width = 0.23
    width_with_cb = 1-(3*width)
    width_cb = 0.219#(1-(4*width))/width_with_cb
    #print('width of colorbar', width_cb)

    fig, ax = plt.subplots(3,4,width_ratios=[width, width, width, width_with_cb], dpi=250)
    ax[0,0].imshow(img)
    ax[0,0].axis('off')
    for axi in ax.ravel():
        axi.set_axis_off()
    
    if not info:
        pass
    else:
        ax[0,1].text(0,0, 'architecture ' + str(info['architecture']) + '\n' +
                    'mosaic index ' + str(info['mosaic_idx']) + '\n' +
                    'classes in mosaic \n' + str(info['classes']) + '\n' +
                    'target class ' + str(info['target_class']) + '\n' +
                    'top 3 classes \n' + str(info['preds'].numpy()) + '\n' +
                    'top 3 classes bcos \n' + str(info['preds_bcos'].numpy()), size='x-small')

    idx_n = 0
    idx_p = 1

    for xai_method in heatmap_dict.keys():
        heatmap = heatmap_dict[xai_method]
        
        # for lime-heatmaps plotting works differently
        if xai_method == 'lime':
            row = 2
            col = 0
            viz = np.where(heatmap > 0, img[:,:,0], 1)

            ax[row,col].imshow(viz, cmap='gray')
            ax[row,col].axis('off')
            if not info:
                ax[row,col].set_title(f'{get_xmethod_name(xai_method)}')
            else:
                metrics_list = list(np.around(info[xai_method][metric],2) for metric in ['precision','sensitivity','false-negative-rate','false-positive-rate','specificity','accuracy','f1-score'])
                ax[row, col].set_title('prec: {} sens: {} \n fn-rate: {} fp-rate: {} \n spec: {} acc: {} \n f1: {} {}'.format(
                    metrics_list[0], metrics_list[1], metrics_list[2], metrics_list[3], metrics_list[4], metrics_list[5], metrics_list[6], get_xmethod_name('lime')
                ), size='x-small')

        else:
            seismic_methods = ['bcos', 'intgrad', 'lrp', 'shap']        # methods that generate negative and positive feature attribution

            # resize heatmap if necessary
            if not img.shape[:2] == heatmap.shape[:2]:
                #print('Heatmap:', heatmap.shape, ' Image:', img.shape)
                heatmap = cv2.resize(heatmap, (img.shape[0], img.shape[1]))
            
            # rescale to [-1, 1] for methods with negative feature attribution or [0, 1] for methods with positive fa only
            if rescale:
                heatmap = heatmap / np.max(abs(heatmap))

            # choose heatmap and colorbar scale according to fa-range
            cm = 'seismic' if xai_method in seismic_methods else 'jet'
            vm = -1 if xai_method in seismic_methods else 0
            
            row = 1 if xai_method in seismic_methods else 2
            col = idx_n if xai_method in seismic_methods else idx_p

            ax[row, col].imshow(img)
            ax[row, col].axis('off')

            ax[row, col].imshow(img[:,:,0], cmap='gray', interpolation='nearest', origin='upper')
            ax1 = ax[row, col].imshow(heatmap, cmap=cm, alpha=0.7, vmin=vm, vmax=1) if rescale else ax[row,col].imshow(heatmap, cmap=cm, alpha=0.7)

            if col == 3 and xai_method in seismic_methods:
                plt.colorbar(ax1, ticks=[-1,0,1], fraction=width_cb, pad=0.04)
            elif col == 3 and xai_method not in seismic_methods:
                plt.colorbar(ax1, ticks=[0,1], fraction=width_cb, pad=0.04)
            if xai_method in seismic_methods:
                idx_n += 1 
            else:
                idx_p += 1

            ax[row, col].axis('off')
            if not info:
                ax[row, col].set_title(f'{get_xmethod_name(xai_method)}')
            else:
                metrics_list = list(np.around(info[xai_method][metric],2) for metric in ['precision','sensitivity','false-negative-rate','false-positive-rate','specificity','accuracy','f1-score'])
                ax[row, col].set_title('prec: {} sens: {} \n fn-rate: {} fp-rate: {} \n spec: {} acc: {} \n f1: {} {}'.format(
                    metrics_list[0], metrics_list[1], metrics_list[2], metrics_list[3], metrics_list[4], metrics_list[5], metrics_list[6], get_xmethod_name(xai_method)
                ), size='x-small')

    plt.tight_layout()
    return fig

