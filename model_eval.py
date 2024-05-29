import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from utils import get_model
from consts.consts import MosaicArgs, Split
from explainability import pipelines as ppl


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

architectures = ['resnet50', 'vgg11_bn']
datasets = [MosaicArgs.CARSNCATS_MOSAIC, MosaicArgs.MOUNTAINDOGS_MOSAIC, MosaicArgs.ILSVRC2012_MOSAIC]

transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

df = pd.DataFrame(columns=['architecture_dataset', 'top1_tc1', 'top5_tc1', 'top1_tc2', 'top5_tc2', 'top1_tc3', 'top5_tc3'])

for architecture in architectures:
    for mosaics in datasets:

        top1_tc1, top5_tc1 = 0, 0
        top1_tc2, top5_tc2 = 0, 0
        top1_tc3, top5_tc3 = 0, 0
        bcos_top1_tc1, bcos_top5_tc1 = 0, 0
        bcos_top1_tc2, bcos_top5_tc2 = 0, 0
        bcos_top1_tc3, bcos_top5_tc3 = 0, 0

        model = get_model(architecture, use_bcos=False).to(device)
        bcos_model = get_model(architecture, use_bcos=True).to(device)
        mosaic_dataset = ppl.DATASETS[mosaics].load_dataset()
        num_mosaics = len(mosaic_dataset.get_subset(Split.VAL)[0])

        for mosaic_filepath, target_class, images_filenames, image_labels in tqdm(zip(*mosaic_dataset.get_subset(Split.VAL)), total=num_mosaics):

            other_classes = list(set(image_labels)).copy()                # second class in image
            other_classes.remove(target_class)

            if str(mosaics) == 'ilsvrc2012_mosaic':
                tc2 = other_classes[0]
                tc3 = other_classes[1]
            else:
                tc2 = other_classes[0]

            img = Image.open(mosaic_filepath)
            bcos_img_tensor = bcos_model.transform(img)
            bcos_img_tensor = bcos_img_tensor[None].to(device)
            img = img.convert('RGB')
            img_tensor = transformation(img)
            img_tensor = torch.unsqueeze(img_tensor, 0).to(device)

            prediction = model(img_tensor)
            bcos_prediction = bcos_model(bcos_img_tensor)
            top5 = prediction.detach().cpu().numpy().argsort(axis=1)[:,-5:]
            bcos_top5 = bcos_prediction.detach().cpu().numpy().argsort(axis=1)[:,-5:]

            # normal
            if prediction.argmax().item() == target_class:
                top1_tc1 += 1
            if target_class in top5:
                top5_tc1 += 1

            if prediction.argmax().item() == tc2:
                top1_tc2 += 1
            if tc2 in top5:
                top5_tc2 += 1

            if str(mosaics) == 'ilsvrc2012_mosaic':
                if prediction.argmax().item() == tc3:
                    top1_tc3 += 1
                if tc3 in top5:
                    top5_tc3 += 1

            #bcos
            if bcos_prediction.argmax().item() == target_class:
                bcos_top1_tc1 += 1
            if target_class in bcos_top5:
                bcos_top5_tc1 += 1

            if bcos_prediction.argmax().item() == tc2:
                bcos_top1_tc2 += 1
            if tc2 in bcos_top5:
                bcos_top5_tc2 += 1

            if str(mosaics) == 'ilsvrc2012_mosaic':
                if bcos_prediction.argmax().item() == tc3:
                    bcos_top1_tc3 += 1
                if tc3 in bcos_top5:
                    bcos_top5_tc3 += 1

        new_row = {'architecture_dataset': f'{architecture}_{str(mosaics)}', 'top1_tc1': top1_tc1 / num_mosaics, 'top5_tc1': top5_tc1 / num_mosaics, 'top1_tc2': top1_tc2 / num_mosaics, 'top5_tc2': top5_tc2 / num_mosaics, 'top1_tc3': top1_tc3 / num_mosaics, 'top5_tc3': top5_tc3 / num_mosaics}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        new_row = {'architecture_dataset': f'bcos_{architecture}_{str(mosaics)}', 'top1_tc1': bcos_top1_tc1 / num_mosaics, 'top5_tc1': bcos_top5_tc1 / num_mosaics, 'top1_tc2': bcos_top1_tc2 / num_mosaics, 'top5_tc2': bcos_top5_tc2 / num_mosaics, 'top1_tc3': bcos_top1_tc3 / num_mosaics, 'top5_tc3': bcos_top5_tc3 / num_mosaics}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

df.to_csv('evaluation/model_accs.csv')
