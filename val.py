"""Evaluate fixed checkpoints at multiple thresholds on a common reference grid."""
import argparse
import json
import os
from pathlib import Path
import hashlib
import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from albumentations import Compose, Resize, Normalize
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import archs
from comparison_utils import ReferenceDataset, load_split, probability_on_reference, scores_from_probability


def main():
    os.chdir(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--name')
    group.add_argument('--names', nargs='+')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.3, 0.5, 0.7])
    parser.add_argument('--eval_size', type=int, default=256)
    parser.add_argument('--out', default='outputs/threshold_resolution_comparison')
    args = parser.parse_args()
    names = args.names or [args.name]
    if len(set(names)) != len(names) or len(set(args.thresholds)) != len(args.thresholds):
        raise ValueError('Repeated models or thresholds')
    if any(not 0 < t < 1 for t in args.thresholds):
        raise ValueError('Thresholds must be between 0 and 1')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    records, panel_predictions, metadata = [], {}, []
    canonical_split, dataset_name = None, None
    panels = []
    for name in names:
        root = Path('models') / name
        config = yaml.safe_load((root / 'config.yml').read_text(encoding='utf-8'))
        if config['num_classes'] != 1 or config['deep_supervision']:
            raise ValueError('Only binary models without deep supervision supported')
        split = load_split(root / 'split.json', config)
        if canonical_split is None:
            canonical_split, dataset_name = split, config['dataset']
            panels = sorted(split['val'])[:6]  # Deterministic, not selected by performance.
            panel_config = config
        elif split != canonical_split or config['dataset'] != dataset_name:
            raise ValueError('All models must use identical ordered train/validation IDs and dataset')
        ds = ReferenceDataset(img_ids=split['val'], img_dir=f'inputs/{config["dataset"]}/images',
            mask_dir=f'inputs/{config["dataset"]}/masks', img_ext=config['img_ext'],
            mask_ext=config['mask_ext'], num_classes=1, eval_size=args.eval_size,
            transform=Compose([Resize(config['input_h'], config['input_w']), Normalize()]))
        loader = torch.utils.data.DataLoader(ds, batch_size=config['batch_size'], shuffle=False, num_workers=0)
        model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision']).to(device)
        checkpoint = root / 'model.pth'
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        metadata.append({'model':name, 'config':config, 'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
        with torch.no_grad():
            for images, _, meta in tqdm(loader, desc=name):
                reference = meta['reference_mask']
                prob = probability_on_reference(model(images.to(device)), reference)
                for threshold in args.thresholds:
                    scores = scores_from_probability(prob, reference, threshold)
                    dest = out / name / f'threshold_{threshold:g}'
                    dest.mkdir(parents=True, exist_ok=True)
                    for i, image_id in enumerate(meta['img_id']):
                        row = {'model':name, 'input_h':config['input_h'], 'input_w':config['input_w'],
                               'eval_size':args.eval_size, 'threshold':threshold, 'img_id':image_id}
                        row.update({k:float(v[i]) for k,v in scores.items()})
                        records.append(row)
                        mask = (prob[i,0] >= threshold).astype(np.uint8)*255
                        if not cv2.imwrite(str(dest / (image_id+'.png')), mask):
                            raise OSError('Failed to save prediction')
                        if image_id in panels:
                            panel_predictions[name, threshold, image_id] = mask
    detail = pd.DataFrame(records)
    keys = ['model','input_h','input_w','eval_size','threshold']
    summary = detail.groupby(keys, sort=False)[['iou','dice','precision','recall']].mean().reset_index()
    summary['n_images'] = detail.groupby(keys, sort=False).size().to_numpy()
    baseline = summary[(summary.input_h==256)&(summary.input_w==256)&(summary.threshold==0.5)]
    if len(baseline)==1:
        for metric in ['iou','dice','precision','recall']:
            summary[f'delta_{metric}_pp_vs_256_t05'] = 100*(summary[metric]-baseline.iloc[0][metric])
    detail.to_csv(out/'per_image_metrics.csv',index=False,encoding='utf-8-sig')
    summary.to_csv(out/'summary.csv',index=False,encoding='utf-8-sig')
    # Paired differences against 0.5 within each resolution.
    paired = detail.merge(detail[detail.threshold==0.5][['model','img_id','iou','dice']],
                          on=['model','img_id'],suffixes=('','_t05'))
    for metric in ['iou','dice']:
        paired[f'delta_{metric}_vs_t05'] = paired[metric]-paired[f'{metric}_t05']
    paired.to_csv(out/'paired_threshold_differences.csv',index=False,encoding='utf-8-sig')
    manifest={'models':metadata, 'eval_size':args.eval_size, 'thresholds':args.thresholds,
              'split':canonical_split, 'panel_ids':panels,
              'preprocessing':'Unchanged uploaded Dataset: OpenCV BGR, Normalize, then /255.',
              'evaluation':'Sigmoid then bilinear probability resize (align_corners=False); nearest GT resize; >= threshold.',
              'metric_policy':'Per-image macro means. IoU/Dice smoothing 1e-5; zero denominator precision/recall = 0.',
              'scope':'Validation-set sensitivity analysis; not independent test performance.'}
    (out/'evaluation_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    columns=[(name,t) for name in names for t in args.thresholds]
    for image_id in panels:
        data=Path('inputs')/dataset_name
        raw=cv2.imread(str(data/'images'/(image_id+panel_config['img_ext'])))
        raw=cv2.cvtColor(cv2.resize(raw,(args.eval_size,args.eval_size)),cv2.COLOR_BGR2RGB)
        gt=cv2.imread(str(data/'masks'/'0'/(image_id+panel_config['mask_ext'])),0)
        gt=cv2.resize(gt,(args.eval_size,args.eval_size),interpolation=cv2.INTER_NEAREST)
        fig,axs=plt.subplots(1,2+len(columns),figsize=(2.5*(2+len(columns)),3.2))
        axs[0].imshow(raw); axs[0].set_title('Image')
        axs[1].imshow(gt,cmap='gray',vmin=0,vmax=255); axs[1].set_title('Ground truth')
        for ax,(name,t) in zip(axs[2:],columns):
            row=detail[(detail.model==name)&(detail.threshold==t)&(detail.img_id==image_id)].iloc[0]
            ax.imshow(panel_predictions[name,t,image_id],cmap='gray',vmin=0,vmax=255)
            ax.set_title(f'{int(row.input_w)}px | t={t:g}\nDice={row.dice:.3f}')
        for ax in axs: ax.axis('off')
        fig.suptitle(image_id); fig.tight_layout(); fig.savefig(out/f'comparison_{image_id}.png',dpi=160); plt.close(fig)
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print('Results:',out.resolve())


if __name__=='__main__':
    main()
