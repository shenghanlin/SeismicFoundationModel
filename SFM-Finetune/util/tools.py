'''
Author: Jintao Li
Date: 2022-05-30 16:42:14
LastEditors: Jintao Li
LastEditTime: 2022-07-11 23:05:53
2022 by CIG.
'''

import os, shutil
import yaml, argparse
from sklearn.metrics import confusion_matrix
import numpy as np
import torch


def accuracy(output, target):
    '''
    output: [N, num_classes, ...], torch.float
    target: [N, ...], torch.int
    '''
    output = output.argmax(dim=1).flatten().detach().cpu().numpy()
    target = target.flatten().detach().cpu().numpy()
    return _pxiel_acc(output, target), _miou(output, target)


def _pxiel_acc(output, target):
    r"""
    计算像素准确率 (Pixel Accuracy, PA)
    $$ PA = \frac{\sum_{i=0}^k p_{ii}}
    {\sum_{i=0}^k \sum_{j=0}^k p_{ij}} $$ and
    $n_class = k+1$ 
    Parameters:
    -----------
        shape: [N, ], (use flatten() function)
    return:
    ----------
        - PA
    """
    assert output.shape == target.shape, "shapes must be same"
    cm = confusion_matrix(target, output)
    return np.diag(cm).sum() / cm.sum()


def _miou(output, target):
    r"""
    计算均值交并比 MIoU (Mean Intersection over Union)
    $$ MIoU = \frac{1}{k+1} \sum_{i=0}^k \frac{p_{ii}}
    {\sum_{j=0}^k p_{ij} + \sum_{j=0}^k p_{ji} - p_{ii}} $$
    Parameters:
        output, target: [N, ]
    return:
        MIoU
    """
    assert output.shape == target.shape, "shapes must be same"
    cm = confusion_matrix(target, output)
    intersection = np.diag(cm)
    union = np.sum(cm, 1) + np.sum(cm, 0) - np.diag(cm)
    iou = intersection / union
    miou = np.nanmean(iou)

    return miou


def yaml_config_hook(config_file: str) -> argparse.Namespace:
    """ 
    加载yaml文件里面的参数配置, 并生成argparse形式的参数集合
    """
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir,
                              cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    parser = argparse.ArgumentParser()
    for k, v in cfg.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    return args


def backup_code(work_dir, back_dir, exceptions=[], include=[]):
    r"""
    备份本次运行的代码到指定目录下, 并排除某些文件和目录

    Args:
        work_dir: 工作目录, i.e. 需要备份的代码
        back_dir: 目标目录.备份代码放置的目录
        exception (list): 被排除的目录和以指定后缀结尾的文件, 默认的有
                ["__pycache__", ".pyc", ".dat", "backup", ".vscode"]
        include (list): 某些必须被备份的文件,该文件可能在exception里面
    """
    _exp = [
        "*__pycache__*", "*.pyc", "*.dat", "backup", ".vscode", "*.log",
        "*log*"
    ]
    exceptions = exceptions + _exp

    # if not os.path.exists(back_dir):
    os.makedirs(back_dir, exist_ok=True)

    shutil.copytree(work_dir,
                    back_dir + 'code/',
                    ignore=shutil.ignore_patterns(*exceptions),
                    dirs_exist_ok=True)

    for f in include:
        shutil.copyfile(os.path.join(work_dir, f),
                        os.path.join(back_dir + 'code', f))


def list_files(path, full=False):
    r"""
    递归列出目录下所有的文件，包括子目录下的文件
    """
    out = []
    for f in os.listdir(path):
        fname = os.path.join(path, f)
        if os.path.isdir(fname):
            fname = list_files(fname)
            out += [os.path.join(f, i) for i in fname]
        else:
            out.append(f)
    if full:
        out = [os.path.join(path, i) for i in out]
    return out


if __name__ == "__main__":
    output = torch.randn(4, 2, 6, 6)
    target = torch.randn(4, 2, 6, 6)
    # output = output.cuda()
    # target = target.cuda()
    target = target.argmax(1)

    accuracy(output, target)