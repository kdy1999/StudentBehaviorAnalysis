# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections.abc import Sequence
import cv2
import numpy as np
from .builder import DataContainer as DC
from .builder import build_from_cfg
from torchvision.transforms import functional as F
from .builder import PIPELINES

@PIPELINES.register_module()
class Resize:
    def __init__(self,
                 size=None):
        self.size=size
    def __call__(self, result):
        img=result["img"]
        h0,w0,_=img.shape
        ratio=min(self.size[0]/w0,self.size[1]/h0)
        h,w=int(ratio*h0),int(ratio*w0)
        pad_h,pad_w=self.size[1]-h,self.size[0]-w
        pad_h=pad_h/2
        pad_w=pad_w/2
        import cv2
        img=cv2.resize(img,dsize=(w,h),interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))

        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img=cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=(0,0,0))
        result["img"]=img
        #########用于最后的关键点映射回原图
        result["shape"]=[w,h]
        result["pad"]=[left,top]
        # cv2.imshow("resize",img)
        # cv2.waitKey()
        return result

@PIPELINES.register_module()
class ToTensor:
    def __call__(self, results):
        results['img'] = F.to_tensor(results['img'])
        return results


@PIPELINES.register_module()
class NormalizeTensor:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, results):
        results['img'] = F.normalize(
            results['img'], mean=self.mean, std=self.std)
        return results


@PIPELINES.register_module()
class Compose:
    def __init__(self, transforms):
        assert isinstance(transforms, Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict, but got'
                                f' {type(transform)}')

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        """Compute the string representation."""
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += f'\n    {t}'
        format_string += '\n)'
        return format_string


@PIPELINES.register_module()
class Collect:
    def __init__(self, keys, meta_keys, meta_name='img_metas'):
        self.keys = keys
        self.meta_keys = meta_keys
        self.meta_name = meta_name

    def __call__(self, results):
        if 'ann_info' in results:
            results.update(results['ann_info'])

        data = {}
        for key in self.keys:
            if isinstance(key, tuple):
                assert len(key) == 2
                key_src, key_tgt = key[:2]
            else:
                key_src = key_tgt = key
            data[key_tgt] = results[key_src]

        meta = {}
        if len(self.meta_keys) != 0:
            for key in self.meta_keys:
                if isinstance(key, tuple):
                    assert len(key) == 2
                    key_src, key_tgt = key[:2]
                else:
                    key_src = key_tgt = key
                meta[key_tgt] = results[key_src]
        if 'bbox_id' in results:
            meta['bbox_id'] = results['bbox_id']
        data[self.meta_name] = DC(meta, cpu_only=True)

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'keys={self.keys}, meta_keys={self.meta_keys})')
