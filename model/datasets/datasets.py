# YOLOv5 dataset utils and dataloaders

import glob
import hashlib
import json
import logging
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool, Pool
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from .augmentations import Albumentations, Albumentations_mine,augment_hsv, copy_paste, letterbox, mixup, random_perspective, \
    random_perspective_mine
from model.utils.general import check_requirements, check_file, check_dataset, xywh2xyxy, xywhn2xyxy, xyxy2xywhn, \
    xyn2xy, segments2boxes, clean_str
from model.utils.torch_utils import torch_distributed_zero_first

import sys

# Parameters
HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
NUM_THREADS = min(8, os.cpu_count())  # number of multiprocessing threads
# NUM_THREADS=1

# Get orientation exif tag     获取翻转的键值 0x0112 即274
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height
    try:
        rotation = dict(img._getexif().items())[orientation]
        ######是否转换角度
        if rotation == 6:  # rotation 270
            print("ori")
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            print("ori")
            s = (s[1], s[0])

    except:
        pass

    return s


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    From https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {2: Image.FLIP_LEFT_RIGHT,
                  3: Image.ROTATE_180,
                  4: Image.FLIP_TOP_BOTTOM,
                  5: Image.TRANSPOSE,
                  6: Image.ROTATE_270,
                  7: Image.TRANSVERSE,
                  8: Image.ROTATE_90,
                  }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image


def create_dataloader(path, imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, cache=False, pad=0.0,
                      rect=False, rank=-1, workers=8, image_weights=False, quad=False, prefix='', part="train",type="all",
                      repeat=None,repeat_num=0,block=False):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      cache_images=cache,
                                      single_cls=single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights,
                                      prefix=prefix,
                                      part=part,
                                      type=type,
                                      repeat=repeat,
                                      repeat_num=repeat_num,
                                      block=block)

    batch_size = min(batch_size, len(dataset))

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])  # number of workers

    # nw = 1

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()

    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn)
    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadBlockImage:
    def __init__(self,
                 path,
                 img_size=1024,
                 stride=32,
                 block_num=3,
                 factor=0.25):
        self.block_num=block_num
        self.factor=factor
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]

        ni= len(images)

        self.img_size = img_size
        self.stride = stride
        self.files = images
        self.nf = ni

        self.mode = 'image'
        assert self.nf > 0, f'No images found in {p}. ' \
            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def clip_cordinate(self,cordinates, shape):
        for sub_cor in cordinates:
            sub_cor[0] = max(0,min(sub_cor[0], shape))
            sub_cor[1] = max(0,min(sub_cor[1], shape))

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]
        # Read image
        self.count += 1
        image = cv2.imread(path)  # BGR
        assert image is not None, 'Image Not Found ' + path
        print(f'image {self.count}/{self.nf} {path}: \n', end='')

        # 获取分块的图像
        block_images_rect=[]
        block_images=[]
        block_cordinates=[]
        shape = image.shape
        row_block = shape[0] // self.block_num
        col_block = shape[1] // self.block_num
        row_cordinates = [[row * row_block, (row + 1) * row_block + int(row_block * self.factor)] for row in
                          range(self.block_num)]
        col_cordinates = [[col * col_block, (col + 1) * col_block + int(col_block * self.factor)] for col in
                          range(self.block_num)]
        self.clip_cordinate(row_cordinates, shape[0])
        self.clip_cordinate(col_cordinates, shape[1])

        for row in range(self.block_num):
            for col in range(self.block_num):
                # index = row * self.block_num + col
                row_cordinate = row_cordinates[row]
                col_cordinate = col_cordinates[col]
                block_image = image[row_cordinate[0]:row_cordinate[1], col_cordinate[0]:col_cordinate[1], :]
                block_image_rect = letterbox(block_image, self.img_size, stride=self.stride)[0]
                block_image_rect=block_image_rect.transpose((2, 0, 1))[::-1][None]
                # print(block_image_rect.shape)
                block_images_rect.append(np.ascontiguousarray(block_image_rect))
                block_images.append(block_image)
                block_cordinates.append([row_cordinate,col_cordinate])
        image_rect=letterbox(image, self.img_size, stride=self.stride)[0]
        image_rect=np.ascontiguousarray(image_rect.transpose((2, 0, 1))[::-1][None])

        return path, block_images_rect, block_images, block_cordinates,image_rect,image

    def __len__(self):
        return self.nf  # number of files


class LoadImages:  # for inference
    def __init__(self, path, img_size=1024, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print(f'image {self.count}/{self.nf} {path}: ', end='')


        # Padded resize  rectangle inference
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.pipe = eval(pipe) if pipe.isnumeric() else pipe
        self.cap = cv2.VideoCapture(self.pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        ret_val, img0 = self.cap.read()
        img0 = cv2.flip(img0, 1)  # flip left-right

        # Print
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        print(f'webcam {self.count}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640, stride=32):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            print(f'{i + 1}/{n}: {s}... ', end='')
            if 'youtube.com/' in s or 'youtu.be/' in s:  # if source is YouTube video
                check_requirements(('pafy', 'youtube_dl'))
                import pafy
                s = pafy.new(s).getbest(preftype="mp4").url  # YouTube URL
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0  # 30 FPS fallback
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f" success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap):
        # Read stream `i` frames in daemon thread
        n, f, read = 0, self.frames[i], 1  # frame number, frame array, inference every 'read' frame
        while cap.isOpened() and n < f:
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n % read == 0:
                success, im = cap.retrieve()
                self.imgs[i] = im if success else self.imgs[i] * 0
            time.sleep(1 / self.fps[i])  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img0 = self.imgs.copy()
        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix='', part="train",type="part",
                 repeat=None,repeat_num=0,block=False):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp["datasets"]
        self.image_weights = image_weights
        # self.rect = False if image_weights else rect
        self.rect=rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = os.path.join(self.hyp['path'],path)
        self.batch_size=batch_size
        self.part=part
        self.index=0
        self.type=type
        self.repeat=repeat
        self.repeat_num=repeat_num
        ##########是否分块的标志
        self.block=block

        ##################blur，togray,并将目标框转换为label标签
        if self.part=="val":
            self.aug_hsv=False
        else:
            self.aug_hsv=True
        self.albumentations = Albumentations_mine() if augment or self.aug_hsv else None

        if part == "train":
            mid_path = "images/train"
        else:
            mid_path = "images/val"

        import sys
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(os.path.join(self.hyp["path"],p))  # os-agnostic
                if p.is_dir():  # dir

                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()
                        ###############注意此处修改了路径
                        parent = str(p.parent) + os.sep + mid_path
                        f += [os.path.join(parent, os.path.splitext(x)[0] + ".jpg") for x in t]

                        # f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        ##############

                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')

            #######################修改代码，为了对图像文件排序   所以文件名必须为数字
            self.img_files = sorted(f, key=lambda x: float(os.path.splitext(x.split("/")[-1])[0][5:]))

            ##########原始代码
            # self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS])
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats])  # pathlib
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {HELP_URL}')

        # Check cache
        self.label_files = img2label_paths(self.img_files)  # labels


        ########对展厅照片(最后5张)进行重复采样得到50张   该扩增策略不行，因为cache文件会去除重复的key，即重复的文件名
        # import copy
        # samples=copy.deepcopy(self.label_files[-5:])*10
        # self.label_files.extend(samples)
        ##################
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')

        # if part=="val":
        #     print(self.img_files)
        #     sys.exit()
        # try:
        #     cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
        #     assert cache['version'] == 0.4 and cache['hash'] == get_hash(self.label_files + self.img_files), "not sdhfd"
        # except:
        #     # 不对图像数量进行清点 防止图像数量和标签数量不相同
        #     cache, exists = self.cache_labels(cache_path, prefix), False  # cache
        cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupted, total

        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
            if cache['msgs']:
                logging.info('\n'.join(cache['msgs']))  # display warnings
        # assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {HELP_URL}'
        assert nf > 0 , f'{prefix}No labels in {cache_path}. Can not train without labels. See {HELP_URL}'
        # # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items

        #####分解出label,shapes等    注意此处的label仍然为归一化的label
        labels, shapes, self.segments = zip(*cache.values())

        self.labels = list(labels)
###################################测试
        # annos = []
        # print(len(self.labels))
        # for i in range(len(self.labels)):
        #     annos.extend(self.labels[i])
        #
        # annos = np.stack(list(annos), axis=0)
        # print(annos.shape)
        # annos = annos[:, :9].astype(np.int)
        # for i in range(8):
        #     if i == 0:
        #         print(np.bincount(annos[:, i + 1]))
        #     else:
        #         print(np.bincount(annos[:, i + 1]))
        # # # sys.exit()
        # #######################测试

        self.shapes = np.array(shapes, dtype=np.float64)
        #######跟新图像列表,标签列表
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update

        if single_cls:
            for x in self.labels:
                x[:, 0] = 0


        n = len(shapes)  # number of images
        ########batch索引
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = list(range(n))

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            ########根据宽长比调整图像的顺序
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                #######获取每个batch内所有的宽长比
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]
            #######实现自适应的长宽比训练，加快训练速度
            # self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride  #####ori
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride ).astype(np.int) * stride


        ############对indices中展厅照片的索引进行重复采样
        if self.repeat_num>0 and self.repeat:
            if part == "train":
                repeat_type = self.hyp["path"]+'/labels/train/'
                with open(os.path.join(self.hyp["path"], self.repeat),"r") as f:
                    keys=[repeat_type+line.strip() for line in f.readlines()]
            else:
                repeat_type = self.hyp["path"]+'/labels/val/'
                keys=[]
            index_list=[]
            for key in keys:
                index_list.append(self.label_files.index(key))
            self.indices.extend(index_list*self.repeat_num)
        # print(np.array(self.label_files)[index_list])
        # print(index_list)
        # print(len(self.label_files))
        # import sys
        # sys.exit()
        self.n=len(self.indices)
        ############对indices进行超采样

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(NUM_THREADS).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # img, hw_original, hw_resized = load_image(self, i)
                gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB)'
            pbar.close()


    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        ######不对图像检测
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(
                pool.imap_unordered(verify_image_label_mine, zip(self.img_files, self.label_files, repeat(prefix),repeat(self.block))),
                desc=desc, total=len(self.img_files))
            for im_file, l, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f

                if im_file:
                    x[im_file] = [l, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

        pbar.close()
        if msgs:
            logging.info('\n'.join(msgs))
        if nf == 0:
            logging.info(f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, len(self.img_files)
        x['msgs'] = msgs  # warnings
        x['version'] = 0.4  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            logging.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            logging.info(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # path not writeable
        return x

    def __len__(self):
        # return len(self.img_files)
        return len(self.indices)
        # return 64

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights
        hyp = self.hyp["pipeline"]
        mosaic = self.mosaic and random.random() < hyp['mosaic']

        if mosaic:
            # Load mosaic
            img, labels = load_mosaic(self, index)
            shapes = None
            # MixUp augmentation
            if random.random() < hyp['mixup']:
                img, labels = mixup(img, labels, *load_mosaic(self, random.randint(0, self.n - 1)))

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)
            # print("shape",img.shape)
            # sys.exit()

            # Letterbox      rect train    注意此处会导致输出变化    rect改变self.shapes，但是padding依然是由letter_box生成的
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            # print(self.batch_shapes[0],self.rect,len(self.batch_shapes),self.batch_size,self.part)
            # print(self.img_size)
            # sys.exit()
            ##########去除短边多余的空白区域
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
            # print(shapes)
            # import sys
            # sys.exit()

            labels = self.labels[index].copy()

            ###########由于需要加上pad，因此转换为xyxy类型
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 0:4] = xywhn2xyxy(labels[:, 0:4], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
            # print("part",self.part)
            ##########由于我们的属性基于整个人物预测，不接受破坏人物边界框的任何物理变换
            if self.augment:
                ######仿射变换
                img, labels = random_perspective_mine(img, labels,
                                                      degrees=hyp['degrees'],
                                                      translate=hyp['translate'],
                                                      scale=hyp['scale'],
                                                      shear=hyp['shear'],
                                                      perspective=hyp['perspective'])

        #######test
        # if self.part=="train":
        #     for box in labels[:, 0:4]:
        #         box = [int(x) for x in box]
        #         cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 0))
        #     cv2.imshow("test_data", img)
            # cv2.waitKey()
        ##########test

        nl = len(labels)  # number of labels


        if nl:
            #################[cx,cy,w,h]对图像尺度的归一化   修改标签索引
            labels[:, 0:4] = xyxy2xywhn(labels[:, 0:4], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        ###########只可以使用不调整图像大小的扩增
        if self.aug_hsv:
        # if self.augment:
            # Albumentations
            # print(self.part,labels[:2],self.label_files[index])
            img, labels = self.albumentations(img, labels)

            # HSV color-space    色调，饱和度，明度变换
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down   修改索引
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    if self.block:
                        labels[:, 13] = 1 - labels[:, 13]
                    else:
                        labels[:, 11] = 1 - labels[:, 11]

            # Flip left-right    修改索引
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    if self.block:
                        labels[:, 12] = 1 - labels[:, 12]
                    else:
                        labels[:, 10] = 1 - labels[:, 10]

        if self.part=="train":
            #######all代表所有的类别使用三分类，part代表二分类
            if self.type=="all":
                labels_out = torch.zeros((nl, 32))
            else:
                if self.block:
                    labels_out = torch.zeros((nl, 20))
                else:
                    labels_out=torch.zeros((nl,18))
        else:
            if self.block:
                labels_out = torch.zeros((nl, 17))
            else:
                labels_out=torch.zeros((nl,15))

        # if self.part == "train":
        #     shape_now=img.shape
        #     for box in labels[:, 12:]:
        #         box[0]=box[0]*shape_now[1]
        #         box[1] = box[1] * shape_now[0]
        #         box[2] = box[2] * shape_now[1]
        #         box[3] = box[3] * shape_now[0]
        #
        #         box[0]=box[0]-box[2]/2
        #         box[1] = box[1] - box[3] / 2
        #         box[2] = box[0] + box[2]
        #         box[3] = box[1] + box[3]
        #         box = [int(x) for x in box]
        #         img=img.astype(np.uint8)
        #         cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 0))
        #     cv2.imshow("test_data1", img)
        #     cv2.waitKey()

        if nl:
            if self.part=="train":
                if self.type=="all":
                    ######第0列为batch内的图像序列号，具体见collect_fn函数
                    shape = labels.shape
                    labels_final = []
                    for i in range(nl):
                        label_sub = []
                        for index in range(shape[1]):
                            if index < 10 :
                                if index == 2:
                                    label_sub.extend([0, 0, 0, 0])
                                    label_sub[int(labels[i][index]) + 1] = float(1)
                                else:
                                    label_sub.extend([0, 0, 0])
                                    label_sub[int(labels[i][index]) + 5 + (index - 2) * 3] = float(1)
                            else:
                                label_sub.append(labels[i][index])
                        labels_final.append(label_sub)
                        # print(label_sub,labels[i])
                    labels_final = np.stack(labels_final, axis=0)
                    labels_out[:, 1:] = torch.from_numpy(labels_final)

                else:
                    if self.block:
                        index_thr=12
                    else:
                        index_thr=10
                    shape = labels.shape
                    labels_final = []
                    for i in range(nl):
                        label_sub = []
                        # print("ori",labels[i][:9])
                        ######[x,y,x,y,person,phone,....]
                        for index in range(shape[1]):
                            if index<index_thr:
                                if index == 2:
                                    label_sub.extend([float(0), float(0), float(0), float(0)])
                                    label_sub[int(labels[i][index])+2] = float(1)
                                else:
                                    if int(labels[i][index]) == 2:
                                        label_sub.append(float(0))
                                    else:
                                        label_sub.append(float(labels[i][index]))
                            else:
                                label_sub.append(float(labels[i][index]))
                        # print(label_sub)
                        # print(labels[i])
                        labels_final.append(label_sub)
                    # import sys
                    # sys.exit()
                    labels_final = np.stack(labels_final, axis=0)
                    labels_out[:, 1:] = torch.from_numpy(labels_final)


            else:
                shape = labels.shape
                labels_final = []
                if self.block:
                    index_thr=12
                else:
                    index_thr=10
                for i in range(nl):
                    label_sub = []
                    for index in range(shape[1]):
                        if index<index_thr:
                            if index == 2:
                                label_sub.extend([labels[i][index]])
                            else:
                                if int(labels[i][index]) == 2:
                                    label_sub.append(float(0))
                                else:
                                    label_sub.append(float(labels[i][index]))
                        else:
                            label_sub.append(float(labels[i][index]))
                    labels_final.append(label_sub)
                labels_final = np.stack(labels_final, axis=0)
                labels_out[:, 1:] = torch.from_numpy(labels_final)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB   通道转换
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        ########对图像和标签进行堆叠（因为datasets返回不可重叠的path，所以要指定堆叠哪些数据）
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0., 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0., 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, .5, .5, .5, .5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        ####缩放尺度
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                             interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)

        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized

    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


def load_mosaic(self, index):
    # loads images in a 4-mosaic

    labels4, segments4 = [], []
    s = self.img_size
    # print(self.img_size)
    #
    # print("border",self.mosaic_border,s)


    #######确保图像拼接的中点在一定范围内
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    # print("mosaic center",yc,xc)


    #######获取4张图片的索引
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)
        # print(i,img.shape,h,w)

        # place img in img4
        if i == 0:  # top left
            #######默认区域为114像素值
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            ######大图上的坐标
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            ######本身子图上的坐标
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        #####根据坐标对应关系进行填充
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        #####计算pad大小,实现小图坐标转换为大图的坐标
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()

        if labels.size:
            ###########将目标框坐标调整到大图中,并转换为角点坐标模式
            labels[:, 0:4] = xywhn2xyxy(labels[:, 0:4], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]

        labels4.append(labels)
        segments4.extend(segments)

    # 合并标签
    labels4 = np.concatenate(labels4, 0)
    ######将label进行剪裁，剪掉超出大图的部分
    for x in (labels4[:, 0:4], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
    img4, labels4 = random_perspective(img4, labels4, segments4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img4, labels4


def load_mosaic9(self, index):
    # loads images in a 9-mosaic

    labels9, segments9 = [], []
    s = self.img_size
    indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img9
        if i == 0:  # center
            img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            h0, w0 = h, w
            c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
        elif i == 1:  # top
            c = s, s - h, s + w, s
        elif i == 2:  # top right
            c = s + wp, s - h, s + wp + w, s
        elif i == 3:  # right
            c = s + w0, s, s + w0 + w, s + h
        elif i == 4:  # bottom right
            c = s + w0, s + hp, s + w0 + w, s + hp + h
        elif i == 5:  # bottom
            c = s + w0 - w, s + h0, s + w0, s + h0 + h
        elif i == 6:  # bottom left
            c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
        elif i == 7:  # left
            c = s - w, s + h0 - h, s, s + h0
        elif i == 8:  # top left
            c = s - w, s + h0 - hp - h, s, s + h0 - hp

        padx, pady = c[:2]
        x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
        labels9.append(labels)
        segments9.extend(segments)

        # Image
        img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
        hp, wp = h, w  # height, width previous

    # Offset
    yc, xc = [int(random.uniform(0, s)) for _ in self.mosaic_border]  # mosaic center x, y
    img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

    # Concat/clip labels
    labels9 = np.concatenate(labels9, 0)
    labels9[:, [1, 3]] -= xc
    labels9[:, [2, 4]] -= yc
    c = np.array([xc, yc])  # centers
    segments9 = [x - c for x in segments9]

    for x in (labels9[:, 1:], *segments9):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img9, labels9 = replicate(img9, labels9)  # replicate

    # Augment
    img9, labels9 = random_perspective(img9, labels9, segments9,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img9, labels9


def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def flatten_recursive(path='../datasets/coco128'):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(path + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path='../datasets/coco128'):  # from utils.datasets import *; extract_boxes()
    # Convert detection dataset into classification dataset, with one directory per class
    path = Path(path)  # images dir
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in IMG_FORMATS:
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file, 'r') as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


def autosplit(path='../datasets/coco128/images', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.datasets import *; autosplit()
    Arguments
        path:            Path to images directory
        weights:         Train, val, test weights (list, tuple)
        annotated_only:  Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    files = sum([list(path.rglob(f"*.{img_ext}")) for img_ext in IMG_FORMATS], [])  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
    [(path.parent / x).unlink(missing_ok=True) for x in txt]  # remove existing

    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], 'a') as f:
                f.write('./' + img.relative_to(path.parent).as_posix() + '\n')  # add image to txt file


def verify_image_label_mine(args):
    # Verify one image-label pair
    im_file, lb_file, prefix, block= args
    if block:
        label_len=16
    else:
        label_len=14

    nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, corrupt
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size   获取图像的尺度，以及旋转方向
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                assert f.read() == b'\xff\xd9', 'corrupted JPEG'  ######判断文件是否损坏

        # verify labels
        segments = []  # instance segments
        if os.path.isfile(lb_file):
            nf = 1  # 找到标签文件
            with open(lb_file, 'r') as f:
                ######注意此处label:[x,y,w,h]对图像尺度作归一化
                l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                l = np.array(l, dtype=np.float32)
            if len(l):
                assert l.shape[1] == label_len
                assert (l >= 0).all(), 'negative labels'
                assert (l[:, 0:4] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'  ####判断是否有重复的一行标签
            else:
                ne = 1  # label empty
                l = np.zeros((0, label_len), dtype=np.float32)
                print("empty:" ,lb_file)

        else:
            nm = 1  # label missing
            l = np.zeros((0, label_len), dtype=np.float32)
        return im_file, l, shape, segments, nm, nf, ne, nc, ''
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]


def verify_image_label(args):
    # Verify one image-label pair
    im_file, lb_file, prefix = args
    nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, corrupt
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size   获取图像的尺度，以及旋转方向
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                assert f.read() == b'\xff\xd9', 'corrupted JPEG'  ######判断文件是否损坏

        # verify labels
        segments = []  # instance segments
        if os.path.isfile(lb_file):
            nf = 1  # 找到标签文件
            with open(lb_file, 'r') as f:
                ######注意此处label:[x,y,w,h]对图像尺度作归一化
                l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any([len(x) > 8 for x in l]):  # is segment   判断是否是分割标签
                    print("segmentation")
                    classes = np.array([x[0] for x in l], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                    l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                l = np.array(l, dtype=np.float32)
            if len(l):
                assert l.shape[1] == 5, 'labels require 5 columns each'
                assert (l >= 0).all(), 'negative labels'
                assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'  ####判断是否有重复的一行标签
            else:
                ne = 1  # label empty
                l = np.zeros((0, 5), dtype=np.float32)

        else:
            nm = 1  # label missing
            l = np.zeros((0, 5), dtype=np.float32)
        return im_file, l, shape, segments, nm, nf, ne, nc, ''
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]


def dataset_stats(path='coco128.yaml', autodownload=False, verbose=False):
    """ Return dataset statistics dictionary with images and instances counts per split per class
    Usage1: from utils.datasets import *; dataset_stats('coco128.yaml', verbose=True)
    Usage2: from utils.datasets import *; dataset_stats('../datasets/coco128.zip', verbose=True)

    Arguments
        path:           Path to data.yaml or data.zip (with data.yaml inside data.zip)
        autodownload:   Attempt to download dataset if not found locally
        verbose:        Print stats dictionary
    """

    def round_labels(labels):
        # Update labels to integer class and 6 decimal place floats
        return [[int(c), *[round(x, 6) for x in points]] for c, *points in labels]

    def unzip(path):
        # Unzip data.zip TODO: CONSTRAINT: path/to/abc.zip MUST unzip to 'path/to/abc/'
        if str(path).endswith('.zip'):  # path is data.zip
            assert os.system(f'unzip -q {path} -d {path.parent}') == 0, f'Error unzipping {path}'
            data_dir = path.with_suffix('')  # dataset directory
            return True, data_dir, list(data_dir.rglob('*.yaml'))[0]  # zipped, data_dir, yaml_path
        else:  # path is data.yaml
            return False, None, path

    zipped, data_dir, yaml_path = unzip(Path(path))
    with open(check_file(yaml_path)) as f:
        data = yaml.safe_load(f)  # data dict
        if zipped:
            data['path'] = data_dir  # TODO: should this be dir.resolve()?
    check_dataset(data, autodownload)  # download dataset if missing
    nc = data['nc']  # number of classes
    stats = {'nc': nc, 'names': data['names']}  # statistics dictionary
    for split in 'train', 'val', 'test':
        if data.get(split) is None:
            stats[split] = None  # i.e. no test set
            continue
        x = []
        dataset = LoadImagesAndLabels(data[split], augment=False, rect=True)  # load dataset
        if split == 'train':
            cache_path = Path(dataset.label_files[0]).parent.with_suffix('.cache')  # *.cache path
        for label in tqdm(dataset.labels, total=dataset.n, desc='Statistics'):
            x.append(np.bincount(label[:, 0].astype(int), minlength=nc))
        x = np.array(x)  # shape(128x80)
        stats[split] = {'instance_stats': {'total': int(x.sum()), 'per_class': x.sum(0).tolist()},
                        'image_stats': {'total': dataset.n, 'unlabelled': int(np.all(x == 0, 1).sum()),
                                        'per_class': (x > 0).sum(0).tolist()},
                        'labels': [{str(Path(k).name): round_labels(v.tolist())} for k, v in
                                   zip(dataset.img_files, dataset.labels)]}

    # Save, print and return
    with open(cache_path.with_suffix('.json'), 'w') as f:
        json.dump(stats, f)  # save stats *.json
    if verbose:
        print(json.dumps(stats, indent=2, sort_keys=False))
        # print(yaml.dump([stats], sort_keys=False, default_flow_style=False))
    return stats
