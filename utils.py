# -*- coding: utf-8 -*-

import yaml
import numpy as np
import pandas as pd
from PIL import Image
from scipy import interpolate, signal


class LossAcc(object):
    def __init__(self):
        self.loss = None
        self.acc = None
        self.cnt = None

    def push(self, loss, acc, cnt):
        self.loss = loss if self.loss is None else self.loss + loss
        self.acc = acc if self.acc is None else self.acc + acc
        self.cnt = cnt if self.cnt is None else self.cnt + cnt

    def clear(self):
        self.__init__()

    def get_acc(self):
        return self.acc / self.cnt

    def get_loss(self):
        return self.loss / self.cnt


def save_config_file(config_file, args):
    with open(config_file, 'w') as outfile:
        yaml.safe_dump(args, outfile, default_flow_style=False)


def read_data(file, mode=0):
    if mode == 1:
        df = pd.read_csv(file, index_col=None, header=None)
        t = np.array(df[0], dtype=np.double) / 1000000000.
        t -= t[0]
        x = np.array(df[1])
        y = np.array(df[2])
        z = np.array(df[3])
        return t, x, y, z
    elif mode == 0:
        df = pd.read_csv(file, index_col=None, header=0)
        sec = 'seconds_elapsed'
        x = np.array(df['x'])
        y = np.array(df['y'])
        z = np.array(df['z'])
        t = np.array(df[sec]) - df[sec][0]
        return t, x, y, z


def normalization(*args):
    def standard(x):
        m = x.mean()
        v = x.std()
        return (x - m) / v
    return [standard(x) for x in args]


def interpolation(t, *args, hz=1000):
    nt = np.linspace(0, t[-1], int(t[-1] * hz))
    return nt, *[interpolate.interp1d(t, x, kind='slinear')(nt) for x in args]


def clipping(t, *args, s, e, hz=1000):
    s = int(s * hz)
    e = int(e * hz)
    if s < 0:
        s += len(t)
        e += len(t)
    return t[0: e-s], *[x[s:e] for x in args]


def sliding_window(t, *args, hz=1000, window=1.2, stride=0.1, func=np.mean):
    window = int(window * hz)
    stride = int(stride * hz)
    st = np.arange(0, len(t)-window, stride)
    return t[st], *[np.array([func(x[i: i + window]) for i in st]) for x in args]


def non_maximum_suppression(t, *args, threshold=1.2, choose_num=5, filter_num=100):
    def nms(x: np.ndarray):
        order = x.argsort()[:: -1][0: filter_num]
        keep = []
        while len(keep) < choose_num and len(order):
            i = order[0]
            keep.append(i)
            ovr = np.abs(t[i] - t[order])
            order = order[np.where(ovr > threshold)[0]]
        return keep
    return [nms(x) for x in args]


def smooth(*args, config=200):
    if isinstance(config, int):
        config = (config, )

    def smt(x):
        for conf in config:
            x = np.convolve(x, np.ones(conf) / conf, mode='same')
        return x
    return [smt(np.abs(x)) for x in args]


def find_cutting_point(*args, para=0.5):
    def fcp(x):
        cutting_points = []
        threshold = np.min(x) * para + np.max(x) * (1 - para)
        pre = 0
        for idx, z in enumerate(x):
            if pre < threshold <= z:
                cutting_points.append(idx)
            if z <= threshold < pre:
                cutting_points.append(idx)
            pre = z
        return cutting_points
    return [fcp(x) for x in args]


def fast_ft(*args, hz=1000):
    length = len(args[0])
    nt = np.fft.fftfreq(length, d=1./hz)[:(length + 1) // 2]
    return nt, *[np.fft.fft(x)[:(length + 1) // 2] for x in args]


def highpass_filter(*args, hz=1000, s_threshold=0, e_threshold=200):
    def hp_filter(x):
        f, t, z = signal.stft(x, fs=hz, nperseg=128, noverlap=120, window='hann', boundary=None, padded=False,
                              return_onesided=True)
        for i in range(0, z.shape[0]):
            if f[i] < s_threshold or f[i] > e_threshold:
                z[i, :] = 0
        return signal.istft(z, fs=hz, nperseg=128, nfft=128, noverlap=120)
    res = [hp_filter(x) for x in args]
    return res[0][0] + 64 / hz, *[x[1] for x in res]


def stime_ft(*args, hz=1000, s_hz=0, e_hz=300):
    def stft(x):
        f, t, z = signal.stft(x, fs=hz, window='hann', nperseg=512, noverlap=508, detrend=False)
        s = 0
        e = 0
        for i in range(len(f)):
            if f[i] < s_hz:
                s = i
            if f[i] < e_hz:
                e = i
        return f[s:e + 1], t, z[s:e + 1][:]
    return [stft(x) for x in args]


def generate_image(*args):
    def calc(xx):
        xx = np.abs(xx[2])
        xx = np.power(xx, 0.5)
        min_x = np.min(xx)
        max_x = np.max(xx)
        xx = np.ceil((xx - min_x) * (255. / (max_x - min_x)))
        return xx
    x, y, z = [calc(x) for x in args]
    img = np.zeros((3, x.shape[0], x.shape[1]))
    img[0, :, :] = x[:, :]
    img[1, :, :] = y[:, :]
    img[2, :, :] = z[:, :]
    img = img.transpose((1, 2, 0))
    img = Image.fromarray(img.astype(np.uint8))
    img.save('./tmptmp.png')
    return img
    return Image.fromarray(img.astype(np.uint8))


def solve_image_PIL(file, hz):
    t, x, y, z = read_data(file, mode=1)
    t, x, y, z = interpolation(t, x, y, z, hz=hz)
    t, x, y, z = clipping(t, x, y, z, s=-1.3, e=-0.3, hz=hz)
    x, y, z = normalization(x, y, z)
    ax, ay, az = stime_ft(x, y, z, hz=hz, s_hz=0, e_hz=200)
    return generate_image(ax, ay, az)
