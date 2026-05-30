import numpy as np
from helpers.util import visSph
path = './datasets/'
seg_list = [True, True, True]  # [True, False, False]

data = np.load(path+'visCubic.npz', allow_pickle=True)
y, yols, yridge = data['ytest'].tolist(), data['yols'].tolist(), data['yriddge'].tolist()

ypoly, ypolyols, ypolyridge = np.array(y), np.array(yols)[0], np.array(yridge)[0]
visSph([y, yols, yridge], ['b', 'r', 'g'], segment_list=seg_list)

data = np.load(path+'visSin.npz', allow_pickle=True)
y, yols, yridge = data['ytest'].tolist(), data['yols'].tolist(), data['yriddge'].tolist()
yelse, yelseols, yelseridge = np.array(y), np.array(yols)[0], np.array(yridge)[0]

visSph([y, yols, yridge], ['b', 'r', 'g'], seg_list)