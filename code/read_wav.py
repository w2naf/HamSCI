#!/usr/bin/env python

import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np 
import os
import sys
import soundfile as sf

#import datetime

filename='B2_20151128_003000_7091kHz.wav'

rate, data=sf
