# CryoTrack LSTM-AE Only Production Version
# (No Isolation Forest/Ensemble, Full Production GUI/Reporting/Analysis/Management)
# Author: SeJun Kang
# =============================================

import os
import sys
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.preprocessing import RobustScaler
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
import pickle
from threading import Thread, Event
import time
from scipy import signal

# ... (이하 전체 코드 - 이전 code block들에서 작성한 내용을 모두 이어붙여 포함)