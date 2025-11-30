import os
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import tensorflow as tf
import torch
from torch_geometric.data import Data, HeteroData
from waymo_open_dataset.protos import scenario_pb2
from tqdm import tqdm