"""
It contains the most basic project settings at one place to change...
"""
import os
import sys
from pathlib import Path

# basic path information
os.environ['project_root'] = Path(__file__).parent.parent.as_posix()
sys.path.append(os.environ['project_root'])

# CRS = "EPSG:3857"  # This is Mercator system that can work for most places globally
CRS = "EPSG:2240"  # this is for Georgia West

