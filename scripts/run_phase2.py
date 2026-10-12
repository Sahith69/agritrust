import sys; sys.path.insert(0, '/home/sahith/Projects/Crop'); import config
from src.split_data import split_arecanut, split_paddy

if __name__ == "__main__":
    print("Running Phase 2...")
    split_arecanut()
    split_paddy()
