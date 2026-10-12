import os
import json
import urllib.request
import base64
import subprocess
import sys

DEST_DIR = "/home/sahith/Projects/Crop/data/external/arecanut_kaggle_tejpaviraj"
os.makedirs(DEST_DIR, exist_ok=True)

with open('/home/sahith/.kaggle/kaggle.json') as f:
    creds = json.load(f)

u = creds['username']
k = creds['key']
auth = base64.b64encode(f'{u}:{k}'.encode()).decode()

print("Fetching signed download URL from Kaggle API...")
req = urllib.request.Request(
    'https://www.kaggle.com/api/v1/datasets/download/tejpaviraj/arecanut',
    headers={'Authorization': f'Basic {auth}'}
)

with urllib.request.urlopen(req) as resp:
    url = resp.geturl()

print("Launching aria2c to download archive.zip...")
cmd = [
    'aria2c', '-c', '-x', '16', '-s', '16', '-k', '1M',
    f'--dir={DEST_DIR}',
    '-o', 'archive.zip',
    '--summary-interval=10',
    url
]

subprocess.run(cmd, check=True)
print("Download complete!")
