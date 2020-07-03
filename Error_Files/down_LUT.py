import pathlib
import argparse
import requests
from tqdm import tqdm

import models as models
from data import valid_datasets as dataset_names


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 1024 * 32

    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE), desc=destination, miniters=0, unit='MB', unit_scale=1/32, unit_divisor=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch TestBench for Weight Prediction')
    parser.add_argument('-e', '--err', type=float, metavar='ERR', default='0.1',
                        help='Error Rate')
    parser.add_argument('-a', '--all', dest='resume', action='store_true',
                        help='Resume model?')
    parser.add_argument('-o', '--out', type=str, default='pretrained_model.pth',
                        help='output filename of pretrained model from our google drive')
    args = parser.parse_args()

    # 'file_id' is 
    if args.all:
        file_id = '1lFkYlZ_8uL3MCzbg2YJvMSJoOlRBK3Go'

    ckpt_dir = pathlib.Path('checkpoint')
    dir_path = ckpt_dir / args.arch / args.dataset
    dir_path.mkdir(parents=True, exist_ok=True)
    destination = dir_path / args.out

    download_file_from_google_drive(file_id, destination.as_posix())
