
import argparse
import os
import inception
import torchvision.transforms as transforms
import pathlib
import torch
from PIL import Image
import numpy as np
import seaborn as sns
import pandas as pd
IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

# 코드를 실행할 때 입력 인자를 받습니다.
parser = argparse.ArgumentParser(
    description='Assessing Generative Models via Precision and Recall',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# 참고하려는 이미지가 포함되어 있는 Directory Name (필수)
parser.add_argument('--ref_dir', type=str, required=True,
                    help='directory containing reference images')
# 평가하려는 이미지가 포함되어 있는 Directory Name (필수), nargs가 +인 경우, 1개 이상의 값을 전부 받아들인다.  *인 경우, 0개 이상의 값을 전부 받아들인다.
parser.add_argument('--eval_dirs', type=str, nargs='+', required=True,
                    help='directory or directories containing images to be '
                         'evaluated')
# store_false에 인자를 적으면 해당 인자에 Flase 값이 저장된다.  적지 않으면 True값이 나옴.
parser.add_argument('--silent', dest='verbose', action='store_false',
                    help='disable logging output')

args = parser.parse_args()


# inceptionV3 모델의 pooling 계층을 이용하여 이미지의 feature를 뽑는다.
def generate_inception_embedding(imgs, device):
    return inception.embed_images_in_inception(imgs, device)


# inceptionV3을 통해서 임베딩을 한다.
def load_or_generate_inception_embedding(directory, device):
    # 디렉토리로부터 이미지를 갖고온다.
    imgs = load_images_from_dir(directory)
    embeddings = generate_inception_embedding(imgs, device=device)
    return embeddings


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


# 디렉토리로부터 이미지를 갖고온다.
def load_images_from_dir(directory, types=('png', 'jpg', 'bmp', 'gif')):
    directory = pathlib.Path(directory)
    files = sorted([file for ext in IMAGE_EXTENSIONS
                    for file in directory.glob('*.{}'.format(ext))])
    dataset = ImagePathDataset(files, transforms=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=32,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=8)
    print('dataloader', len(dataloader))
    return dataloader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

if __name__ == '__main__':
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')


    # ref 폴더 경로와 eval 폴더 경로의 절대 경로를 얻는다.
    ref_dir = os.path.abspath(args.ref_dir)
    eval_dirs = [os.path.abspath(directory) for directory in args.eval_dirs]
    eval_embeddingsLists = []

    feature_dim = 1000
    nearest_k = 5

    if args.verbose:
        print('computing inception embeddings for ' + ref_dir)
    real_embeddings = load_or_generate_inception_embedding(
        ref_dir, device)

    prd_data = []
    for directory in eval_dirs:
        if args.verbose:
            print('computing inception embeddings for ' + directory)
        eval_embeddings = load_or_generate_inception_embedding(
            directory, device)
        eval_embeddingsLists.append(eval_embeddings)
        if args.verbose:
            print('finish embedding')


    if args.verbose:
        print('plotting results')
    print(type(real_embeddings))
    print(real_embeddings.shape[0])
    print(eval_embeddings.shape[0])

    embeddings = np.concatenate((real_embeddings, eval_embeddings), axis=0)
    labels = ['real'] * real_embeddings.shape[0]
    labels = labels + ['fake'] * eval_embeddings.shape[0]

    print(embeddings.shape)
    print(len(labels))
    tsne = TSNE(n_components=2).fit_transform(embeddings)

    tsne_df = pd.DataFrame({'x': tsne[:, 0], 'y': tsne[:, 1], 'classes': labels})

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x='x', y='y',
        hue='classes',
        palette=sns.color_palette("Set1", 2),
        data=tsne_df,
        legend="full",
        alpha=0.4
    )

    plt.title("tSNE-mnist")

    plt.savefig('TSNE.png', bbox_inches='tight')
    plt.show()





