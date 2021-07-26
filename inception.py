# Inception.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch

from inceptionModel import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x


def embed_images_in_inception(dataloader, device, batch_size=32):
    dims = 2048
    # 이미지를 담을 input_tensor를 선언한다.

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    embeddings = []
    start_idx = 0

    print(len(dataloader))

    for batch in tqdm(dataloader):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)[0]
            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.cpu().numpy()
        b, f, a, d = pred.shape
        pred = pred.reshape(b, f)
        
        embeddings.append(pred)
    return np.concatenate(embeddings, axis=0)







