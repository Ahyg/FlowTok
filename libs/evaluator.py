from typing import Any, List, Sequence, Optional, Mapping, Text
from scipy import linalg
import numpy as np
import gc

import torch
import torch.nn.functional as F

from torch_fidelity.feature_extractor_base import FeatureExtractorBase
from torch_fidelity.helpers import vassert
from torch_fidelity.feature_extractor_inceptionv3 import BasicConv2d, InceptionA, InceptionB, InceptionC, InceptionD, InceptionE_1, InceptionE_2
from torch_fidelity.interpolate_compat_tensorflow import interpolate_bilinear_2d_like_tensorflow1x
from torch.utils.model_zoo import load_url as load_state_dict_from_url


# Note: Compared shasum and models should be the same.
FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'

class FeatureExtractorInceptionV3(FeatureExtractorBase):
    INPUT_IMAGE_SIZE = 299

    def __init__(
            self,
            name,
            features_list,
            **kwargs,
    ):
        """
        InceptionV3 feature extractor for 2D RGB 24bit images.

        Args:

            name (str): Unique name of the feature extractor, must be the same as used in
                :func:`register_feature_extractor`.

            features_list (list): A list of the requested feature names, which will be produced for each input. This
                feature extractor provides the following features:

                - '64'
                - '192'
                - '768'
                - '2048'
                - 'logits_unbiased'
                - 'logits'

        """
        super(FeatureExtractorInceptionV3, self).__init__(name, features_list)
        self.feature_extractor_internal_dtype = torch.float64

        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.MaxPool_1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.MaxPool_2 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE_1(1280)
        self.Mixed_7c = InceptionE_2(2048)
        self.AvgPool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = torch.nn.Linear(2048, 1008)

        # state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
        state_dict = torch.load("/home/tiger/.cache/torch/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth", map_location="cpu", weights_only=True)
        #state_dict = torch.load(FID_WEIGHTS_URL, map_location='cpu')
        self.load_state_dict(state_dict)

        self.to(self.feature_extractor_internal_dtype)
        self.requires_grad_(False)
        self.eval()

    def forward(self, x):
        vassert(torch.is_tensor(x) and x.dtype == torch.uint8, 'Expecting image as torch.Tensor with dtype=torch.uint8')
        vassert(x.dim() == 4 and x.shape[1] == 3, f'Input is not Bx3xHxW: {x.shape}')
        features = {}
        remaining_features = self.features_list.copy()

        x = x.to(self.feature_extractor_internal_dtype)
        # N x 3 x ? x ?

        x = interpolate_bilinear_2d_like_tensorflow1x(
            x,
            size=(self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE),
            align_corners=False,
        )
        # N x 3 x 299 x 299

        # x = (x - 128) * torch.tensor(0.0078125, dtype=torch.float32, device=x.device)  # really happening in graph
        x = (x - 128) / 128  # but this gives bit-exact output _of this step_ too
        # N x 3 x 299 x 299

        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.MaxPool_1(x)
        # N x 64 x 73 x 73

        if '64' in remaining_features:
            features['64'] = F.adaptive_avg_pool2d(x, output_size=(1, 1)).squeeze(-1).squeeze(-1)
            remaining_features.remove('64')
            if len(remaining_features) == 0:
                return features

        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.MaxPool_2(x)
        # N x 192 x 35 x 35

        if '192' in remaining_features:
            features['192'] = F.adaptive_avg_pool2d(x, output_size=(1, 1)).squeeze(-1).squeeze(-1)
            remaining_features.remove('192')
            if len(remaining_features) == 0:
                return features

        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17

        if '768' in remaining_features:
            features['768'] = F.adaptive_avg_pool2d(x, output_size=(1, 1)).squeeze(-1).squeeze(-1).to(torch.float32)
            remaining_features.remove('768')
            if len(remaining_features) == 0:
                return features

        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        x = self.AvgPool(x)
        # N x 2048 x 1 x 1

        x = torch.flatten(x, 1)
        # N x 2048

        if '2048' in remaining_features:
            features['2048'] = x
            remaining_features.remove('2048')
            if len(remaining_features) == 0:
                return features

        if 'logits_unbiased' in remaining_features:
            x = x.mm(self.fc.weight.T)
            # N x 1008 (num_classes)
            features['logits_unbiased'] = x
            remaining_features.remove('logits_unbiased')
            if len(remaining_features) == 0:
                return features

            x = x + self.fc.bias.unsqueeze(0)
        else:
            x = self.fc(x)
            # N x 1008 (num_classes)

        features['logits'] = x
        return features

    @staticmethod
    def get_provided_features_list():
        return '64', '192', '768', '2048', 'logits_unbiased', 'logits'

    @staticmethod
    def get_default_feature_layer_for_metric(metric):
        return {
            'isc': 'logits_unbiased',
            'fid': '2048',
            'kid': '2048',
            'prc': '2048',
        }[metric]

    @staticmethod
    def can_be_compiled():
        return True

    @staticmethod
    def get_dummy_input_for_compile():
        return (torch.rand([1, 3, 4, 4]) * 255).to(torch.uint8)

def get_inception_model():
    model = FeatureExtractorInceptionV3("inception_model", ["2048", "logits_unbiased"])
    return model


def get_covariance(sigma: torch.Tensor, total: torch.Tensor, num_examples: int) -> torch.Tensor:
    """Computes covariance of the input tensor.

    Args:
        sigma: A torch.Tensor, sum of outer products of input features.
        total: A torch.Tensor, sum of all input features.
        num_examples: An integer, number of examples in the input tensor.
    Returns:
        A torch.Tensor, covariance of the input tensor.
    """
    if num_examples == 0:
        return torch.zeros_like(sigma)

    sub_matrix = torch.outer(total, total)
    sub_matrix = sub_matrix / num_examples

    return (sigma - sub_matrix) / (num_examples - 1)
    
class TextCondBertEvaluator:
    def __init__(
        self,
        device,
        enable_fid: bool = True,
        stat_path: str = "",
    ):
        self._device = device

        self._enable_fid = enable_fid

        # Variables related to Inception score and rFID.
        self._fid_num_features = 2048
        self._is_num_features = 1008
        self._inception_model = get_inception_model().to(self._device)
        self._inception_model.eval()

        fid_stat = torch.load(stat_path, map_location="cpu")
        self.mu_real = fid_stat['mu']
        self.sigma_real = fid_stat['sigma']      
        self._is_eps = 1e-16
        self._fid_eps = 1e-6
            
        self.reset_metrics()

    def reset_metrics(self):
        """Resets all metrics."""
        self._num_examples = 0
        self._num_updates = 0

        self._fid_fake_sigma = torch.zeros(
            (self._fid_num_features, self._fid_num_features),
            dtype=torch.float64, device=self._device
        )
        self._fid_fake_total = torch.zeros(
            self._fid_num_features, dtype=torch.float64, device=self._device
        )

    def update(
        self,
        fake_images: torch.Tensor,
        text_conditions: List[str],
        codebook_indices: Optional[torch.Tensor] = None
    ):
        batch_size = fake_images.shape[0]
        dim = tuple(range(1, fake_images.ndim))
        self._num_examples += batch_size
        self._num_updates += 1

        unscaled_image = (fake_images * 255).to(torch.uint8)

        if self._enable_fid:
            # Quantize to uint8 as a real image.
            features_fake = self._inception_model(unscaled_image)
            for f_fake in features_fake['2048']:
                self._fid_fake_total += f_fake
                self._fid_fake_sigma += torch.outer(f_fake, f_fake)
            del features_fake

        del unscaled_image
        gc.collect()

    def result(self) -> Mapping[Text, torch.Tensor]:
        """Returns the evaluation result."""
        eval_score = {}

        if self._num_examples < 1:
            raise ValueError("No examples to evaluate.")

        print("Compute FID")
        if self._enable_fid:
            mu_real = self.mu_real
            mu_fake = self._fid_fake_total / self._num_examples
            sigma_real = self.sigma_real
            sigma_fake = get_covariance(self._fid_fake_sigma, self._fid_fake_total, self._num_examples)

            mu_real, mu_fake = mu_real.cpu(), mu_fake.cpu()
            sigma_real, sigma_fake = sigma_real.cpu(), sigma_fake.cpu()

            diff = mu_real - mu_fake

            # Product might be almost singular.
            covmean, _ = linalg.sqrtm(sigma_real.mm(sigma_fake).numpy(), disp=False)
            # Numerical error might give slight imaginary component.
            if np.iscomplexobj(covmean):
                if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                    m = np.max(np.abs(covmean.imag))
                    raise ValueError("Imaginary component {}".format(m))
                covmean = covmean.real

            tr_covmean = np.trace(covmean)

            if not np.isfinite(covmean).all():
                tr_covmean = np.sum(np.sqrt((
                    (np.diag(sigma_real) * self._fid_eps) * (np.diag(sigma_fake) * self._fid_eps))
                    / (self._fid_eps * self._fid_eps)
                ))

            gfid = float(diff.dot(diff).item() + torch.trace(sigma_real) + torch.trace(sigma_fake) 
                - 2 * tr_covmean
            )

            eval_score["FID"] = gfid

        return eval_score