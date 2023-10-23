import cv2
import pyvips

import torch
import torch.nn as nn
from torchinfo import summary

from models.encoders import load_resnet
from models.clam import CLAM_SB, CLAM_MB


# resnet 50 clam: 16x downsample @ 1024 features (this is the adjusted resnet50 clam uses)
# Technically, it is a ResNet39
# 8192 x 8192    -> 512 x 512
# 16384 x 16384  -> 1024 x 1024
# 32768 x 32786  -> 2048 x 2048

# Resnet 50: 32x downsample @ 2048 features
# 8192 x 8192    -> 256 x 256
# 16384 x 16384  -> 512 x 512
# 32768 x 32786  -> 1024 x 1024


def cv2_resize(img, width, height):

    dim = (width, height)
    print(img.shape, width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    print("image size after resizing ", img.shape)
    return img

def read_image(path):
    image = pyvips.Image.new_from_file(path)
    image = image.write_to_memory()


class StreamingCLAM(nn.Module):

    def __init__(self, max_pool_kernel=0, attention_only=False, *args, **kwargs):
        super().__init__()

        assert isinstance(max_pool_kernel, int) and max_pool_kernel >= 0

        # return only attention, or instance cluster feature vectors
        self.attention_only = attention_only
        self.return_features = False

        #encoder options
        self.encoder_type = kwargs.get('encoder')

        # Downsample options
        self.max_pool_kernel = max_pool_kernel
        self.remove_first_maxpool = kwargs.get('remove_first_maxpool', False)
        self.remove_first_stride = kwargs.get('remove_first_stride', False)

        # Clam specific options
        self.branch = kwargs.get('branch', 'sb')
        self.disable_gate = kwargs.get('disable_gate', False)
        self.use_dropout = kwargs.get('use_dropout', False)
        self.k_sample = kwargs.get('k_sample', 8)
        self.instance_loss_fn = kwargs.get('instance_loss_fn', nn.CrossEntropyLoss())
        self.n_classes = kwargs.get('n_classes', 2)
        self.subtyping = kwargs.get('subtyping', False)

        assert self.encoder_type in ('resnet34', 'resnet50', 'convnext')
        print(f"Loading {self.encoder_type} encoder")

        if self.encoder_type == 'resnet50':
            self.encoder = load_resnet('resnet50')
            # resnet 50 encoder has 2048 channels @ 32 downsample
            size = [2048, 512, 256]
        elif self.encoder_type == 'resnet34':
            self.encoder = load_resnet('resnet34')

            if self.remove_first_maxpool:
                print("Removing first max pooling layer")
                self.encoder.maxpool = torch.nn.Sequential()
            if self.remove_first_stride:
                print("setting stride of first convolution equal to 1")
                self.encoder.conv1.stride = (1, 1)

            # resnet 34 encoder has 512 channels @ 32 downsample
            size = [512, 512, 256]


        self.ds_blocks = nn.Sequential()

        print("Using extra max pool layer with kernel size and stride", self.max_pool_kernel)
        self.ds_blocks = nn.Sequential(
            nn.MaxPool2d((self.max_pool_kernel, self.max_pool_kernel))
        )

        # size args original: self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        if self.branch == 'sb':
            print("Loading CLAM with single branch \n")
            self.head = CLAM_SB(
                gate=not self.disable_gate,
                size=size,
                dropout=self.use_dropout,
                k_sample=self.k_sample,
                n_classes=self.n_classes,
                instance_loss_fn=self.instance_loss_fn,
                subtyping=self.subtyping,
            )
        elif self.branch == 'mb':
            print("Loading CLAM with multiple branches \n")

            self.head = CLAM_MB(
                gate=not self.disable_gate,
                size=size,
                dropout=self.use_dropout,
                k_sample=self.k_sample,
                n_classes=self.n_classes,
                instance_loss_fn=self.instance_loss_fn,
                subtyping=self.subtyping,
            )
        else:
            raise NotImplementedError(f"branch must be specified as single branch "
                                      f"'sb' or multi branch 'mb', not {self.branch}")

    def forward(self, x, mask=None, instance_eval=False, label=None, return_features=False, attention_only=False, debug=False):

        out = self.encoder(x)
        if self.max_pool_kernel > 0:
            out = self.ds_blocks(out)

        # Streamingclam forward pass
        if debug:
            return out

        batch_size, num_features, h, w = out.shape

        # Tensor of shape [batch_size, C, H, W]
        # change dimensions to [batch_size, C, H * W]

        # TODO: automate number of features, downsample of network, mask downsample
        if mask is not None:
            out = torch.masked_select(out, mask)
            del mask

        # Put everything back together into an array [channels, #unmasked_pixels]
        # This operation heavily reduces the number of inputs to the clam network as well

        out = torch.reshape(out, (num_features, -1)).transpose(0, 1)

        if self.attention_only:
            return self.head(out, label=None, instance_eval=False, attention_only=self.attention_only)

        logits, Y_prob, Y_hat, A_raw, instance_dict = self.head(
            out, label=label, instance_eval=instance_eval, return_features=return_features, attention_only=attention_only
         )

        return logits, Y_prob, Y_hat, A_raw, instance_dict


