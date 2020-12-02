import torch
from torch import nn



class ContractingBlock(nn.Module):
    """ContractingBlock Class
    Performs two convolutions followed by a max pool operation

    Args:
        input_channels (int): the number of channels to expect from a given input
    """
    def __init__(self, input_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=3)
        self.conv2 = nn.Conv2d(input_channels * 2, input_channels * 2, kernel_size=3)
        self.activation = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """Function form completing agorward pass of ContractingBlock
            Given an image tensor, completes a contracting block and returns the transformed tensor

        Args:
            x (tensor): iamge tensor of shape (batch_size, channels, height, width)
        """
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.maxpool(x)
        return x


def crop(image, new_shape):
    """Function for center cropping an image tensor:

    Args:
        image (tensor): tensor image of shape (batch_size, channels, width, height)
        new_shape (torch.size): torch.size object with the new shape 
    """
    height_middle = image.shape[2] // 2
    width_middle = image.shape[3] // 2

    min_height = height_middle - (new_shape[2] // 2)
    max_height = min_height + new_shape[2]
    min_width = width_middle - (new_shape[3] // 2)
    max_width = min_width + new_shape[3]

    return image[:, :, min_height:max_height, min_width:max_width]


class ExpandingBlock(nn.Module):
    """ExpandingBlock class
        Performs an upsampling comvolution, a concatenation of its two inputs, followed by two more convolutions
    Args:
        input_channels (int): the number of input channels
    """
    def __init__(self, input_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=2)
        self.conv2 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=3)
        self.conv3 = nn.Conv2d(input_channels // 2, input_channels //2, kernel_size=3)
        self.activation = nn.ReLU()

    def forward(self, x, skip_con_x):
        """Function fror completing a forward pass of Expanding Block:

        Args:
            x (tensor): image tensor of shape (batch_size, channels, height, width)
            skip_con_x (tensor): image tensor from the contracting path, from the opposing block of x
        """
        x = self.upsample(x)
        x = self.conv1(x)
        skip_con_x = crop(skip_con_x, x.shape)
        x = torch.cat([x, skip_con_x], axis=1)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        
        return x


class FeatureMapBlock(nn.Module):
    """FeatureMapBlock Class
        The final layer of a U-NET. Maps each pixel to a pixel with the correct number of output dimensions using a 1x1 conv
    Args:
        input_channels (int): the number of channels to expect from a given input
        output_channels (int): the number of channels the ooutput should have
    """
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        """Method to completing a forward pass of the FeatureMapBlock

        Args:
            x (tensor): image tensor of shape (batch_size, channels, height, width)
        """
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """UNet Class
        A series of 4 contracting Blocks followed by 4 expanding Blocks to transform an input image
        into the corresponding paired image, with an upfeature layer at the start, and downfeature layer at the end
    Args:
        input_channels (int): the number of channels to expect from a given input
        output_channels (int): the number of channels the output should have
        hidden_channels (int): the number of hidden channels
    """
    def __init__(self, input_channels, output_channels, hidden_channels=64):
        super().__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        self.contract3 = ContractingBlock(hidden_channels * 4 )
        self.contract4 = ContractingBlock(hidden_channels * 8)
        self.expand1 = ExpandingBlock(hidden_channels * 16)
        self.expand2 = ExpandingBlock(hidden_channels * 8)
        self.expand3 = ExpandingBlock(hidden_channels * 4)
        self.expand4 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)

    def forward(self, x):
        """Method for completing a forward pass of UNet

        Args:
            x (tensor): image tensor of shape (batch_size, channels, height, width)
        """
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        x5 = self.expand1(x4, x3)
        x6 = self.expand2(x5, x2)
        x7 = self.expand3(x6, x1)
        x8 = self.expand4(x7, x0)
        xn = self.downfeature(x8)

        return xn
