from torchvision.utils import make_grid
import matplotlib.pyplot as plt



def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    """Function for visualizing images
    """
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=4)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
