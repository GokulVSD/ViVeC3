from torch import no_grad
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights
from torchvision.transforms.transforms import Compose, ToTensor
from utils.image_utils import convert_to_rgb


class ResNetExtractor:
    """
    ResNet-50 Pretrained with default weights.
    https://pytorch.org/vision/master/models/generated/torchvision.models.resnet50.html
    """

    def __init__(self, image):
        """
        Preprocess PIL image for ResNet extraction.
        """
        # Convert image that isn't RGB to RGB.
        image = convert_to_rgb(image)

        # Resize original image to 244x244.
        image224x224 = image.resize((224, 224))

        # Convert to float tensor.
        self.img_tensor = Compose([ToTensor()])(image224x224)

        # Use default pre-trained weights (ImageNet1K_V2).
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Set ResNet-50 to evaluation mode.
        self.model.eval()

        self.__setup_hooks()

        self.__generate_output()


    def get_avgpool_vector(self):
        """
        Vector of the modified output of average pool layer for the provided image.
        Shape is 1x2048, we have to average every 2 values sequentially
        to obtain a 1024 dimension vector. We do this by reshaping to
        a vector of 2 value vectors, and mean each 2 value vector.
        """
        return self.layer_outputs["avgpool"].reshape([1024, 2]).mean(dim=1)


    def get_layer3_vector(self):
        """
        Vector of the modified output of layer 3 for the provided image.
        Shape is 1x1024x14x14, We have to take the mean of each 14x14 slice,
        resulting in a 1024 dimension vector.
        """
        return self.layer_outputs["layer3"].mean(dim=(2, 3))[0]


    def get_fc_vector(self):
        """
        Vector of the output of the fully connected layer for the provided image.
        Shape is 1x1000.
        """
        return self.layer_outputs["fc"][0]


    def get_output_vector(self):
        """
        Vector of the final output of the ResNet model for the provided image.
        """
        return self.layer_outputs["resnet_output"][0]


    def __setup_hooks(self):
        """
        Attach hooks for relevant layers.
        """
        self.layer_outputs = {}

        def avgpool_hook(module, input, output):
            self.layer_outputs["avgpool"] = output

        def layer3_hook(module, input, output):
            self.layer_outputs["layer3"] = output

        def fc_hook(module, input, output):
            self.layer_outputs["fc"] = output

        self.hooks = {}

        self.hooks["avgpool"] = self.model.avgpool.register_forward_hook(avgpool_hook)
        self.hooks["layer3"] = self.model.layer3.register_forward_hook(layer3_hook)
        self.hooks["fc"] = self.model.fc.register_forward_hook(fc_hook)


    def __generate_output(self):
        """
        Generate output for input image and store in
        """
        # Don't compute gradient, we are only inferencing.
        with no_grad():
            self.layer_outputs["resnet_output"] = self.model(self.img_tensor.unsqueeze(dim=0))

        # Detach hooks.
        for hook in self.hooks.values():
            hook.remove()