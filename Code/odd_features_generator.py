from feature_models.color_moments import ColorMomentsExtractor
from feature_models.hog import HOGExtractor
from feature_models.resnet import ResNetExtractor
from utils.database_utils import store
from utils.dataset_utils import initialize_dataset, get_image_with_label


FEATURES_TO_GENERATE = {"resnet"}


def main():
    """
    Extract feature vectors for odd images in Caltech101 and store.
    """
    dataset = initialize_dataset()

    print(dataset, "\n")

    color_vectors = {}
    hog_vectors = {}
    avgpool_vectors = {}
    layer3_vectors = {}
    fc_vectors = {}
    resnet_vectors = {}

    for img_id in range(len(dataset)):

        # Skip images that do not have an odd image ID.
        if img_id % 2 == 0:
            continue

        print(f"Processing image {img_id} \t ({int(100*(img_id+1)/len(dataset))} %)", end='\r')

        image, label = get_image_with_label(dataset, img_id)

        # Compute color moments vector.
        if "color" in FEATURES_TO_GENERATE:
            color_vectors[img_id] = (label, ColorMomentsExtractor(image).get_color_vector())

        # Compute HOG moments vector.
        if "hog" in FEATURES_TO_GENERATE:
            hog_vectors[img_id] = (label, HOGExtractor(image).get_hog_vector())

        # Generate outputs from ResNet50.
        if "layer3" in FEATURES_TO_GENERATE or "avgpool" in FEATURES_TO_GENERATE or "fc" in FEATURES_TO_GENERATE or "resnet" in FEATURES_TO_GENERATE:
            resnet = ResNetExtractor(image)

            # Retrieve vectors from hooked layers and output.
            if "avgpool" in FEATURES_TO_GENERATE:
                avgpool_vectors[img_id] = (label, resnet.get_avgpool_vector())

            if "layer3" in FEATURES_TO_GENERATE:
                layer3_vectors[img_id] = (label, resnet.get_layer3_vector())

            if "fc" in FEATURES_TO_GENERATE:
                fc_vectors[img_id] = (label, resnet.get_fc_vector())

            if "resnet" in FEATURES_TO_GENERATE:
                resnet_vectors[img_id] = (label, resnet.get_output_vector())


    # Save all vectors
    if "color" in FEATURES_TO_GENERATE:
        store(color_vectors, "oddcolor.pt")
    if "hog" in FEATURES_TO_GENERATE:
        store(hog_vectors, "oddhog.pt")
    if "avgpool" in FEATURES_TO_GENERATE:
        store(avgpool_vectors, "oddavgpool.pt")
    if "layer3" in FEATURES_TO_GENERATE:
        store(layer3_vectors, "oddlayer3.pt")
    if "fc" in FEATURES_TO_GENERATE:
        store(fc_vectors, "oddfc.pt")
    if "resnet" in FEATURES_TO_GENERATE:
        store(resnet_vectors, "oddresnet.pt")


if __name__ == "__main__":
    main()