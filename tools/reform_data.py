import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2

ia.seed(1)

# Example batch of images.
# The array has shape (32, 64, 64, 3) and dtype uint8.
images = np.array(
    [ia.quokka(size=(64, 64)) for _ in range(32)],
    dtype=np.uint8
)
src = cv2.imread("C:\\Users\\xinyi\\Documents\\XYBin_Collected\\tangle_final_fine\\U\\10\\depth.png")
images = np.array(
    [src for _ in range(32)], dtype=np.uint8
)
degrees_aug = [0,]
seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    #iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.5,
        iaa.Sometimes(0.5, iaa.ElasticTransformation(alpha=0.5, sigma=0.5)),
        iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(loc=0, scale=0.025*255))
    ),
    iaa.Affine(
        rotate=90,
    )
], random_order=True) # apply augmenters in random order

images_aug = seq(images=images)
print(images_aug.shape)
ia.show_grid(images_aug, rows=4, cols=8)