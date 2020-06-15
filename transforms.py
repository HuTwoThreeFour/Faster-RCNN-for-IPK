import torch
import math
import cv2
import numpy as np
import random
import numbers
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data

class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ComposeForColorJitter(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            # print('flip on.')
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]

            ## wiesooooooooo?????
            #image_transform = transforms.ToPILImage()(image).convert('RGB')
            #bbox_array = bbox[0].numpy()
            #print(bbox_array)
            #draw = ImageDraw.Draw(image_transform)
            #draw.rectangle(bbox_array, outline=(255, 0, 0))
            #image_transform.show()



            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target

class RandomRotation(object):
    def __call__(self, image, target):
        bbox = target['boxes']
        ## for testttttt1
        #bbox_array1 = bbox[0].numpy()
        #print(bbox_array1)
        #image_test1 = image
        #draw1 = ImageDraw.Draw(image_test1)
        #draw1.rectangle(bbox_array1, outline=(255, 0, 0))
        #image_test1.show()

        #set the random angles
        angles = [0, 90, 180, -90]
        angle = random.choice(angles)
        #print('the random angle is: ', angle)
        image_array = F.to_tensor(image)
        width, height = image_array.shape[-2:]
        # print(type(height))
        # print(type(int((height + width)/2)))
        image = F.rotate(image, angle)
        bbox_new = torch.zeros_like(bbox)
        if angle == 180:
            bbox_new[:, [0, 2]] = width - bbox[:, [2, 0]]
            bbox_new[:, [1, 3]] = height - bbox[:, [3, 1]]
        elif angle == -90:
            bbox_new[:, [0, 2]] = (height + width)/2 - bbox[:, [3, 1]]
            bbox_new[:, [1, 3]] = (height - width)/2 + bbox[:, [0, 2]]

        elif angle == 90:
            bbox_new[:, [0, 2]] = (width - height)/2 + bbox[:, [1, 3]]
            bbox_new[:, [1, 3]] = (width + height)/2 - bbox[:, [2, 0]]
        else:
            bbox_new = bbox
        #print('the rotated bbox is:',bbox)
        ## for testtttt2
        #bbox_array2 = bbox_new[0].numpy()
        #print(bbox_array2)
        #image_test2 = image
        #draw2 = ImageDraw.Draw(image_test2)
        #draw2.rectangle(bbox_array2, outline=(255, 0, 0))
        #image_test2.show()

        target['boxes'] = bbox_new
        return image, target


# class RandomRotation(object):
#     def __call__(self, image, target):
#         height, width = image.shape[-2:]
#         bbox = target['boxes']
#         # create a image with size of origin image
#         bbox_shape = image.size
#         bbox_img = np.zeros(bbox_shape)
#         #print(bbox)
#         bbox = bbox.numpy()
#         #print(bbox)
#         # set the corresponding values into the array
#         bbox_img[int(bbox[0][1])][int(bbox[0][0])] = 100
#         bbox_img[int(bbox[0][3])][int(bbox[0][0])] = 100
#         bbox_img[int(bbox[0][1])][int(bbox[0][2])] = 100
#         bbox_img[int(bbox[0][3])][int(bbox[0][2])] = 100
#         bbox_img = Image.fromarray(bbox_img)
#         bbox_img = bbox_img.convert('RGB')
#         image.show('original image')
#         bbox_img.show('bbox before rotation')
#
#         #set the random angles
#         angles = [-90, 0, 90, 180]
#         angle = random.choice(angles)
#         print('the random angle is: ', angle)
#         image = F.rotate(image, angle)
#         image.show('rotated img:')
#         #print(image)
#         #print(bbox_img)
#
#         # update the values of bbox
#         bbox_img = F.rotate(bbox_img, angle)
#         bbox_img.show('bbox after rotation')
#         if angle in [-90, 90]:
#             bbox_x = np.nonzero(bbox_img)[1]
#             bbox_y = np.nonzero(bbox_img)[0]
#         else:
#             bbox_x = np.nonzero(bbox_img)[0]
#             bbox_y = np.nonzero(bbox_img)[1]
#         x_min = min(bbox_x)
#         x_max = max(bbox_x)
#         y_min = min(bbox_y)
#         y_max = max(bbox_y)
#         boxes = []
#         boxes.append([x_min, y_min, x_max, y_max])
#         boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         print('rotated bboxes is: ', boxes)
#         target['boxes'] = boxes
#         return image, target


# class random_affine(object):
#     def __init__(self, degrees=10, translate=.1, scale=.1, shear=10, border=0):
#         self.degrees = degrees
#         self.translate = translate
#         self.scale = scale
#         self.shear = shear
#         self.border = border
#     def __call__(self, img, target):
#         # rewrite from yolov3-datasets.py
#         # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
#         # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
#
#         bbox = target['boxes']
#
#         # if targets is None:  # targets = [cls, xyxy]
#         #     targets = []
#         height = img.size[1] + self.border * 2
#         width = img.size[0] + self.border * 2
#
#         # Rotation and Scale
#         R = np.eye(3)
#         a = random.uniform(-self.degrees, self.degrees)
#         # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
#         s = random.uniform(1 - self.scale, 1 + self.scale)
#         R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.size[0] / 2, img.size[1] / 2), scale=s)
#
#         # Translation
#         T = np.eye(3)
#         T[0, 2] = random.uniform(-self.translate, self.translate) * img.size[1] + self.border  # x translation (pixels)
#         T[1, 2] = random.uniform(-self.translate, self.translate) * img.size[0] + self.border  # y translation (pixels)
#
#         # Shear
#         S = np.eye(3)
#         S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
#         S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)
#
#         # Combined rotation matrix
#         M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
#         changed = (self.border != 0) or (M != np.eye(3)).any()
#         if changed:
#             img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#             img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_AREA, borderValue=(128, 128, 128))
#
#         # Transform label coordinates
#         n = len(bbox)
#         if n:
#             # warp points
#             xy = np.ones((n * 4, 3))
#             xy[:, :2] = bbox[[0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
#             # xy[:, :2] = bbox[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
#             xy = (xy @ M.T)[:, :2].reshape(n, 8)
#
#             # create new boxes
#             x = xy[:, [0, 2, 4, 6]]
#             y = xy[:, [1, 3, 5, 7]]
#             xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
#
#             # # apply angle-based reduction of bounding boxes
#             # radians = a * math.pi / 180
#             # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
#             # x = (xy[:, 2] + xy[:, 0]) / 2
#             # y = (xy[:, 3] + xy[:, 1]) / 2
#             # w = (xy[:, 2] - xy[:, 0]) * reduction
#             # h = (xy[:, 3] - xy[:, 1]) * reduction
#             # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T
#
#             # reject warped points outside of image
#             xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
#             xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
#             w = xy[:, 2] - xy[:, 0]
#             h = xy[:, 3] - xy[:, 1]
#             area = w * h
#             area0 = (bbox[:, 3] - bbox[:, 1]) * (bbox[:, 4] - bbox[:, 2])
#             ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
#             i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.2) & (ar < 10)
#
#             bbox = bbox[i]
#             bbox[:, 1:5] = xy[i]
#             target['boxes'] = bbox
#         img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#         return img, target



# class RandomAffine(object):
#     def __init__(self, degrees, translate=None, scale=None, shear=None):
#         if isinstance(degrees, numbers.Number):
#             if degrees < 0:
#                 raise ValueError("If degrees is a single number, it must be positive.")
#             self.degrees = (-degrees, degrees)
#         else:
#             assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
#                 "degrees should be a list or tuple and it must be of length 2."
#             self.degrees = degrees
#
#         if translate is not None:
#             assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
#                 "translate should be a list or tuple and it must be of length 2."
#             for t in translate:
#                 if not (0.0 <= t <= 1.0):
#                     raise ValueError("translation values should be between 0 and 1")
#         self.translate = translate
#
#         if scale is not None:
#             assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
#                 "scale should be a list or tuple and it must be of length 2."
#             for s in scale:
#                 if s <= 0:
#                     raise ValueError("scale values should be positive")
#         self.scale = scale
#
#         if shear is not None:
#             if isinstance(shear, numbers.Number):
#                 if shear < 0:
#                     raise ValueError("If shear is a single number, it must be positive.")
#                 self.shear = (-shear, shear)
#             else:
#                 assert isinstance(shear, (tuple, list)) and \
#                     (len(shear) == 2 or len(shear) == 4), \
#                     "shear should be a list or tuple and it must be of length 2 or 4."
#                 # X-Axis shear with [min, max]
#                 if len(shear) == 2:
#                     self.shear = [shear[0], shear[1], 0., 0.]
#                 elif len(shear) == 4:
#                     self.shear = [s for s in shear]
#         else:
#             self.shear = shear
#
#     @staticmethod
#     def get_params(degrees, translate, scale_ranges, shears, img_size):
#         """Get parameters for affine transformation
#
#         Returns:
#             sequence: params to be passed to the affine transformation
#         """
#         angle = random.uniform(degrees[0], degrees[1])
#         if translate is not None:
#             max_dx = translate[0] * img_size[0]
#             max_dy = translate[1] * img_size[1]
#             translations = (np.round(random.uniform(-max_dx, max_dx)),
#                             np.round(random.uniform(-max_dy, max_dy)))
#         else:
#             translations = (0, 0)
#
#         if scale_ranges is not None:
#             scale = random.uniform(scale_ranges[0], scale_ranges[1])
#         else:
#             scale = 1.0
#
#         if shears is not None:
#             if len(shears) == 2:
#                 shear = [random.uniform(shears[0], shears[1]), 0.]
#             elif len(shears) == 4:
#                 shear = [random.uniform(shears[0], shears[1]),
#                          random.uniform(shears[2], shears[3])]
#         else:
#             shear = 0.0
#         return angle, translations, scale, shear
#
#     def __call__(self, img, target):
#         """
#             img (PIL Image): Image to be transformed.
#             target: target to be transformed
#
#         Returns:
#             PIL Image: Affine transformed image.
#             target: Affine transformed target(bbox)
#         """
#         bbox = target['boxes']
#         # create a image with size of origin image
#         bbox_shape = img.size
#         bbox_img = np.zeros(bbox_shape)
#         #print(bbox)
#         bbox = bbox.numpy()
#         #print(bbox)
#         # set the corresponding values into the array
#         bbox_img[int(bbox[0][0])][int(bbox[0][1])] = 100
#         bbox_img[int(bbox[0][0])][int(bbox[0][3])] = 100
#         bbox_img[int(bbox[0][2])][int(bbox[0][1])] = 100
#         bbox_img[int(bbox[0][2])][int(bbox[0][3])] = 100
#         bbox_img = Image.fromarray(bbox_img)
#         bbox_img = bbox_img.convert('RGB')
#
#         ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
#         img = F.affine(img, *ret)
#
#         # update the values of bbox
#         bbox_img = F.affine(bbox_img, *ret)
#         bbox_x = np.nonzero(bbox_img)[0]
#         x_min = min(bbox_x)
#         x_max = max(bbox_x)
#         bbox_y = np.nonzero(bbox_img)[1]
#         y_min = min(bbox_y)
#         y_max = max(bbox_y)
#         boxes = []
#         boxes.append([x_min, y_min, x_max, y_max])
#         boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         #print(boxes)
#         target['boxes'] = boxes
#         return img, target
#
#     def __repr__(self):
#         s = '{name}(degrees={degrees}'
#         if self.translate is not None:
#             s += ', translate={translate}'
#         if self.scale is not None:
#             s += ', scale={scale}'
#         if self.shear is not None:
#             s += ', shear={shear}'
#         s += ')'
#         d = dict(self.__dict__)
#         return s.format(name=self.__class__.__name__, **d)

class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = ComposeForColorJitter(transforms)

        return transform

    def __call__(self, img, target):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        img = transform(img)
        return img, target

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string



class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)

        return image, target
