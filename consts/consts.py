import enum


class EnumConstant(enum.Enum):
    def __str__(self):
        return self.value


class Split(EnumConstant):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class DatasetArgs(EnumConstant):
    CATSDOGS = 'catsdogs'
    MIT67 = 'mit67'
    ILSVRC2012 = 'ilsvrc2012'
    MAME = 'mame'
    CARSNCATS = 'carsncats'
    MOUNTAINDOGS = 'mountaindogs'


class MosaicArgs(EnumConstant):
    CATSDOGS_MOSAIC = 'catsdogs_mosaic'
    MIT67_MOSAIC = 'mit67_mosaic'
    ILSVRC2012_MOSAIC = 'ilsvrc2012_mosaic'
    MAME_MOSAIC ='mame_mosaic'
    CARSNCATS_MOSAIC = 'carsncats_mosaic'
    MOUNTAINDOGS_MOSAIC = 'mountaindogs_mosaic'


class XmethodArgs(EnumConstant):
    LRP = 'lrp'
    GRADCAM = 'gradcam'
    GRADCAMPLUSPLUS = 'gradcam++'
    SMOOTHGRAD = 'smoothgrad'
    INTGRAD = 'intgrad'
    LIME = 'lime'
    BCOS = 'bcos'
    SHAP = 'shap'

class ArchArgs(EnumConstant):
    RESNET18 = 'resnet18'
    RESNET34 = 'resnet34'
    RESNET50 = 'resnet50'
    RESNET101 = 'resnet101'
    RESNET152 = 'resnet152'
    VGG11_BN = 'vgg11_bn'
    VGG13 = 'vgg13'
    VGG16 = 'vgg16'
    VGG19 = 'vgg19'
    ALEXNET = 'alexnet'
