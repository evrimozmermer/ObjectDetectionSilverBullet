from . import retinanet

# Create the model
def load(cfg):
    if cfg.backbone == "resnet18":
        model = retinanet.resnet18(num_classes=cfg.nb_classes, pretrained=True)
    elif cfg.backbone == "resnet34":
        model = retinanet.resnet34(num_classes=cfg.nb_classes, pretrained=True)
    elif cfg.backbone == "resnet50":
        model = retinanet.resnet50(num_classes=cfg.nb_classes, pretrained=True)
    elif cfg.backbone == "resnet101":
        model = retinanet.resnet101(num_classes=cfg.nb_classes, pretrained=True)
    elif cfg.backbone == "resnet152":
        model = retinanet.resnet152(num_classes=cfg.nb_classes, pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    
    return model