from flowerclf.model import build_model


def test_classifier_head_matches_num_classes():
    model = build_model(5, pretrained=False)
    assert model.fc.out_features == 5


def test_backbone_is_frozen_and_head_is_trainable():
    model = build_model(5, pretrained=False)
    assert not model.conv1.weight.requires_grad
    assert model.fc.weight.requires_grad
