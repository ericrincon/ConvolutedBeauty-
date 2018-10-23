from convolutedbeauty.models.pretrained import DenseNet121, DenseNet169, DenseNet201

from convolutedbeauty.models.architectures import DenseNet

_pretrained_models = {
    "dense121": DenseNet121,
    "desnse169": DenseNet169,
    "dense201": DenseNet201
}

_models = {
    "se-dense": DenseNet
}


def get_pretrained_model(model_name):
    """
    Gets a pretrained a defined pretrained model
    :param model_name: (str) the pretrained model name
    :return: a pytorch Module object with pretrained weights

    """

    model_name = model_name.lower()

    if model_name not in _pretrained_models:
        raise ValueError("The model {} is not defined!".format(model_name))

    return _pretrained_models[model_name]


def get_model(model_name):
    model_name = model_name.lower()

    if model_name not in _models:
        raise ValueError("The model {} is not defined!".format(model_name))

    return _models[model_name]
