import torch

def feature_loss(true_features, gen_features):
    result = 0
    for true_feature, gen_feature in zip(true_features, gen_features):
        for true_conv_out, gen_conv_out in zip(true_feature, gen_feature):
            result += torch.mean(torch.abs(true_conv_out - gen_conv_out))

    return 2 * result

def discriminator_loss(true_outputs, gen_outputs):
    result = 0
    for true_output, gen_output in zip(true_outputs, gen_outputs):
        result += torch.mean((1 - true_output)**2) + torch.mean(gen_output**2)

    return result

def generator_loss(gen_outputs):
    result = 0
    for gen_output in gen_outputs:
        result += torch.mean((1 - gen_output)**2)

    return result
