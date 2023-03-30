import torch


def load_conv_batch_norm(buf, start, conv_layer, batch_norm_layer):
    num_b = batch_norm_layer.bias.numel()
    batch_norm_layer.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start += num_b

    batch_norm_layer.weight.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start += num_b

    batch_norm_layer.running_mean.copy_(torch.from_numpy(buf[start:start + num_b]))
    start += num_b

    batch_norm_layer.running_var.copy_(torch.from_numpy(buf[start:start + num_b]))
    start += num_b

    num_w = conv_layer.weight.numel()
    conv_layer.weight.data.copy_(
        torch.from_numpy(buf[start:start + num_w]).reshape(conv_layer.weight.data.shape))
    start += num_w
    return start


def load_conv(buf, start, conv_layer):
    num_b = conv_layer.bias.numel()
    conv_layer.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start += num_b

    num_w = conv_layer.weight.numel()
    conv_layer.weight.data.copy_(
        torch.from_numpy(buf[start:start + num_w]).reshape(conv_layer.weight.data.shape))
    start += num_w
    return start

