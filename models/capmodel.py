import torch.nn as nn
class CapModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(CapModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, cnn_feats, depth_feats, captions, teacher_forcing_ratio=1.0):
        frame_feats, vid_feats = self.encoder(cnn_feats, depth_feats)
        outputs, module_weights = self.decoder(frame_feats, vid_feats, captions, teacher_forcing_ratio)
        return outputs, module_weights