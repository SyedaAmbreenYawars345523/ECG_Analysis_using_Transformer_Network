import torch.nn as nn
import numpy as np
from utils import *



class EcgModel(nn.Module):

    def __init__(self, d_model, nhead, num_layers, num_conv_layers):
        super(EcgModel, self).__init__()
        # self.init_weights()

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.n_conv_layers = num_conv_layers

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=d_model, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding=0)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.pos_encoder = PositionalEncoding(d_model)  # why 748???
        #self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512)
        #self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        #self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=512)
        #self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)
        #self.self_att_pool = SelfAttentionPooling(d_model)
        self.transformer = nn.Transformer(nhead=4, num_encoder_layers=4, d_model=64, dim_feedforward=512,
                                          num_decoder_layers=4, batch_first=False)
        ## ADD A MLP ITEM here###
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, src, tgt):
        # 1. make input of shape [batch, input_channels, signal_length]")
        src = src.view(src.shape[0], 1, src.shape[1])  # Resize to --> [batch, input_channels, signal_length]
        # print("\n1. setting input channel for convolution,  # Resize to --> [batch, input_channels, signal_length]")
        # print(src.shape)

        # print("\n2 convolution embedding")
        src = self.relu(self.conv1(src))
        # print("After conv1 shape: [batch,out_channel, embedded_seq_len_out]")
        # print(src.shape)

        src = self.relu(self.conv2(src))
        # print("conv2 shape : : [batch,out_channel, embedded_seq_len_out]")
        # print(src.shape)

        for i in range(self.n_conv_layers):
            src = self.relu(self.conv(src))
            src = self.maxpool(src)
        # print("src after relu and max pooling: : [batch,out_channel, embedded_seq_len_out]")
        # print(src.shape)

        # print("\n 3 Positional encoding")
        # print("required shape for pos_encod: [embedd_seq_len, batch size, d_model]")

        # print("change shape from [batch, seq, d_model]--> [embedd_seq_len,batch,d_model]")
        src = src.permute(2, 0, 1)

        # print(src.shape)

        src = self.pos_encoder(src)
        # print("/n positional encoder size [batch, embedding, sequence] ")
        # print("/n after pos_encoding:[embedded_seq_len, batch, d_model]")

        # print("`\n queri shape")
        # print(tgt.shape) # 8,4,1
        tgt = tgt.view(tgt.shape[0], 1, tgt.shape[1])  # 8,1,4
        # print(tgt.shape)
        # print(tgt.shape)
        # tgt = tgt.permute(2, 0, 1)
        tgt = self.conv3(tgt)
        # print(tgt.shape)
        tgt = tgt.permute(2, 0, 1)
        # print("after convolution and permute)
        # print(tgt.shape)


        tgt = self.pos_encoder(tgt)
        # print("src shape")
        # print(src.shape)
        # print("tgt final shape")
        # print(tgt.shape)

        output = self.transformer(src, tgt)
        # print(output.shape)
        # print(" above output after decoder") # seq, batch, embeding

        output = output.permute(1, 0, 2)
        # print("now the shape after permute")
        # print(output.shape) # batch , seq, embedding

        # print("applying mlp")
        # print(results)
        out = self.mlp(output)
        # print(out.shape)
        # print(out)

        return out


