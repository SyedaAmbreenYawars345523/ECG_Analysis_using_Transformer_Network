import torch
import torch.nn as nn

from positionalencoding import PositionalEncoding
from selfattentionpooling import SelfAttentionPooling





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
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding=0)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.pos_encoder = PositionalEncoding(d_model)  # why 748???
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead,dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.self_att_pool = SelfAttentionPooling(d_model)
        ## ADD A MLP ITEM here###
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,1)
        )


    #def init_weights(self):
     #   initrange = 0.1
      #  self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        #print("\ninitial size of input [batch, seq_len]")
        #print(src.shape)
        # size input: [batch, seq_len]
        # src = resize?? Resize to --> [batch, input_channels, signal_length]

        #1. make input of shape [batch, input_channels, signal_length]")
        src = src.view(src.shape[0], 1, src.shape[1])  # Resize to --> [batch, input_channels, signal_length]
        #print("\n1. setting input channel for convolution,  # Resize to --> [batch, input_channels, signal_length]")
        #print(src.shape)

        #print("\n2 convolution embedding")
        src = self.relu(self.conv1(src))
        #print("After conv1 shape: [batch,out_channel, embedded_seq_len_out]")
        #print(src.shape)

        src = self.relu(self.conv2(src))
        #print("conv2 shape : : [batch,out_channel, embedded_seq_len_out]")
        #print(src.shape)

        for i in range(self.n_conv_layers):
            src = self.relu(self.conv(src))
            src = self.maxpool(src)
        #print("src after relu and max pooling: : [batch,out_channel, embedded_seq_len_out]")
        #print(src.shape)

        #print("\n 3 Positional encoding")
        #print("required shape for pos_encod: [embedd_seq_len, batch size, d_model]")

        #print("change shape from [batch, seq, d_model]--> [embedd_seq_len,batch,d_model]")
        src = src.permute(2, 0, 1)

        #print(src.shape)

        src = self.pos_encoder(src)
        #print("/n positional encoder size [batch, embedding, sequence] ")
        #print("/n after pos_encoding:[embedded_seq_len, batch, d_model]")

        #print(src.shape) # [embedded_seq_len, batch, d_model]



        #print("\n4.transformer encoder")

        #TA models require input data to be stored in a 3-dim array/tensor with shape [seq_len, bat_size, embed_dim]. So for the batch of 2 sentences, each with 3 words, and each word having 4 embedding values, there would be 3 * 2 * 4 = 24 numeric values.
        #src = torch.reshape(src.shape[0],batch,src.shape[2])
        #src =src.view(-1,batch,src.shape[2])
        #[sequence, batch, embedding dim.]



        # src with batch.
        # changing shape??  to send it to transformer encoder
        #src = src.permute(2, 0,
        #                  1)  # reshape from [batch, embedding dim., sequnce] --> [sequence, batch, embedding dim.] embedded_seq_len, batch, d_model
        #src = (src.shape[0],batch,src.shape[2])
        #print('src shape before encoder:', src.shape)


        output = self.transformer_encoder(src)  # output: [sequence, batch, embedding dim.], (ex. [3000, 5, 512])
        #print("/n output from transformer encoder layer: [sequence, batch, embedding dim.], (ex. [3000, 5, 512])")
        #print(output.shape)

        #print("\n 5. self attention pooling")
        #print("changing shape for pooling layer to [Batch size, Embedding]")
        ## The output of the transformer module is a tensor of attention weights with the shape, [Sequence length, Batch size, Embedding].
        output = output.permute(1, 0, 2)
        #print("shape after changing output for self attention")
        #print(output.shape)
        ## The output must be transformed in some way such that it be fed into the classification head module, i.e. it needs to be transformed to the shape [Batch size, Embedding].

        ## The self attention pooling layer is applied to the output of the transformer module which produces an embedding that is a learned average of the features in the encoder sequence.
        output = self.self_att_pool(output)
        #print("output after self attention pooling [Batch size, Embedding]")
        #print(output.shape)
        ### ADD A MLP HERE ###
        # logits = out
        #print("\n 6. Applying regressor")
        out = self.mlp(output)
        #print(out)
        return out

