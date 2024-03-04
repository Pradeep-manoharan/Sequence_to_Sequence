import torch
import torch.nn as nn
import torch.optim as optim
import random


class seq2seqModel(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.encoder = nn.LSTM(input_size,hidden_size)
        self.decoder = nn.LSTM(hidden_size,hidden_size)
        self.fc = nn.Linear(hidden_size,output_size)

    def forward(self,input_sequence):
        Encoder,_ = self.encoder(input_sequence)
        decoder,_ = self.decoder(Encoder)
        ctc_fully_connected = self.fc(decoder)

        return ctc_fully_connected


input_size = 10
hidden_size= 32
output_size = 5
batch_size = 5
sequence_lengh = 5


input = torch.rand(batch_size,sequence_lengh,input_size)


model = seq2seqModel(input_size,hidden_size,output_size)
output = model(input)

ctc_loss  = nn.CTCLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)


print(input.shape)
print(output.shape)

optimizer.zero_grad()



