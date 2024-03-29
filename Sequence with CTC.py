# 1 Setup and Hyper-parameters

from comet_ml import Experiment

experiment = Experiment(
    api_key="cPLOOFcoToQzYhxvPOwMoupAd",
    project_name="sequence-model",
    workspace="pradeep-manoharan"
)

import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import torchaudio
from torchaudio.datasets import LIBRISPEECH
from torch.utils.data import random_split
import torch.nn.functional as F
from comet_ml.integration.pytorch import log_model
import numpy as np

# Data Preparation
path = "/path/to/min_librispeech"

dataset = LIBRISPEECH(root="data", url='dev-clean', download=True)

# Data split

total_sample = len(dataset)

train_size = int(0.8 * total_sample)
test_size = total_sample - train_size

train_data, test_data = random_split(dataset, [train_size, test_size])

train_audio_transformation = nn.Sequential(torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
                                           torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
                                           torchaudio.transforms.TimeMasking(time_mask_param=35)
                                           )
valid_audio_transformation = torchaudio.transforms.MelSpectrogram()

char_map_str = """
 ' 0
  <SPACE> 1
 a 2
 b 3
 c 4
 d 5
 e 6
 f 7
 g 8
 h 9
 i 10
 j 11
 k 12
 l 13
 m 14
 n 15
 o 16
 p 17
 q 18
 r 19
 s 20
 t 21
 u 22
 v 23
 w 24
 x 25
 y 26
 z 27
 """

def avg_wer(wer_scores, combined_ref_len):
    return float(sum(wer_scores)) / float(combined_ref_len)


def _levenshtein_distance(ref, hyp):
    """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein disctance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other. We can naturally
    extend the edits to word level when calculate levenshtein disctance for
    two sentences.
    """
    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0,n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]


def word_errors(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in word-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Levenshtein distance and word number of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)

    edit_distance = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)


def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Levenshtein distance and length of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space == True:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)


def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Calculate word error rate (WER). WER compares reference text and
    hypothesis text in word-level. WER is defined as:
    .. math::
        WER = (Sw + Dw + Iw) / Nw
    where
    .. code-block:: text
        Sw is the number of words subsituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the reference
    We can use levenshtein distance to calculate WER. Please draw an attention
    that empty items will be removed when splitting sentences by delimiter.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Word error rate.
    :rtype: float
    :raises ValueError: If word number of reference is zero.
    """
    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case,
                                         delimiter)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    wer = float(edit_distance) / ref_len
    return wer


def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    """Calculate charactor error rate (CER). CER compares reference text and
    hypothesis text in char-level. CER is defined as:
    .. math::
        CER = (Sc + Dc + Ic) / Nc
    where
    .. code-block:: text
        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the reference
    We can use levenshtein distance to calculate CER. Chinese input should be
    encoded to unicode. Please draw an attention that the leading and tailing
    space characters will be truncated and multiple consecutive space
    characters in a sentence will be replaced by one space character.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Character error rate.
    :rtype: float
    :raises ValueError: If the reference length is zero.
    """
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,
                                         remove_space)

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    return cer


# Object creating for text transformation

class TextTransform:
    def __init__(self):

        self.char_map_str = char_map_str
        self.char_map = {}
        self.index_map = {}

        for line in char_map_str.strip().split("\n"):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch

        self.index_map[1] = " "

    def text_to_int(self, text):
        """ use  a charactor  map and covert into text to an integer sequence"""
        int_sequence = []

        for c in text:
            if c == " ":
                ch = self.char_map["<SPACE>"]


            else:
                ch = self.char_map[c]

            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ use a charactor map and covert integer labels into to an text sequence"""
        string = []

        for i in labels:
            string.append(self.index_map[i])
        return "".join(string).replace("", " ")


text_transform = TextTransform()


def data_processing(data, data_type="train"):
    spectrograms = []
    labels = []
    input_length = []
    label_length = []

    for (waveform, _, utterance, _, _, _) in data:
        if data_type == "train":
            spec = train_audio_transformation(waveform).squeeze(0).transpose(0, 1)
        else:
            spec = valid_audio_transformation(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)

        input_length.append(spec.shape[0] // 2)
        label_length.append(len(label))

        spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

        return spectrograms, labels, input_length, label_length


def avg_wer(wer_score, comblined_ref_len):
    return float(sum(wer_score)) / float(comblined_ref_len)


def uppper_case(element: list[str]) -> list[str]:
    return [i.upper() for i in element]


# Define the model
class CNNlayer(nn.Module):
    # Layer normalization build for CNN
    def __init__(self, n_feat):
        super().__init__()
        self.layer_norm = nn.LayerNorm(n_feat)

    def forward(self, x):
        # X (batch,channel,feature,time)
        x = x.transpose(2, 3).contiguous()  # (batch,channel,time,feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()  # (batch,channel,feature,time)


class ResidualCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, stride, dropout, n_feat):
        super().__init__()
        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernal_size, stride, padding=kernal_size // 2)
        self.cnn2 = nn.Conv2d(in_channels, out_channels, kernal_size, stride, padding=kernal_size // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.normal_layer1 = CNNlayer(n_feat)
        self.normal_layer2 = CNNlayer(n_feat)

    def forward(self, x):
        residual = x  # (batch,channel,feature,time)

        x = self.normal_layer1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.normal_layer2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual

        return x  # (batch,channel,feature,time)


class BidirectionalGRU(nn.Module):
    def __init__(self, rnn_dim, hidden_size, droupout, batch_first):
        super().__init__()
        self.BiGRU = nn.GRU(input_size=rnn_dim, hidden_size=hidden_size, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.droupout = nn.Dropout(droupout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x,_ = self.BiGRU(x)
        x = self.droupout(x)
        return x


class SpeechRecogitionModel(nn.Module):

    def __init__(self, n_cnn_layer, n_rnn_layer, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super().__init__()
        n_feats = n_feats // 2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3 // 2)

        # n residual cnn layer with filter 32

        self.rescnn = nn.Sequential(*[ResidualCNN(32, 32, kernal_size=3, stride=1, dropout=dropout, n_feat=n_feats)
                                      for _ in range(n_cnn_layer)
                                      ])

        self.fully_connected = nn.Linear(n_feats * 32, rnn_dim)

        self.birnn_layer = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i == 0 else rnn_dim * 2, hidden_size=rnn_dim, droupout=dropout,
                             batch_first=i == 0)
            for i in range(n_rnn_layer)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim * 2, rnn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])

        x = x.transpose(1, 2)

        x = self.fully_connected(x)
        x = self.birnn_layer(x)
        x = self.classifier(x)
        return x


# Define the Evaluation matrics of the model
def GreedyDecoder(output, labels, labels_length, blank_length=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decoder = []
    target = []

    for i, arg in enumerate(arg_maxes):
        decode = []
        target.append(text_transform.int_to_text(labels[i][:labels_length].tolist()))
        for j, index in enumerate(arg):
            if index != blank_length:
                if collapse_repeated and j != 0 and index == arg[j - 1]:
                    continue
                decode.append(index.item())
        decoder.append(text_transform.int_to_text(decode))

    return decoder, target


# Training the model

class IterMeter(object):
    # Keep the track of the total iterataion

    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 0

    def get(self):
        return self.val


def train(model, device, train_loader, criterion, optimizer, sheduler, epoch, iter_meter, experiment):
    model.train()
    data_len = len(train_data.dataset)
    with experiment.train():
        for batch_idex, _data in enumerate(train_loader):
            spectrogram, labels, input_length, label_length = _data
            optimizer.zero_grad()

            output = model(spectrogram)  # (batch,time,n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time,batch,n_class)
            loss = criterion(output, labels, input_length, label_length)
            loss.backward()

            experiment.log_metric('loss', loss.item(), step=iter_meter.get())
            experiment.log_metric('learning_rate', sheduler.get_lr(), step=iter_meter.get())

            optimizer.step()
            sheduler.step()
            iter_meter.step()

            if batch_idex % 100 == 0 or batch_idex == data_len:
                print('Train Epoch: {} [{}{}({:.0f}%)]\tloss: {:.6f}'.format(epoch, batch_idex * len(spectrogram),
                                                                             data_len,
                                                                             100 * batch_idex / len(train_loader),
                                                                             loss.item()))


def test(model, device, test_loader, criterion, epoch, iter_meter, experiment):
    print('\nEvaluating')
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    with experiment.test():
        with torch.no_grad():
            for I, _data in enumerate(test_loader):
                spectrogram, label, input_length, label_lengh = _data

                output = model(spectrogram)
                output = F.log_softmax(output, dim=2)
                output = output.transpose(0, 1)  # (time,batch, n_class)
                loss = criterion(output, label, input_length, label_lengh)
                test_loss += loss.item() / len(test_loader)

                decode_pred, decord_target = GreedyDecoder(output.transpose(0, 1), label, label_lengh)

                for j in range(len(decode_pred)):
                    test_cer.append(cer(decode_pred[j], decord_target[j]))
                    test_wer.append(wer(decode_pred[j]), decord_target[j])

                ave_char = sum(test_cer) / len(test_cer)
                ave_wer = sum(test_wer) / len(test_wer)
                # track metrics

                experiment.log_metric("test_loss", test_loss, step=iter_meter.get())
                experiment.log_metric('Cer', ave_char, step=iter_meter().get())
                experiment.log_metric('wer', ave_wer, step=iter_meter().get())

                print('Test set: Average loss: {:.4f},Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss,
                                                                                                       ave_char,
                                                                                                       ave_wer))


def main(learning_rate=5e-4, batch_size=20,
         epochs=10, experiment=Experiment(api_key="cPLOOFcoToQzYhxvPOwMoupAd", disabled=True)):
    hparams = {'n_cnn_layers': 3,
               'n_rnn_layers': 5,
               'rnn_dim': 512,
               'n_class': 29,
               'n_feats': 128,
               'stride': 2,
               'dropout': 0.1,
               'learning_rate': learning_rate,
               'batch_size': batch_size,
               'epochs': epochs}

    experiment.log_parameters(hparams)

    use_cuda = torch.cuda.is_available()

    torch.manual_seed(7)
    device = torch.device('cuda' if use_cuda else "cpu")

    if not os.path.isdir('./data'):
        os.makedirs('./data')

    kwargm = {'num_works': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True,
        collate_fn=lambda x: data_processing(x, 'train'), **kwargm)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=batch_size, shuffle=True,
        collate_fn=lambda x: data_processing(x, data_type='test'), **kwargm)

    model = SpeechRecogitionModel(hparams['n_cnn_layers'],
                                  hparams['n_rnn_layers'], hparams['rnn_dim'], hparams['n_class'],
                                  hparams['n_feats'], hparams['stride'], hparams['dropout'])

    print(model)
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    criterion = nn.CTCLoss(blank=28).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), learning_rate)
    sheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'],
                                                   steps_per_epoch=int(len(train_data)), epochs=epochs,
                                                   anneal_strategy='linear')

    iter_meter = IterMeter()


    for epoch in range(1, epochs + 1):
        train(model, device, train_loader,
              criterion, optimizer, sheduler,
              epoch, iter_meter, experiment)

        test(model, device, test_loader,
             criterion, epoch, iter_meter, experiment)


if __name__ == "__main__":
    learning_rate = 5e-4
    batch_size = 10
    epochs = 10
    libri_train_set = "train-clean-100"
    libri_test_set = "test-clean"

    main(learning_rate, batch_size, epochs, experiment)

    experiment.end()
