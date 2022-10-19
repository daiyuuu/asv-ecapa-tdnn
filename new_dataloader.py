'''
for training noisy audio data load
'''
import glob, numpy, os, random, soundfile, torch
from tkinter import N

class new_train_loader(object):
    def __init__(self, train_list, train_path, **kwargs):
        # Load data & labels
        self.data_list  = []
        self.data_label = []
        lines = open(train_list).read().splitlines()
        dictkeys = list(set([x.split()[0] for x in lines]))  # first row of train list(id01794 somewhat)
        dictkeys.sort()
        dictkeys = { key : ii for ii, key in enumerate(dictkeys) }  # put label to dict, assign a label for sorted id (id00012 is 0)
        for index, line in enumerate(lines):
            speaker_label = dictkeys[line.split()[0]]   # find the wav corresponding to the speaker label
            file_name     = os.path.join(train_path, line.split()[1])
            self.data_label.append(speaker_label)
            self.data_list.append(file_name)

    def __getitem__(self, index):
        # Read the utterance and randomly select the segment 
        audio, sr = soundfile.read(self.data_list[index]) 
        return torch.FloatTensor(audio), self.data_label[index]

    def __len__(self):
        return len(self.data_list)