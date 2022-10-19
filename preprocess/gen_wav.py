'''
The original speech data is enhanced in advance to avoid io bottleneck problem during model training
'''

import glob, numpy, os, random, soundfile, torch, argparse, wavfile, librosa
from scipy import signal
import sys,time,tqdm

class Data_process(object):
    def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, save_path):
        self.train_path = train_path
        self.save_path = save_path
        self.num_frames = num_frames
        # Load and configure augmentation files
        self.noisetypes = ['noise','speech','music']
        self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
        self.noiselist = {}
        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))
        for file in augment_files:
            if file.split('/')[-4] not in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)
        self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))
        # Load data & labels
        self.data_list  = []
        self.save_dir = []
        lines = open(train_list).read().splitlines()
        for index, line in enumerate(lines):
            save_dir = os.path.join(save_path, (line.split()[1]).split('/')[0], (line.split()[1]).split('/')[1])
            file_name = os.path.join(train_path, line.split()[1])
            self.save_dir.append(save_dir)
            self.data_list.append(file_name)
        return self.save_audio(self.data_list, self.save_dir,self.save_path)
        
    def save_audio(self, data_list, save_dir, save_path):
        # mkdir save path
        save_dir = list(set(save_dir))
        for i in range(len(save_dir)):
            if not os.path.exists(save_dir[i]):
                os.makedirs(save_dir[i])  
        # Read the utterance and randomly select the segment
        # for index in range(len(data_list)):
        for index in tqdm.tqdm(enumerate(data_list), total = len(data_list)):
            audio, sr = soundfile.read(data_list[index])
            length = self.num_frames * 160 + 240
            if audio.shape[0] <= length:
               shortage = length - audio.shape[0]
               audio = numpy.pad(audio, (0, shortage), 'wrap')
            start_frame = numpy.int64(random.random()*(audio.shape[0]-length))
            audio = audio[start_frame:start_frame + length]
            audio = numpy.stack([audio],axis=0)
           # Data Augmentation
            augtype = random.randint(0,5)
            if augtype == 0:   # Original 
               audio = audio
            elif augtype == 1: # Reverberation
               audio = self.add_rev(audio)
            elif augtype == 2: # Babble
               audio = self.add_noise(audio, 'speech')
            elif augtype == 3: # Music
                audio = self.add_noise(audio, 'music')
            elif augtype == 4: # Noise
                audio = self.add_noise(audio, 'noise')
            elif augtype == 5: # Television noise
                audio = self.add_noise(audio, 'speech')
                audio = self.add_noise(audio, 'music')
            audio_path = os.path.join(save_path, data_list[index].split('/')[-3], data_list[index].split('/')[-2], data_list[index].split('/')[-1])
            soundfile.write(audio_path, audio[0,:], sr, 'PCM_16')
        
    def add_rev(self, audio):
        rir_file    = random.choice(self.rir_files)
        rir, sr     = soundfile.read(rir_file)
        rir         = numpy.expand_dims(rir.astype(numpy.float),0)
        rir         = rir / numpy.sqrt(numpy.sum(rir**2))
        return signal.convolve(audio, rir, mode='full')[:,:self.num_frames * 160 + 240]

    def add_noise(self, audio, noisecat):
        clean_db    = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 
        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
        # noiselist   =  numpy.random.choice(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            length = self.num_frames * 160 + 240
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = numpy.int64(random.random()*(noiseaudio.shape[0]-length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = numpy.stack([noiseaudio],axis=0)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4) 
            noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
        return noise + audio


if __name__ == "__main__":
    # define the argument
    train_list = '/media/dell/203A28373A280C7A/xlt/SASVC2022_Baseline-main/ECAPATDNN/data/train_list.txt'  # path of training data list (download by https://mm.kaist.ac.kr/datasets/voxceleb/)
    train_path = '/media/dell/203A28373A280C7A/xlt/voxceleb/dev/aac'          # path of original training data 
    musan_path = '/media/dell/203A28373A280C7A/xlt/SASVC2022_Baseline-main/ECAPATDNN/data/musan_split'    # path of musan (musan_split dataset download by https://github.com/clovaai/voxceleb_trainer)
    rir_path = '/media/dell/203A28373A280C7A/xlt/SASVC2022_Baseline-main/ECAPATDNN/data/RIRS_NOISES/simulated_rirs' # path of rirs (rirs dataset download by https://github.com/clovaai/voxceleb_trainer)
    num_frames = 200                                                                 # Duration of the input segments 
    save_path = '/media/dell/60480DB3480D88CC/voceleb_test'   # path to save processed audio 
    Data_process(train_list, train_path, musan_path, rir_path, num_frames, save_path) 
