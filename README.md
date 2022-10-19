# Instruction
This repository contains a few modifications to the IO bottleneck encountered with [ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN) data loading. After modification, each epoch training time grows up to about 20 minutes.

# The main changes (Data preparation)
Added a script ./preprocess/gen_audio.py to augument the training data in advance, the dataloader section has been modified accordingly. This allows the model to reduce the CPU processing time during the dataloader stage. 

# Run the training code
Before running the trainECAPAModel.py, please run the ./preprocess/gen_audio.py to preprocess your data.Please refer to other more detailed steps [ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN).


