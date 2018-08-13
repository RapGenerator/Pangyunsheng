﻿# Pangyunsheng  
## Code
* RapGenerator文件夹下的模型是第二版的seq2seq（已弃用）
* RapGeneratorV2文件夹下的模型是第三版的seq2seq（已弃用）
* Seq2Seq(baseline)是Seq2Seq的baseline版本，主要包含以下mode：  
  **Data loading mode:** reversed  
  **Training mode:** ground truth  
  **Training mode:** scheduled sampling(teacher forcing)  
  **Predicting mode:** greedy  
  **Predicting mode:** beam search  
  **Attention mode:** bahdanau attention  
 * Seq2SeqV2在Seq2Seq(baseline)的基础上增加了双向encoder
 * SkipThought文件是SkipThought的第一版实现，在Seq2Seq(baseline)的基础上增加了一个decoder
## Diary
### 2018-08-07
* Modified "RapGeneratorV2" file to make it can decode more than one line Rap lyrics, but because there are too many "UNK" symbols, our model(refer to RapGeneratorV2) can only decode about four to six lines without symbol "UNK" at one time. Our final seq2seq baseline model "Seq2Seq(baseline)" can decode any number of lines of lyrics.  
* Readed paper "Skip-Thought Vectors" and try to reproduce the model. Until now, I have finished the training phase of this model and try to train it with my GPU. I'm planning to finish the predicting phase in tomorrow.  
* Shijun, Weiwen, Mengxin, Ruichang, Ziqun and Juecen shared their ideas about how to improve our model in the next work. I think our coding group should divide into two or three groups to implement their ideas, we can not judge which one is good before we see their effects.  
* Next work may need to implement some papers' idea from BaiDu, HaGongda.
### 2018-08-08
* Finished "Skip-Thought Vectors" model and generated some lines with this model, the effect is a little improved, The context between the sentence and the sentence is more closely related.
* Zi qun added Bidirectional RNN encoder based on our baseline model "Seq2Seq", we are going to merge this model with Skip-Thought model.
* We talked about how to generate rap lyrics according to some key words, we want to try to generate a line according to a key word, and generate more lines according to this line.
* Weiwen will extract a key word from a line of rap lyrics for traning sets, and we will use this training sets to train our model.
### 2018-8-09
* Talked to our teacher about how to generate rap lyrics based on subjects, and how to make rap lyrics rhyme. On my conclusion, I will do some work next days below:  
&emsp;1.Try datasets with different rhyming rules.  
&emsp;2.Try to make our decoder reverse output.  
&emsp;3.Train a model that can generate sentences from topics  
&emsp;4.Train a new model based on Skip-Thought model that can classifier current sentence and generate next sentence.  
&emsp;5.Modify beam search function so that we can select a word whose rhyme is same as the last word of last sentence.  
&emsp;6.Modify our model so that it can generate sentences of any length we want.
* Until now, I have finished the model mentioned in the third article above, but I haven't got our datasets. I'm going to train it tomorrow. And I have finished the model mentioned in the fourth article above, only a LSTM sequence model to generate a sentence based on a subject.
### 2018-8-10
* Read paper from HaGongda. In the afternoon, we visit the company of MeiTuan in BeiJing, and I do noing.  
### 2018-8-11
* Read some basic knowledge about GAN
* Finished the code of model "SkipThoughtCG"(I named it), it looks like a model of SkipThought, but we replace a decoder of a classifier.  
* Read paer from HaGongda and check it codes on github.
### 2018-8-12
* Trained the model "SkipThoughtCG" and finished the inference phase of this model. 
* Read some basic knowledge about RL and GAN in text generation task.
* ZiQun modified the parameter of "num_layers" as 2, and modified the type of bidirectional RNN, that made a good results. I follow these parameters and change my model, I'm going to see my model's effect.
### 2018-8-13
* Used the model of "SkipThoughtCG" to generate sentences, but it didn't work well, our data is very bad.  
* Learned some knowledge about SeqGAN and RL in text generation task.  
* Finshed codes of paper of HaGongda(named "MTA_LSTM"). And try to train it with their datasets. I'm going to check the result of that model.  
* In next two days, the main thing is to get more data with subjects, and the final model may from "SkipThoughtCG" and "MTA_LSTM". I'm going to use the two models to generate sentences with subjects.
