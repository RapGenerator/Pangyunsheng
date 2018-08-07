# Pangyunsheng
Seq2Seq文件夹下的模型是第一个版本的seq2seq，只加了Attention，且数据集不是rap数据集  
RapGenerator文件夹下的模型是第二版的seq2seq，在第一版seq2seq基础上加入了beam serach，加入了GRUCell作为可选的cell_type，数据集换成了rap数据集
RapGeneratorV2文件夹下的模型是第三版的seq2seq

Seq2Seq（终极版）是最终的Seq2Seq model，以后的改进都基于这个版本的Seq2Seq
## Diary
### 2018-08-07
* Modified "RapGeneratorV2" file to make it can decode more than one line Rap lyrics, but because there are too many "UNK" symbols, our model(refer to RapGeneratorV2) can only decode about four to six lines without symbol "UNK" at one time. Our final seq2seq baseline model "Seq2Seq（终极版)" can decode any number of lines of lyrics.  
* Readed paper "Skip-Thought Vectors" and try to reproduce the model. Until now, I have finished the training phase of this model and try to train it with my GPU. I'm planning to finish the predicting phase in tomorrow.  
* Shijun, Weiwen, Mengxin, Ruichang, Ziqun and Juecen shared their ideas about how to improve our model in the next work. I think our coding group should divide into two or three groups to implement their ideas, we can not judge which one is good before we see their effects.  
* Next work may need to implement some papers' idea from BaiDu, HaGongda.
