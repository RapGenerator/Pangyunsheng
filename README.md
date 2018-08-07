# Pangyunsheng  
## Code
* RapGenerator文件夹下的模型是第二版的seq2seq（已弃用）
* RapGeneratorV2文件夹下的模型是第三版的seq2seq（已弃用）
* Seq2Seq（终极版）是最终版的Seq2Seq，主要包含以下mode：  
  **Data loading mode:** reversed  
  **Training mode:** ground truth  
  **Training mode:** scheduled sampling(teacher forcing)  
  **Predicting mode:** greedy  
  **Predicting mode:** beam search
  **Attention mode:** bahdanau attention
## Diary
### 2018-08-07
* Modified "RapGeneratorV2" file to make it can decode more than one line Rap lyrics, but because there are too many "UNK" symbols, our model(refer to RapGeneratorV2) can only decode about four to six lines without symbol "UNK" at one time. Our final seq2seq baseline model "Seq2Seq（终极版)" can decode any number of lines of lyrics.  
* Readed paper "Skip-Thought Vectors" and try to reproduce the model. Until now, I have finished the training phase of this model and try to train it with my GPU. I'm planning to finish the predicting phase in tomorrow.  
* Shijun, Weiwen, Mengxin, Ruichang, Ziqun and Juecen shared their ideas about how to improve our model in the next work. I think our coding group should divide into two or three groups to implement their ideas, we can not judge which one is good before we see their effects.  
* Next work may need to implement some papers' idea from BaiDu, HaGongda.
