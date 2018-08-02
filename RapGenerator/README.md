这个版本的seq2seq最大的改动是加了beam search，还加入了GRUCell作为可选cell  
  
  
此外，对rap数据集做了处理，rap数据集中共有70000+条数据，将其分成了sources和targets，分好的数据保存在data文件夹下，分法是所有的奇数行组成sources，
所有的偶数行组成targets，必须保证sources和targets的数量相等  
  
  
在data_helpers.py文件中也有较大修改，首先对所有sources和targets做了分词，然后统计了每个词出现的次数，共有4W+个非重复词，构建字典时去掉了使用次数为1的词，去掉后还剩下25000左右的非重复词，即字典的大小为25000（具体数值忘了，大概25000左右）  
  
  
model.py文件中主要是加了beam search，具体改动查看代码  
  
    
    问题：1.原本有4W+个非重复字，但是频率为1的就有2W多，全部去掉后会不会有影响？  
         2.2W+的字典大小算不算大？ 后面构建embedding矩阵的时候会不会太大？embedding矩阵的大小为：字典大小*embedding_size
