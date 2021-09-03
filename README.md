基于非结构化文本的检务公开自动问答系统研究与实现
===========================

###########环境依赖
pytorch-pretrained-bert 0.4.0  
scikit-learn 0.23.2  
torch 1.8.1  
torchvision 0.9.1  
jieba 0.42.1  
/单片 NVIDIA RTX 3090

###########运行说明
data/chinese\_L-12\_H-768\_A-12中的pytorch\_model.bin文件直接去bert官网下载bert-base-chinese模型或者用哈工大的bert-wwm模型
data目录结构及命名：  
--chinese\_L-12\_H-768\_A-12 模型文件  
--search.dev.json  
--search.train.json  
--search.test.json  
--zhidao.dev.json  
--zhidao.train.json  
--zhidao.test.json  
生成的文件：  
--dev-v2.0.json  
--dureader-dev.json  
--search.dev.local.json  
--zhidao.dev.local.json  
--search\_dev\_rank\_output.json  
--zhidao\_dev\_rank\_output.json  
--result.json  

￼
dureader数据集使用预处理后的dureader2.0版本  

optiBERT\_Dureader/RRBERT/data/build\_local\_data.py 自定义从数据集中取出多少样本组成新数据集  

optiBERT\_Dureader/RRBERT/retriever/prepare.py 准备被检索文档的数据集：optiBERT\_Dureader/RRBERT/retriever/retriever\_data: dev.tsv/train.tsv  

optiBERT\_Dureader/RRBERT/retriever/run\_classifier.py 
参数：--data\_dir ./retriever\_data/ --bert\_model ../data/chinese\_L-12\_H-768\_A-12/ --task\_name MRPC --output\_dir ./retriever\_output --do\_train --do\_eval --train\_batch\_size 8  
训练文档选择模型  
do\_train要把retriever\_output清空  

retriever/bert\_rank.py   
--test\_file  
../data/zhidao.dev.json  
--output\_path  
../data/zhidao\_dev\_rank\_output.json  
使用训练好的模型筛查备选文档 运行两次 第二次把zhidao改为search  

optiBERT\_Dureader/RRBERT/data/prepare\_squad,py 把第一步抽取出的local样本融合(search\zhidao)，转成squad格式存入训练时用的数据集  



run\_dureader.py --bert\_model ../data/chinese\_L-12\_H-768\_A-12 --do\_train --train\_file ./data/train-v2.0.json --train\_batch\_size 8 --learning\_rate 3e-5 --num\_train\_epochs 3.0 --max\_seq\_length 384 --doc\_stride 128 --output\_dir ./reader\_output
predict\_dureader.py  
--bert\_model  
./data/chinese\_L-12\_H-768\_A-12/  
--bin\_path  
./reader/reader\_output/pytorch\_model.bin  
--predict\_file  
./reader/dev-v2.0.json  
--output  
./test1\_output  

答案重排序：  
prepare\_reranking.py 将阅读理解模型输出的预测候选答案准备成答案重排序的数据集  
AnswerReranking/prepare.py 分为正负例输入进二分类模型  
AnswerReranking/run\_classifier.py 第一次不要do\_eval 第二次把vocab.txt复制到output文件夹里去除do\_train后do\_eval  
--task\_name QNLI  
--do\_train  
--do\_eval  
--do\_lower\_case  
--data\_dir ./data/QNLI  
--bert\_model ./data/chinese\_L-12\_H-768\_A-12/  
--max\_seq\_length 128  
--train\_batch\_size 32  
--learning\_rate 2e-5  
--num\_train\_epochs 3.0  
--output\_dir ./data/QNLI/output   

训练：run\_squad.py   
--bert\_model ./data/chinese\_L-12\_H-768\_A-12  squad\_model bert\_deep --do\_train  --do\_predict --train\_file ./data/train-v2.0.json --predict\_file ./data/dev-v2.0.json --train\_batch\_size 8 --learning\_rate 3e-5 --num\_train\_epochs 2.0  --max\_seq\_length 384 --doc\_stride 128 --version\_2\_with\_negative --output\_dir ./output  
英文数据集要修改模型，增加--do\_lower\_case  
测试： run\_squad.py  
  --bert\_model ./data/chinese\_L-12\_H-768\_A-12  
  --squad\_model bert\_qanet  
  --do\_predict   
  --predict\_file ./data/tdev-v2.0.json \  
  --max\_seq\_length 384   
  --doc\_stride 128   
  --version\_2\_with\_negative   
  --output\_dir ./output/  

evaluate.py ./data/dev-v2.0.json   
    ./test1\_output/predictions.json   
    --out-file ./output/dev\_eval.json  
    --na-prob-file ./output/null\_odds.json  
    --out-image-dir ./output/charts 


prepare\_answer.py



