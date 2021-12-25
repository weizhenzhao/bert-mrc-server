# bert-mrc-server
machine reading comprehension

1、first run paragraph_extraction.py to find the most related paragraph and compute related score.
   on the condition that you've already got the fake_answer from the dureader data set,
   or you should run the preprocess.py to find the fake_answer then run paragraph_extraction.py
2、second run run_squad.py to make the data into standardization format for trainning and testting.
3、thirdly run train.py to train the model
4、finally run predicting.py to predict the result.
   
