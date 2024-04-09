# Parts of the project

A. Two different domains # PERSON C
    #RQ-a1: Is the balanced accuracy/fscore (depending on RQ-b1) for individual domain models better than the global model
        soln: currently. 
    #RQ-a2 (if RQ-a1 successful): Can we build a good model (balanced accuracy/fscore) for predicting domain
        soln: yes, 99% accuracy
    #RQ-a3 (if RQ-a2 successful): Same as RQ-a1 but Build 2 models using PREDICTED CLASS instead of actual class, is this model still better in bal-accu/f1 than global model?
    #RQ-a4: ...
-classify into two different domains, and then build two models to predict
-add a feature to it?
-#RESEARCH

-Domain Adversarial Neural Network - using gradient reversal to try to minimise impact of different classes (#RESEARCH)
-Person C fill out other parts of research and come up with own #RQ - NEED TO DO MORE RESEARCH


B. Feature Balancing (at least 3 different methods) # PERSON B
    #RQ-b1: does using balanced accu/f1 as our metric help with kaggle (careful don't do this too much - will overfit)
        -soln: balanced accuracy sort of 
    #RQ-b2: does optimising balanced accuracy/f1 in hp tuning improve current models
        -soln: not really for non deep; deep learning inconclusive
    #RQ-b3: on best non-deep ml model so far, is upsample/downsample an improvement
    #RQ-b4: on best non-deep ml model so far, is SMOTE an improvement?
    #RQ-b5: adasyn
    #RQ-b6: ... (other methods that person C comes up with)
-Upsample/Downsample
-#RESEARCH

-in tuning, optimise for balanced acuracy or F1


MODELLING PATH
Deep Learning vs Machine Learning
C. -Deep Learning   #PERSON A, PERSON C (after finish Feature Balancing)
        #RQ-c1: can we train our own W2V 
            -soln: Yes
        #RQ-c2: does our own W2V work better than training embedding WITH the base model
        #RQ-c3: which DL is the best
        #RQ-c4: does different feature creating (e.g. padding) help?
        #RQ-c5: does balanced accuracy loss help (weighted)
        #RQ-c6: does Domain Adversarial Neural Network help
        #RQ-c7: DANN, or reverse DANN
    -Embedding (2 paths)
        -self pre-train W2V? (单独训练embedding)
        -self pre-train BERT
        [X] pytorch embedding... 

    -Model 
        [X] LSTM 
        [X] Transformers (BERT)
        Other models based on local global


D. -Typical Machine Learning # PERSON B
        #RQ-d1: burstiness
        #RQ-d3: perplexity
        #RQ-d4: does adding in sentence length as a variable improve?
        #RQ-d7: out of all models, which non-deep model (and experiment config) gives the BEST validation (or test) score?
    -Feature engineering (2 paths)
    
    -everything on sklearn + xgboost + catboost + lgbm + explianableboost + MLP + Transformer ~= 21 models 


Actual work
[X] MLP and Transformer: Ron
[] W2V
[] Bert
[?] DANN
[] Reverse DANN
[] MLP and Transformer bug fix: Ron

[X] read paper (Ron and Anderson read)
[X] roadmap confirmation
[X] Agreement



ROADMAP
week 0 (holiday)
[X]Ron set up framework
[X] Everyone finds and reads paper on his domain

week 1
-Person B (Didi) completes all basic machine learning experiments
    -adasyn, smote, upsample, downsample on:
        -full dataset LGBM (Ron)
        -separate domain 2x LGBM/XGB
    -separate domain LGBM/XGB to compare

    -add features burstiness, perplexity and full token length as features
        -no upsample (上面几个重做)

-Person C (Anderson) finishes all feature balancing research and experiments
    -w2v
    -bert running
    -dann
    -transformer 验证

-Person A (Ron) begins to experiment NLP 
    -bert
    -dann
    -transformer 验证

-meeting on Tuesday/Friday (over Zoom) to discuss theory of NLP

week 2
-... (everyone works on NLP)
-begin compound experimetns
-begin to think about report

week 3
-report writing 

DUE on 26th (Friday)