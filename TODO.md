# Parts of the project

A. Two different domains # PERSON C
    #RQ-a1: Is the balanced accuracy/fscore (depending on RQ-b1) for individual domain models better than the global model
    #RQ-a2 (if RQ-a1 successful): Can we build a good model (balanced accuracy/fscore) for predicting domain
    #RQ-a3 (if RQ-a2 successful): Same as RQ-a1 but Build 2 models using PREDICTED CLASS instead of actual class, is this model still better in bal-accu/f1 than global model?
    #RQ-a4: ...
-classify into two different domains, and then build two models to predict
-add a feature to it?
-#RESEARCH

-Domain Adversarial Neural Network - using gradient reversal to try to minimise impact of different classes (#RESEARCH)
-Person C fill out other parts of research and come up with own #RQ - NEED TO DO MORE RESEARCH


B. Feature Balancing (at least 3 different methods) # PERSON B
    #RQ-b1: does using balanced accu/f1 as our metric help with kaggle (careful don't do this too much - will overfit)
        -soln: 
    #RQ-b2: does optimising balanced accuracy/f1 in hp tuning improve current models
        -soln: not really
    #RQ-b3: on best non-deep ml model so far, is upsample/downsample an improvement
    #RQ-b4: on best non-deep ml model so far, is SMOTE an improvement?
    #RQ-b5: ... (other methods that person C comes up with)
-SMOTE?
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
        #RQ-d1: which models are good performers on either TFIDF or BOW (should confirm this after confirming #RQ-B1)
        #RQ-d2: is TFIDF or BOW or other path better? (don't have to answer early - can do whatever Engineering-ML Algorithm pair that works)
        #RQ-d3: does dimension reduction such as PCA or something else improve?
        #RQ-d4: does adding in sentence length as a variable improve?
        #RQ-d5: does changing the feature selection base model from XGB to RFR help?
        #RQ-d6: does z-score normalisation help?
        #RQ-d7: out of all models, which non-deep model (and experiment config) gives the BEST validation (or test) score?
    -Feature engineering (2 paths)
        -TFIDF
        -BOW
            -dimension reduction? (PCA or others)
        -#RESEARCH (GloVE etc but not really helpful)

        -add in sentence length?? (based on domain)
        -add in perplexity?

    -feature selection (2 paths) **NOTE: NOT IMPORTANT**
        -Ron's method RFR
        -Ron's method XGB
            -anyone else have any methods?
    
    -Normalisation? (2 paths) **NOTE: NOT IMPORTANT**
    
    -everything on sklearn + xgboost + catboost + lgbm + explianableboost + MLP + Transformer ~= 21 models 


Actual work
[X] MLP and Transformer: Ron
[] MLP and Transformer bug fix: Ron
[] W2V
[] Bert
[] DANN
[] Reverse DANN

[X] read paper (Ron and Anderson read)
[] roadmap confirmation
[] Agreement



ROADMAP
week 0 (holiday)
[X]Ron set up framework
[X] Everyone finds and reads paper on his domain

week 1
-Person B completes all basic machine learning experiments
-Person C finishes all feature balancing research and experiments
-Person A begins to experiment NLP 

-meeting on Tuesday/Friday (over Zoom) to discuss theory of NLP

week 2
-... (everyone works on NLP)
-begin compound experimetns
-begin to think about report

week 3
-report writing 

DUE on 26th (Friday)