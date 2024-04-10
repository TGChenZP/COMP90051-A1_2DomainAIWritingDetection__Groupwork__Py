Built up LSTM and BERT by training up own embeddings

-Problem: LSTM + BERT not learning; neither is BoW and TFIDF Transformer and TFIDF MLP
    -hypothesis: underfitting or overfitting; learning embedding is too much with too few signals; can't learn both DL and embedding together.

    -attempts: reduced the length of tokens for each sentence (worked)


    -EDA: 
        -domain 2 has less total tokens but average sentence is longer
        -for both domains, 50% of the tokens have only 1 observation, 75% have <5. only 38338 tokens more than 1 observation, 13677 more than 10 obs
        -for domain 1, 50% of the tokens have only 1 observation, 75% have <6.
        only 12535 tokens more than 1 observation, 4126 more than 10 obs
        -for domain 2, 50% of tokens ahve only 1 observation, 75% have <4. only 32058 tokens more than 1 observation, 11404 more than 10 obs

    -ideas:

        -freeze the embedding and see if its stable

        -shorter sentences, and map the remaining sentence (in train set) as new samples
            -rationale: avoids short sentences being padded
            -rationale: we are likely more focussed on use of certain words and grammar anyway
                -ELMO style becuase we are focussing on syntax
        
        -only keep high frequency words for embedding
            -from observing data: 
            -rationale: we are likely more focussed on use of certain words and grammar anyway

        -add a w2v loss into the training process: one round of w2v, one round of classification.

        -pretrain w2v
            -problem: will likely only be able to train 1 dimension


-BoW/TFIDF experiments: check loss and how template is setup

-Problem: validation loss increasing significantly despite accuracy going up
    -ideas:
        -change up the saving mechanism to save best accuracy/balanced accuracy/f1
        -test out whether we are doing [0, 1] wrong.


-ideas:
    -just use pretrained model embedding
        -freeze embedding
        -train both
    -just use pretrained bert 
        -freeze bert
        -train both
    -use both pretrained
        -train both
        -train each individually