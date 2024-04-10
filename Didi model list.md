| model | representation | label balancing method | data (domain) |
| --- | --- | --- | --- |
| LGBC | tfidf | upsample | global |
| LGBC | tfidf | downsample | global |
| LGBC | tfidf | smote | global |
| LGBC | tfidf | adasyn | global | 
| LGBC | tfidf | upsample | domain 1 |
| LGBC | tfidf | downsample | domain 1|
| LGBC | tfidf | smote | domain 1|
| LGBC | tfidf | adasyn | domain 1| 
| LGBC | tfidf | upsample | domain 2 |
| LGBC | tfidf | downsample | domain 2 | 
| LGBC | tfidf | smote | domain 2 |
| LGBC | tfidf | adasyn | domain 2 |
| LGBC | tfidf | no change | domain 1|
| LGBC | tfidf | no change | domain 2 |

- report: balanced accuracy (and f1) for val and test sets (same split every time)
    -don't need to put on kaggle, just fill in google sheet
- with domain 1 and domain 2 models, pls also get the two domain's val and test sets aggregated together and then report the overall statistics even though they are predicted via two different models

