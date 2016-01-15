# Morphological Reinflection

This repository contains ongoing work on morphological reinflection. More documentation to come...

### Data pre/post-processing 
Look at the scripts under the utils directory. They can be used to prepare data from [SIGMORPHON 2016](http://ryancotterell.github.io/sigmorphon2016/), and to post-process experiment output so be able to use the official evaluation scripts there.

### Experiments
```run.lua``` controls the settings. Run it like this:
```
 th run.lua -wordsTrainFile <words-train> -wordsTestFile <words-test> -lemmasTrainFile <lemmas-train> -lemmasTestFile <lemmas-test> -featsTrainFile <feats-train> -featsTestFile <feats-test> -alphabet <alphabet>
```
