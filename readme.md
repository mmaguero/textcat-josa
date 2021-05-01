# JOSA ML trainer

Train JOSA (Jopara Sentiment Analysis) corpus with traditional machine learning algorithms.

## Install

### Virtualenv

First create a virtual environment in the root dir by running:

`python3 -m venv venv`

then activate the virtual env with

`source venv/bin/activate`

(to get out of the virtualenv, run `deactivate`)

### Dependencies

install all the dependencies with

`pip install -r requirements.txt`

also make sure to download nltk's corpus by running those line in python
interpreter:

```python
import nltk
nltk.download()
```

### Paths

- Corpus: `corpus-dir/ds/`
    - Files in `Corpus`: `sa3_train.txt`, `sa3_dev.txt`, `sa3_test.txt` (format: one line per tweet; tweet ||| class)
- Log: `log_dir`
- Model: `models`

## Train Unbalanced / Balanced corpus

```
cd src
python main.py "y" "corpus-dir/" "SVC" --train_cat > "log_dir/sa3_SVC`date '+%Y_%m_%d__%H_%M_%S'`.log"
python main.py "y" "corpus-dir/" "SVC" --train_cat --balanced > "log_dir/sa3_SVCBal`date '+%Y_%m_%d__%H_%M_%S'`.log"
python main.py "y" "corpus-dir/" "CNB" --train_cat > "log_dir/sa3_CNB`date '+%Y_%m_%d__%H_%M_%S'`.log"
python main.py "y" "corpus-dir/" "CNB" --train_cat --balanced > "log_dir/sa3_CNBBal`date '+%Y_%m_%d__%H_%M_%S'`.log"
```

## How do I cite this work?

Please, cite this paper [On the logistical difficulties and findings of Jopara Sentiment Analysis](https://code-switching.github.io/2021):

Marvin M. Agüero-Torales, David Vilares, Antonio G. López-Herrera (2021). On the logistical difficulties and findings of Jopara Sentiment Analysis. In Proceedings on *CALCS 2021 (co-located with NAACL 2021) - Fifth Workshop on Computational Approaches to Linguistic Code Switching*, to appear (June).

```
BibTeX format pending
```
