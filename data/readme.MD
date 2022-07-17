## Existing data source


### notebooks
`nb/data_source.ipynb`: a notebook explain all the existing data source


### COVID19 disinfo data
Data under this folder is from [this paper](http://arxiv.org/abs/2104.05745)

`data/covid19_disinfo/covid19_disinfo_binary_english_train_old.tsv`: has 869 English tweets. The labels are **binary** labels from these questions

- Q1: Does the tweet contain a verifiable factual claim?

- Q2: To what extent does the tweet appear to contain false information?

- Q3: Will the tweet have an effect on or be of interest to the general public?

- Q4: To what extent is the tweet harmful to the society, person(s), company(s) or product(s)?

- Q5: Do you think that a professional fact-checker should verify the claim in the tweet?

- Q6: Is the tweet harmful to the society and why?

- Q7: Do you think that this tweet should get the attention of a government entity?


### COVID19 infordemic data
Data under this folder is from [this paper](http://arxiv.org/abs/2007.07996)

`data/covid19_infordemic/covid19_infodemic_english_data.tsv`: has 504 English tweets. The labels are **full answers** from the questions above.



