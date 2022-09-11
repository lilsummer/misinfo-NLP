## I. Data sources used


### Notebooks
`nb/data_source.ipynb`: EDA notebook explaining data sources being used (currently only has analyses for main sources)

### a. Main sources

#### 1. COVID19 disinfo data
Data under this folder is from [this paper](http://arxiv.org/abs/2104.05745)

`data/covid19_disinfo/covid19_disinfo_binary_english_train_old.tsv`: has 869 English tweets. The labels are **binary** labels from these questions

- Q1: Does the tweet contain a verifiable factual claim?

- Q2: To what extent does the tweet appear to contain false information?

- Q3: Will the tweet have an effect on or be of interest to the general public?

- Q4: To what extent is the tweet harmful to the society, person(s), company(s) or product(s)?

- Q5: Do you think that a professional fact-checker should verify the claim in the tweet?

- Q6: Is the tweet harmful to the society and why?

- Q7: Do you think that this tweet should get the attention of a government entity?


#### 2. COVID19 infordemic data
Data under this folder is from [this paper](http://arxiv.org/abs/2007.07996): [Github](https://esoc.princeton.edu/publications/esoc-covid-19-misinformation-dataset)

`data/covid19_infordemic/covid19_infodemic_english_data.tsv`: has 504 English tweets. The labels are **full answers** from the questions above.


### b. Additional sources

#### 1. New data from individual users: manually labeled by Cassie and Linh

`data/data_new/`: these are raw data pulled from twitter API

`data/manual_label/`: include a few labels that we added manually 

#### 2. Data from [CONSTRAINT Workshop Competition](https://constraint-shared-task-2021.github.io/)

Example papers using this data:

- [Fighting an Infodemic: COVID-19 Fake News Dataset](https://arxiv.org/abs/2011.03327)
- [A Heuristic-driven Ensemble Framework for COVID-19 Fake News Detection](https://arxiv.org/abs/2101.03545): [Codes and Data](https://github.com/diptamath/covid_fake_news)

#### 3. Data from [Cross-SEAN: A cross-stitch semi-supervised neural attention model for COVID-19 fake news detection](https://www.sciencedirect.com/science/article/pii/S1568494621003161): [Codes and Data](https://github.com/williamscott701/Cross-SEAN)


## II. Additional datasets to investigate later

### a. Covid Misinfo Annotated Datasets
1. [Dataset for COVID-19 misinformation on Twitter](https://github.com/Gautamshahi/Misinformation_COVID-19)

This dataset is used for a research article titled An Exploratory Study of COVID-19 Misinformation on Twitter. Two datasets are presented in this repository. There are two datasets in this repository. According to the authors, “the first dataset are the tweets which have been mentioned by fact-checking websites and are classified as false or partially false and the second dataset consists of COVID-19 tweets collected from publicly available corpus TweetsCOV19 (January-April 2020) and in-house crawling from May-July 2020.”

2. [CoAID](https://github.com/cuilimeng/CoAID)

CoAID (Covid-19 heAlthcare mIsinformation Dataset) is a diverse COVID-19 healthcare misinformation dataset, including fake news on websites and social platforms, along with users' social engagement about such news. It includes 1,896 news, 183,564 related user engagements, 516 social platform posts about COVID-19, and ground truth labels.”

3. [CMU-MisCov19: A Novel Twitter Dataset for Characterizing COVID-19 Misinformation](https://www.researchgate.net/publication/343414721_Characterizing_COVID-19_Misinformation_Communities_Using_a_Novel_Twitter_Dataset): [Codes and Data](https://zenodo.org/record/4024154#.X7XUkRNKgSI)

The dataset contains 4573 annotated tweets across 17 themes related to COVID-19. Their annotation codebook is also presented. This annotated dataset can be used for studies related to misinformation detection and characterization. 

4. [ANTi-Vax: a novel Twitter dataset for COVID-19 vaccine misinformation detection](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8648668/#bib12): [Codes and Data](https://github.com/SakibShahriar95/ANTiVax)


5. [ESOC COVID-19 Misinformation Dataset](https://esoc.princeton.edu/publications/esoc-covid-19-misinformation-dataset)

6. Data from [Poynter](https://www.poynter.org/coronavirusfactsalliance/)

[Poynter](https://www.poynter.org/coronavirusfactsalliance/) is a good source of annotated misinfo Covid data but this database requires some heavy processing to get the data we can use. But, here's an example paper with processsed data we can test with (articles not tweets though) [FakeCovid- A Multilingual Cross-domain Fact Check News Dataset for COVID-19](https://gautamshahi.github.io/FakeCovid/): [Codes and Data](https://github.com/Gautamshahi/FakeCovid)

Another paper using the same data source [COVID-19 Misinformation on Twitter: Multilingual Analysis](https://openreview.net/pdf?id=aDCizGE1HR2) but this one doesn't have codes/data.


### b. General Covid Datasets
1. [A COVID-19 Rumor Dataset](https://www.frontiersin.org/articles/10.3389/fpsyg.2021.644801/full)
2. [A large-scale COVID-19 Twitter chatter dataset for open scientific research -- an international collaboration](https://arxiv.org/abs/2004.03688v1): [Codes and Data](https://github.com/thepanacealab/covid19_twitter)
3. [COVID-19 Twitter Dataset with Latent Topics, Sentiments and Emotions Attributes](https://www.openicpsr.org/openicpsr/project/120321/version/V12/view)
4. [CORONAVIRUS (COVID-19) TWEETS DATASET](https://ieee-dataport.org/open-access/coronavirus-covid-19-tweets-dataset)