### CSG - Consensus Source Gradient
A framework for making AI models to predict stock market values.

This was a side-project of mine and a finance student I am friends with. It is a framework designed to aid
in testing various neural networks and regressional systems on stock market values. It was created during a 
time of massive change in the investment population of North America, and was based around the theory that
many people invest in stocks based off of the ticker, rather than based off of news or other methods of
prediction.

Fair warning, this was written in a small team (of two) without the idea of publishing particularly in mind.
As such, it is somewhat undocumented, and will likely stay that way for a while.

### Main Data Inputs
Most models were tested on "pure ticker" data. Meaning they were given the open price, close price, high price,
low price, and some feature extractions of these 4. This was to test the popular idea that in order to predict
the value of a stock, you need to understand it's external influences. Some stocks were tested with data from
quarterly reports, and they typically performed with a lower loss, ultimately this idea was canned only because
the process of harvesting data from quarterly reports was not economical for our small team. If I had more time
I would revisit it.

### Conclusion
"Pure Ticker" models usually if given enough training time, performed a bit worse than SOTA for automatic
momentum trading techniques (at least from what I read on the internet).

Models with quarterly-report data usually performed well briefly after the report
but dwindled in accuracy near the end of the three month period (relying again on traditional momentum
trading techniques and weighted averages). This seems exciting at first, but the predictions of the next
day's market after the release of the quarterly report is usually not as enlightening as one would think.
What may be of value to know, is that of our test stocks, people tended to over-estimate the value of 
companies, and have their views re-aligned once the report came out. This may make an interesting basis
for a model of predicting how "naive" a demographic of investors is, which would allow for predicting these
re-adjustments and could be used to extract a great portion of profits. But all models we trained typically 
wound up being naive themselves. 

Of course, no models were able to predict sudden crashes which were common across the whole market (due to
global events), but this was never expected in the least.

### In Future
If I had more time and more money I would look into continuing this research using quarterly report data as 
well as sentiment analysis from news outlets, social media, and popular financial celebrities. I would also
look into adapting more rigid pre-existing financial prediction models using machine learning, rather than
trying to get pre-existing machine learning models to emulate financial prediction models.

### Nomenclature 
If you would like to pick up this project and keep working on these ideas, there's some terms you may need 
to become familiar with. 

"yoink" - meaning to take in a clandestine manner

"yeet"  - meaning to jettison with zeal

### Missing Models
Not all the code behind the models is available here. The point of this repository is really to share the 
framework and maybe save a weekend or so of work for someone who may be interested in this. For this reason 
only two models have been included (to show the general format).

There's a few reasons why not all the models are here:
- Some models pre-date the framework, and were just single-file Python scripts.
- Others were omitted as my partner wanted to use them for a university project, but did not want to be accused of 
plagiarism. This seems entirely reasonable to me, and we agreed there would be no problem in uploading them 
after he completed his project, and that the implementations would fall under the GPLv2.
- I did a lot of work very drunk, and would rather clean it and comment it before sharing.

### Papers
The following papers were either re-implemented entirely for testing, or were adapted to different models
before testing. This list is not comprehensive, as many theories were tested against the market, and we 
didn't keep great track of all of them (mainly the less-promising ideas that were long forgotten).

- Reverse Momentum Trading AI - Xciki Li and Vincent Tam
- Using LSTM in Stock Prediction and Quantitative Trading - Zhichao Zou, Zihao Qu
- Application of Recurrent Neural Networks To Momentum Trading - Suet Yin Wong


