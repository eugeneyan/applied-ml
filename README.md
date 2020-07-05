# applied-ml
Curated papers, articles, and blogs sharing how **data science & machine learning is applied in production**. 

Have a favourite piece you're not seeing here? Want to contribute? Please make a [pull request](https://github.com/eugeneyan/applied-ml/pulls).

**Table of Contents**

- [Data Quality](#data-quality)
- [Data Engineering](#data-engineering)
- [Classification](#classification)
- [Regression](#regression)
- [Recommendation](#recommendation)
- [Search/Ranking](#searchranking)
- [Natural Language Processing](#natural-language-processing)
- [Sequence Modelling](#sequence-modelling)
- [Computer Vision](#computer-vision)
- [Reinforcement Learning](#reinforcement-learning)
- [Anomaly Detection](#anomaly-detection)
- [Optimization](#optimization)
- [Information Extraction](#information-extraction)
- [Validation and A/B Testing](#validation-and-ab-testing)
- [Practices](#practices)

## Data Quality
- [Monitoring Data Quality at Scale with Statistical Modeling](https://eng.uber.com/monitoring-data-quality-at-scale/) `Uber`
- [An Approach to Data Quality for Netflix Personalization Systems](https://databricks.com/session_na20/an-approach-to-data-quality-for-netflix-personalization-systems) `Netflix`

## Data Engineering
- [Zipline: Airbnb’s Machine Learning Data Management Platform](https://databricks.com/session/zipline-airbnbs-machine-learning-data-management-platform) `Airbnb`
- [Sputnik: Airbnb’s Apache Spark Framework for Data Engineering](https://databricks.com/session_na20/sputnik-airbnbs-apache-spark-framework-for-data-engineering) `Airbnb`
- [Feast: Bridging ML Models and Data](https://blog.gojekengineering.com/feast-bridging-ml-models-and-data-efd06b7d1644) `Gojek`

## Classification
- [High-precision phrase-based document classification on a modern scale](https://engineering.linkedin.com/research/2011/high-precision-phrase-based-document-classification-on-a-modern-scale) `LinkedIn`
- [Chimera: Large-Scale Classification using Machine Learning, Rules, and Crowdsourcing](http://pages.cs.wisc.edu/~anhai/papers/chimera-vldb14.pdf) `WalmartLabs`
- [Large-scale Item Categorization for e-Commerce](https://dl.acm.org/doi/10.1145/2396761.2396838) `DianPing`, `eBay`
- [Categorizing Products at Scale](https://engineering.shopify.com/blogs/engineering/categorizing-products-at-scale) `Shopify`
- [Learning to Diagnose with LSTM Recurrent Neural Networks](https://arxiv.org/abs/1511.03677) `Google`
- [Discovering and Classifying In-app Message Intent at Airbnb](https://medium.com/airbnb-engineering/discovering-and-classifying-in-app-message-intent-at-airbnb-6a55f5400a0c) `Airbnb`
- [How we built the good first issues feature](https://github.blog/2020-01-22-how-we-built-good-first-issues/) `GitHub`

## Regression
- [Using Machine Learning to Predict Value of Homes On Airbnb](https://medium.com/airbnb-engineering/using-machine-learning-to-predict-value-of-homes-on-airbnb-9272d3d4739d) `Airbnb`
- [Modeling conversion rates and saving millions of dollars using Kaplan-Meier and gamma distributions](https://better.engineering/modeling-conversion-rates-and-saving-millions-of-dollars-using-kaplan-meier-and-gamma-distributions/) `Better`
- [Using machine learning to predict the value of ad requests](https://blog.twitter.com/engineering/en_us/topics/insights/2020/using-machine-learning-to-predict-the-value-of-ad-requests.html) `Twitter`
- 

## Recommendation
- [Amazon.com Recommendations: Item-toItem Collaborative Filtering](https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf) `Amazon`
- [Recommending Complementary Products in E-Commerce Push Notifications with a Mixture Model Approach](https://arxiv.org/abs/1707.08113) `Alibaba`
- [Behavior Sequence Transformer for E-commerce Recommendation in Alibaba](https://arxiv.org/abs/1905.06874) `Alibaba`
- [Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939) `Telefonica`
- [Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf) `YouTube`
- [Personalized Recommendations for Experiences Using Deep Learning](https://www.tripadvisor.com/engineering/personalized-recommendations-for-experiences-using-deep-learning/) `TripAdvisor`
- [E-commerce in Your Inbox:Product Recommendations at Scale](https://arxiv.org/abs/1606.07154) `Yahoo`
- [Powered by AI: Instagram’s Explore recommender system](https://ai.facebook.com/blog/powered-by-ai-instagrams-explore-recommender-system/) `Facebook`
- [Artwork Personalization at Netflix](https://netflixtechblog.com/artwork-personalization-c589f074ad76) `Netflix`
- [To Be Continued: Helping you find shows to continue watching on Netflix](https://netflixtechblog.com/to-be-continued-helping-you-find-shows-to-continue-watching-on-7c0d8ee4dab6) `Netflix`
- [Learning a Personalized Homepage](https://netflixtechblog.com/learning-a-personalized-homepage-aa8ec670359a) `Netflix`
- [https://eng.uber.com/uber-eats-graph-learning/](https://eng.uber.com/uber-eats-graph-learning/) `Uber`
- [Calibrated Recommendations](https://dl.acm.org/doi/10.1145/3240323.3240372) `Netflix`
- [Explore, Exploit, and Explain: Personalizing Explainable Recommendations with Bandits](https://dl.acm.org/doi/10.1145/3240323.3240354) `Spotify`
- [For Your Ears Only: Personalizing Spotify Home with Machine Learning](https://engineering.atspotify.com/2020/01/16/for-your-ears-only-personalizing-spotify-home-with-machine-learning/) `Spotify`
- [Reach for the Top: How Spotify Built Shortcuts in Just Six Months](https://engineering.atspotify.com/2020/04/15/reach-for-the-top-how-spotify-built-shortcuts-in-just-six-months/) `Spotify`
- [The Evolution of Kit: Automating Marketing Using Machine Learning](https://engineering.shopify.com/blogs/engineering/evolution-kit-automating-marketing-machine-learning) `Shopify`
- [Using machine learning to predict what file you need next (Part 1)](https://dropbox.tech/machine-learning/content-suggestions-machine-learning) `Dropbox`
- [Using machine learning to predict what file you need next (Part 2)](https://dropbox.tech/machine-learning/using-machine-learning-to-predict-what-file-you-need-next-part-2) `Dropbox`
- [A closer look at the AI behind course recommendations on LinkedIn Learning (Part 1)](https://engineering.linkedin.com/blog/2020/course-recommendations-ai-part-one) `LinkedIn`
- [A closer look at the AI behind course recommendations on LinkedIn Learning (Part 2)](https://engineering.linkedin.com/blog/2020/course-recommendations-ai-part-two) `LinkedIn`

## Search/Ranking
- [Amazon Search: The Joy of Ranking Products](https://www.amazon.science/publications/amazon-search-the-joy-of-ranking-products) `Amazon`
- [Using Deep Learning at Scale in Twitter’s Timelines](https://blog.twitter.com/engineering/en_us/topics/insights/2017/using-deep-learning-at-scale-in-twitters-timelines.html) `Twitter`
- [Machine Learning-Powered Search Ranking of Airbnb Experiences](https://medium.com/airbnb-engineering/machine-learning-powered-search-ranking-of-airbnb-experiences-110b4b1a0789) `Airbnb`
- [Applying Deep Learning To Airbnb Search](https://arxiv.org/abs/1810.09591) `Airbnb`
- [Ranking Relevance in Yahoo Search](https://www.kdd.org/kdd2016/papers/files/adf0361-yinA.pdf) `Yahoo`
- [An Ensemble-Based Approach to Click-Through Rate Prediction for Promoted Listings at Etsy](https://arxiv.org/abs/1711.01377) `Etsy`
- [Why Do People Buy Seemingly Irrelevant Items in Voice Product Search?](https://www.amazon.science/publications/why-do-people-buy-irrelevant-items-in-voice-product-search) `Amazon`
- [The AI Behind LinkedIn Recruiter search and recommendation systems](https://engineering.linkedin.com/blog/2019/04/ai-behind-linkedin-recruiter-search-and-recommendation-systems) `LinkedIn`
- [AI at Scale in Bing](https://blogs.bing.com/search/2020_05/AI-at-Scale-in-Bing) `Microsoft`
- [Query Understanding Engine in Traveloka Universal Search](https://medium.com/traveloka-engineering/query-understanding-engine-in-traveloka-universal-search-410ad3895db7) `Traveloka`
- [Search-based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction](https://arxiv.org/abs/2006.05639) `Alibaba`

## Embeddings
- [Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba](https://arxiv.org/abs/1803.02349)`Alibaba`
- [Embeddings@Twitter](https://blog.twitter.com/engineering/en_us/topics/insights/2018/embeddingsattwitter.html) `Twitter` 
- [Listing Embeddings in Search Ranking](https://medium.com/airbnb-engineering/listing-embeddings-for-similar-listing-recommendations-and-real-time-personalization-in-search-601172f7603e) ([**Paper**](Real-time Personalization using Embeddings for SearchRanking at Airbnb)) `Airbnb`

## Natural Language Processing
- [Abusive Language Detection in Online User Content](http://www.yichang-cs.com/yahoo/WWW16_Abusivedetection.pdf) `Yahoo`
- [How natural language processing helps LinkedIn members get support easily](https://engineering.linkedin.com/blog/2019/04/how-natural-language-processing-help-support) `LinkedIn`
- [Building Smart Replies for Member Messages](https://engineering.linkedin.com/blog/2017/10/building-smart-replies-for-member-messages) `LinkedIn`
- [Smart Reply: Automated Response Suggestion for Email](https://research.google/pubs/pub45189/) `Google`
- [Assistive AI Makes Replying Easier](https://www.microsoft.com/en-us/research/group/msai/articles/assistive-ai-makes-replying-easier-2/) `Microsoft`
- [AI advances to better detect hate speech](https://ai.facebook.com/blog/ai-advances-to-better-detect-hate-speech/) `Facebook`
- [Using Neural Networks to Find Answers in Tables](https://ai.googleblog.com/2020/04/using-neural-networks-to-find-answers.html) `Google`
- [A Scalable Approach to Reducing Gender Bias in Google Translate](https://ai.googleblog.com/2020/04/a-scalable-approach-to-reducing-gender.html) `Google`
- [A state-of-the-art open source chatbot](https://ai.facebook.com/blog/state-of-the-art-open-source-chatbot) `Facebook`
- [Goal-Oriented End-to-End Conversational Models with Profile Features in a Real-World Setting](https://assets.amazon.science/47/03/e0d14dc34d3eb6e0d4ec282067bd/goal-oriented-end-to-end-chatbots-with-profile-features-in-a-real-world-setting.pdf) `Amazon`

## Sequence Modelling
- [Recommending Complementary Products in E-Commerce Push Notifications with a Mixture Model Approach](https://arxiv.org/abs/1707.08113) `Alibaba`
- [Practice on Long Sequential User Behavior Modeling for Click-Through Rate Prediction](https://arxiv.org/abs/1905.09248) `Alibaba`
- [Learning to Diagnose with LSTM Recurrent Neural Networks](https://arxiv.org/abs/1511.03677) `Google`
- [Deep Learning for Understanding Consumer Histories](https://engineering.zalando.com/posts/2016/10/deep-learning-for-understanding-consumer-histories.html) `Zalando`
- [Continual Prediction of Notification Attendance with Classical and Deep Network Approaches](https://arxiv.org/abs/1712.07120) `Telefonica`

## Computer Vision
- [Categorizing Listing Photos at Airbnb](https://medium.com/airbnb-engineering/categorizing-listing-photos-at-airbnb-f9483f3ab7e3) `Airbnb`
- [Amenity Detection and Beyond — New Frontiers of Computer Vision at Airbnb](https://medium.com/airbnb-engineering/amenity-detection-and-beyond-new-frontiers-of-computer-vision-at-airbnb-144a4441b72e) `Airbnb`
- [Powered by AI: Advancing product understanding and building new shopping experiences](https://ai.facebook.com/blog/powered-by-ai-advancing-product-understanding-and-building-new-shopping-experiences/) `Facebook`
- [Creating a Modern OCR Pipeline Using Computer Vision and Deep Learning](https://dropbox.tech/machine-learning/creating-a-modern-ocr-pipeline-using-computer-vision-and-deep-learning) `Dropbox`
- [How we improved computer vision metrics by more than 5% only by cleaning labelling errors](https://deepomatic.com/en/how-we-improved-computer-vision-metrics-by-more-than-5-percent-only-by-cleaning-labelling-errors/) `Deepomatic`
- [A Neural Weather Model for Eight-Hour Precipitation Forecasting](https://ai.googleblog.com/2020/03/a-neural-weather-model-for-eight-hour.html) `Google`
- [Converting text to images for product discovery](https://www.amazon.science/blog/converting-text-to-images-for-product-discovery) `Amazon`


## Reinforcement Learning
- [Deep Reinforcement Learning for Sponsored Search Real-time Bidding](https://arxiv.org/pdf/1803.00259.pdf) `Alibaba`
- [Dynamic Pricing on E-commerce Platform with Deep Reinforcement Learning](https://arxiv.org/abs/1912.02572) `Alibaba`
- [Budget Constrained Bidding by Model-free Reinforcement Learning in Display Advertising](https://arxiv.org/abs/1802.08365) `Alibaba`
- [Productionizing Deep Reinforcement Learning with Spark and MLflow](https://databricks.com/session_na20/productionizing-deep-reinforcement-learning-with-spark-and-mlflow) `Zynga`

## Anomaly Detection
- [Detecting Performance Anomalies in External Firmware Deployments](https://netflixtechblog.com/detecting-performance-anomalies-in-external-firmware-deployments-ed41b1bfcf46) `Netflix`
- [Detecting and preventing abuse on LinkedIn using isolation forests](https://engineering.linkedin.com/blog/2019/isolation-forest) `LinkedIn`
- [Uncovering Insurance Fraud Conspiracy with Network Learning](https://arxiv.org/abs/2002.12789) `Ant Financial`

## Optimization
- [How Trip Inferences and Machine Learning Optimize Delivery Times on Uber Eats](https://eng.uber.com/uber-eats-trip-optimization/) `Uber`

## Information Extraction
- [Unsupervised Extraction of Attributes and Their Values from Product Description](https://www.aclweb.org/anthology/I13-1190/) `Rakuten`
- [Information Extraction from Receipts with Graph Convolutional Networks](https://nanonets.com/blog/information-extraction-graph-convolutional-networks/) `Nanonets`

## Validation and A/B Testing
- [The reusable holdout: Preserving validity in adaptive data analysis] (https://ai.googleblog.com/2015/08/the-reusable-holdout-preserving.html) `Google`
- [A/B Testing with Hierarchical Models in Python](https://blog.dominodatalab.com/ab-testing-with-hierarchical-models-in-python/) `Domino`
- [Detecting interference: An A/B test of A/B tests](https://engineering.linkedin.com/blog/2019/06/detecting-interference--an-a-b-test-of-a-b-tests) `LinkedIn`
- [Building inclusive products through A/B testing](https://engineering.linkedin.com/blog/2020/building-inclusive-products-through-a-b-testing) `LinkedIn`
- [Experimenting to solve cramming](https://blog.twitter.com/engineering/en_us/topics/insights/2017/Experimenting-To-Solve-Cramming.html) `Twitter`
- [Announcing a New Framework for Designing Optimal Experiments with Pyro](https://eng.uber.com/oed-pyro-release/) `Uber`
- [Enabling 10x More Experiments with Traveloka Experiment Platform](https://medium.com/traveloka-engineering/enabling-10x-more-experiments-with-traveloka-experiment-platform-8cea13e952c) `Traveloka`

## Practices
- [Practical Recommendations for Gradient-Based Training of Deep Architectures] (https://arxiv.org/abs/1206.5533) `Yoshua Bengio`
- [Machine Learning: The High Interest Credit Card of Technical Debt](https://research.google/pubs/pub43146/) `Google`
- [Rules of Machine Learning: Best Practices for ML Engineering](https://developers.google.com/machine-learning/guides/rules-of-ml) `Google`
- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf) `Google`
- [On Challenges in Machine Learning Model Management](http://sites.computer.org/debull/A18dec/p5.pdf) `Amazon`
- [150 successful Machine Learning models: 6 lessons learned at Booking.com](https://www.kdd.org/kdd2019/accepted-papers/view/150-successful-machine-learning-models-6-lessons-learned-at-booking.com) `Booking.com`

