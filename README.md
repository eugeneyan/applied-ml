# applied-ml
Curated papers, articles, and blogs on **data science & machine learning in production**. ⚙️

[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](./CONTRIBUTING.md) [![Summaries](https://img.shields.io/badge/summaries-in%20tweets-%2300acee.svg?style=flat)](https://twitter.com/eugeneyan/status/1288236455228137473) ![HitCount](http://hits.dwyl.com/eugeneyan/applied-ml.svg)

Figuring out how to implement your ML project? Learn how other organizations did it:

- **How** the problem is framed 🔎(e.g., personalization as recsys vs. search vs. sequences)
- **What** machine learning techniques worked ✅ (and sometimes, what didn't ❌)
- **Why** it works, the science behind it with research, literature, and references 📂
- **What** real-world results were achieved (so you can better assess ROI ⏰💰📈)

P.S., Want a summary of ML advancements? 👉[`ml-surveys`](https://github.com/eugeneyan/ml-surveys)

**Table of Contents**

1. [Data Quality](#data-quality)
2. [Data Engineering](#data-engineering)
3. [Data Discovery](#data-discovery)
4. [Feature Stores](#feature-stores)
5. [Classification](#classification)
6. [Regression](#regression)
7. [Forecasting](#forecasting)
8. [Recommendation](#recommendation)
9. [Search & Ranking](#search--ranking)
10. [Embeddings](#embeddings)
11. [Natural Language Processing](#natural-language-processing)
12. [Sequence Modelling](#sequence-modelling)
13. [Computer Vision](#computer-vision)
14. [Reinforcement Learning](#reinforcement-learning)
15. [Anomaly Detection](#anomaly-detection)
16. [Graph](#graph)
17. [Optimization](#optimization)
18. [Information Extraction](#information-extraction)
19. [Weak Supervision](#weak-supervision)
20. [Generation](#generation)
21. [Audio](#audio)
22. [Validation and A/B Testing](#validation-and-ab-testing)
23. [Model Management](#model-management)
24. [Efficiency](#efficiency)
25. [Ethics](#ethics)
26. [Infra](#infra)
27. [Platforms](#platforms)
26. [Practices](#practices)
27. [Team Structure](#team-structure)
28. [Fails](#fails)

## Data Quality
1. [Monitoring Data Quality at Scale with Statistical Modeling](https://eng.uber.com/monitoring-data-quality-at-scale/) `Uber`
2. [An Approach to Data Quality for Netflix Personalization Systems](https://databricks.com/session_na20/an-approach-to-data-quality-for-netflix-personalization-systems) `Netflix`
3. [Automating Large-Scale Data Quality Verification](https://www.amazon.science/publications/automating-large-scale-data-quality-verification) ([Paper](https://assets.amazon.science/a6/88/ad858ee240c38c6e9dce128250c0/automating-large-scale-data-quality-verification.pdf))`Amazon`
4. [Meet Hodor — Gojek’s Upstream Data Quality Tool](https://blog.gojekengineering.com/meet-hodor-gojeks-upstream-data-quality-tool-fb877447aad1) `Gojek`
5. [Reliable and Scalable Data Ingestion at Airbnb](https://www.slideshare.net/HadoopSummit/reliable-and-scalable-data-ingestion-at-airbnb-63920989) `Airbnb`
6. [Data Management Challenges in Production Machine Learning](https://research.google/pubs/pub46178/) ([Paper](https://thodrek.github.io/CS839_spring18/papers/p1723-polyzotis.pdf)) `Google`
7. [Improving Accuracy By Certainty Estimation of Human Decisions, Labels, and Raters](https://research.fb.com/blog/2020/08/improving-the-accuracy-of-community-standards-enforcement-by-certainty-estimation-of-human-decisions/) ([Paper](https://research.fb.com/wp-content/uploads/2020/08/CLARA-Confidence-of-Labels-and-Raters.pdf)) `Facebook`

## Data Engineering
1. [Zipline: Airbnb’s Machine Learning Data Management Platform](https://databricks.com/session/zipline-airbnbs-machine-learning-data-management-platform) `Airbnb`
2. [Sputnik: Airbnb’s Apache Spark Framework for Data Engineering](https://databricks.com/session_na20/sputnik-airbnbs-apache-spark-framework-for-data-engineering) `Airbnb`
3. [Unbundling Data Science Workflows with Metaflow and AWS Step Functions](https://netflixtechblog.com/unbundling-data-science-workflows-with-metaflow-and-aws-step-functions-d454780c6280) `Netflix`
4. [How DoorDash is Scaling its Data Platform to Delight Customers and Meet Growing Demand](https://doordash.engineering/2020/09/25/how-doordash-is-scaling-its-data-platform/) `DoorDash`
5. [Revolutionizing Money Movements at Scale with Strong Data Consistency](https://eng.uber.com/money-scale-strong-data/) `Uber`
6. [Zipline - A Declarative Feature Engineering Framework](https://www.youtube.com/watch?v=LjcKCm0G_OY) `Airbnb`

## Data Discovery
1. [Amundsen — Lyft’s Data Discovery & Metadata Engine](https://eng.lyft.com/amundsen-lyfts-data-discovery-metadata-engine-62d27254fbb9) `Lyft`
2. [Open Sourcing Amundsen: A Data Discovery And Metadata Platform](https://eng.lyft.com/open-sourcing-amundsen-a-data-discovery-and-metadata-platform-2282bb436234) ([Code](https://github.com/lyft/amundsen)) `Lyft`
3. [Amundsen: One Year Later](https://eng.lyft.com/amundsen-1-year-later-7b60bf28602) `Lyft`
3. [Using Amundsen to Support User Privacy via Metadata Collection at Square](https://developer.squareup.com/blog/using-amundsen-to-support-user-privacy-via-metadata-collection-at-square/) `Square`
4. [Discovery and Consumption of Analytics Data at Twitter](https://blog.twitter.com/engineering/en_us/topics/insights/2016/discovery-and-consumption-of-analytics-data-at-twitter.html) `Twitter`
3. [Democratizing Data at Airbnb](https://medium.com/airbnb-engineering/democratizing-data-at-airbnb-852d76c51770) `Airbnb`
4. [Databook: Turning Big Data into Knowledge with Metadata at Uber](https://eng.uber.com/databook/) `Uber`
5. [Turning Metadata Into Insights with Databook](https://eng.uber.com/metadata-insights-databook/) `Uber`
5. [Metacat: Making Big Data Discoverable and Meaningful at Netflix](https://netflixtechblog.com/metacat-making-big-data-discoverable-and-meaningful-at-netflix-56fb36a53520) ([Code](https://github.com/Netflix/metacat)) `Netflix`
6. [DataHub: A Generalized Metadata Search & Discovery Tool](https://engineering.linkedin.com/blog/2019/data-hub) ([Code](https://github.com/linkedin/datahub)) `LinkedIn`
7. [DataHub: Popular Metadata Architectures Explained](https://engineering.linkedin.com/blog/2020/datahub-popular-metadata-architectures-explained) `LinkedIn`
7. [How We Improved Data Discovery for Data Scientists at Spotify](https://engineering.atspotify.com/2020/02/27/how-we-improved-data-discovery-for-data-scientists-at-spotify/) `Spotify`
8. [How We’re Solving Data Discovery Challenges at Shopify](https://engineering.shopify.com/blogs/engineering/solving-data-discovery-challenges-shopify) `Shopify`
9. [Nemo: Data discovery at Facebook](https://engineering.fb.com/data-infrastructure/nemo/) `Facebook`
10. [Apache Atlas: Data Goverance and Metadata Framework for Hadoop](https://atlas.apache.org/#/) ([Code](https://github.com/apache/atlas)) `Apache`
11. [Collect, Aggregate, and Visualize a Data Ecosystem's Metadata](https://marquezproject.github.io/marquez/) ([Code](https://github.com/MarquezProject/marquez)) `WeWork`


## Feature Stores
1. [Introducing Feast: An Open Source Feature Store for Machine Learning](https://cloud.google.com/blog/products/ai-machine-learning/introducing-feast-an-open-source-feature-store-for-machine-learning) ([Code](https://github.com/feast-dev/feast)) `Gojek`
2. [Feast: Bridging ML Models and Data](https://blog.gojekengineering.com/feast-bridging-ml-models-and-data-efd06b7d1644) `Gojek`
3. [Building a Scalable ML Feature Store with Redis, Binary Serialization, and Compression](https://doordash.engineering/2020/11/19/building-a-gigascale-ml-feature-store-with-redis/) `DoorDash`
4. [Michelangelo Palette: A Feature Engineering Platform at Uber](https://www.infoq.com/presentations/michelangelo-palette-uber/) `Uber`
5. [Distributed Time Travel for Feature Generation](https://netflixtechblog.com/distributed-time-travel-for-feature-generation-389cccdd3907) `Netflix`
6. [Fact Store at Scale for Netflix Recommendations](https://databricks.com/session/fact-store-scale-for-netflix-recommendations) `Netflix`
6. [The Architecture That Powers Twitter's Feature Store](https://www.youtube.com/watch?v=UNailXoiIrY) `Twitter`
7. [Building the Activity Graph, Part 2 (Feature Storage Section)](https://engineering.linkedin.com/blog/2017/07/building-the-activity-graph--part-2) `LinkedIn`
7. [Rapid Experimentation Through Standardization: Typed AI features for LinkedIn’s Feed](https://engineering.linkedin.com/blog/2020/feed-typed-ai-features) `LinkedIn`
8. [Accelerating Machine Learning with the Feature Store Service](https://technology.condenast.com/story/accelerating-machine-learning-with-the-feature-store-service) `Condé Nast`
9. [Building a Feature Store](https://nlathia.github.io/2020/12/Building-a-feature-store.html) `Monzo Bank`
10. [Zipline: Airbnb’s Machine Learning Data Management Platform](https://databricks.com/session/zipline-airbnbs-machine-learning-data-management-platform) `Airbnb`
11. [Butterfree: A Spark-based Framework for Feature Store Building](https://medium.com/quintoandar-tech-blog/butterfree-a-spark-based-framework-for-feature-store-building-48c3640522c7) ([Code](https://github.com/quintoandar/butterfree)) `QuintoAndar`

## Classification
1. [High-Precision Phrase-Based Document Classification on a Modern Scale](https://engineering.linkedin.com/research/2011/high-precision-phrase-based-document-classification-on-a-modern-scale) ([Paper](http://web.stanford.edu/~gavish/documents/phrase_based.pdf)) `LinkedIn`
2. [Chimera: Large-scale Classification using Machine Learning, Rules, and Crowdsourcing](https://dl.acm.org/doi/10.14778/2733004.2733024) ([Paper](http://pages.cs.wisc.edu/%7Eanhai/papers/chimera-vldb14.pdf)) `WalmartLabs`
3. [Large-scale Item Categorization for e-Commerce](https://dl.acm.org/doi/10.1145/2396761.2396838) ([Paper](https://www.researchgate.net/profile/Jean_David_Ruvini/publication/262270957_Large-scale_item_categorization_for_e-commerce/links/5512dc3d0cf270fd7e33a0d5/Large-scale-item-categorization-for-e-commerce.pdf)) `DianPing`, `eBay`
4. [Large-scale Item Categorization in e-Commerce Using Multiple Recurrent Neural Networks](https://www.kdd.org/kdd2016/subtopic/view/large-scale-item-categorization-in-e-commerce-using-multiple-recurrent-neur/) ([Paper](https://www.kdd.org/kdd2016/papers/files/adf0392-haAemb.pdf)) `NAVER`
4. [Categorizing Products at Scale](https://engineering.shopify.com/blogs/engineering/categorizing-products-at-scale) `Shopify`
5. [Learning to Diagnose with LSTM Recurrent Neural Networks](https://arxiv.org/abs/1511.03677) ([Paper](https://arxiv.org/pdf/1511.03677.pdf)) `Google`
6. [Discovering and Classifying In-app Message Intent at Airbnb](https://medium.com/airbnb-engineering/discovering-and-classifying-in-app-message-intent-at-airbnb-6a55f5400a0c) `Airbnb`
7. [How We Built the Good First Issues Feature](https://github.blog/2020-01-22-how-we-built-good-first-issues/) `GitHub`
8. [Teaching Machines to Triage Firefox Bugs](https://hacks.mozilla.org/2019/04/teaching-machines-to-triage-firefox-bugs/) `Mozilla`
9. [Testing Firefox More Efficiently with Machine Learning](https://hacks.mozilla.org/2020/07/testing-firefox-more-efficiently-with-machine-learning/) `Mozilla`
10. [Using ML to Subtype Patients Receiving Digital Mental Health Interventions](https://www.microsoft.com/en-us/research/blog/a-path-to-personalization-using-ml-to-subtype-patients-receiving-digital-mental-health-interventions/) ([Paper](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2768347)) `Microsoft`
11. [Prediction of Advertiser Churn for Google AdWords](https://research.google/pubs/pub36678/) ([Paper](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/36678.pdf)) `Google`
12. [Scalable Data Classification for Security and Privacy](https://engineering.fb.com/security/data-classification-system/) ([Paper](https://arxiv.org/pdf/2006.14109.pdf)) `Facebook`
13. [Uncovering Online Delivery Menu Best Practices with Machine Learning](https://doordash.engineering/2020/11/10/uncovering-online-delivery-menu-best-practices-with-machine-learning/) `DoorDash`
14. [Using a Human-in-the-Loop to Overcome the Cold Start Problem in Menu Item Tagging](https://doordash.engineering/2020/08/28/overcome-the-cold-start-problem-in-menu-item-tagging/) `DoorDash`

## Regression
1. [Using Machine Learning to Predict Value of Homes On Airbnb](https://medium.com/airbnb-engineering/using-machine-learning-to-predict-value-of-homes-on-airbnb-9272d3d4739d) `Airbnb`
2. [Using Machine Learning to Predict the Value of Ad Requests](https://blog.twitter.com/engineering/en_us/topics/insights/2020/using-machine-learning-to-predict-the-value-of-ad-requests.html) `Twitter`
3. [Open-Sourcing Riskquant, a Library for Quantifying Risk](https://netflixtechblog.com/open-sourcing-riskquant-a-library-for-quantifying-risk-6720cc1e4968) ([Code](https://github.com/Netflix-Skunkworks/riskquant)) `NetFlix`
4. [Solving for Unobserved Data in a Regression Model Using a Simple Data Adjustment](https://doordash.engineering/2020/10/14/solving-for-unobserved-data-in-a-regression-model/) `DoorDash`

## Forecasting
1. [Forecasting at Uber: An Introduction](https://eng.uber.com/forecasting-introduction/) `Uber`
2. [Engineering Extreme Event Forecasting at Uber with RNN](https://eng.uber.com/neural-networks/) `Uber`
3. [Transforming Financial Forecasting with Data Science and Machine Learning at Uber](https://eng.uber.com/transforming-financial-forecasting-machine-learning/) `Uber`
4. [Under the Hood of Gojek’s Automated Forecasting Tool](https://blog.gojekengineering.com/under-the-hood-of-gojeks-automated-forecasting-tool-3ddb24ad29f1) `GoJek`
5. [BusTr: Predicting Bus Travel Times from Real-Time Traffic](https://dl.acm.org/doi/abs/10.1145/3394486.3403376) ([Paper](https://dl.acm.org/doi/pdf/10.1145/3394486.3403376), [Video](https://crossminds.ai/video/5f3369790576dd25aef288db/)) `Google`
6. [Retraining Machine Learning Models in the Wake of COVID-19](https://doordash.engineering/2020/09/15/retraining-ml-models-covid-19/) `DoorDash`
7. [Automatic Forecasting using Prophet, Databricks, Delta Lake and MLflow](https://www.youtube.com/watch?v=TkcpjnLh690) ([Paper](https://peerj.com/preprints/3190.pdf), [Code](https://github.com/facebook/prophet)) `Atlassian`

## Recommendation
1. [Amazon.com Recommendations: Item-to-Item Collaborative Filtering](https://ieeexplore.ieee.org/document/1167344) ([Paper](https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf)) `Amazon`
2. [Temporal-Contextual Recommendation in Real-Time](https://www.amazon.science/publications/temporal-contextual-recommendation-in-real-time) ([Paper](https://assets.amazon.science/96/71/d1f25754497681133c7aa2b7eb05/temporal-contextual-recommendation-in-real-time.pdf)) `Amazon`
3. [P-Companion: A Framework for Diversified Complementary Product Recommendation](https://www.amazon.science/publications/p-companion-a-principled-framework-for-diversified-complementary-product-recommendation) ([Paper](https://assets.amazon.science/d5/16/3f7809974a899a11bacdadefdf24/p-companion-a-principled-framework-for-diversified-complementary-product-recommendation.pdf)) `Amazon`
2. [Recommending Complementary Products in E-Commerce Push Notifications](https://arxiv.org/abs/1707.08113) ([Paper](https://arxiv.org/pdf/1707.08113.pdf)) `Alibaba`
3. [Behavior Sequence Transformer for E-commerce Recommendation in Alibaba](https://arxiv.org/abs/1905.06874) ([Paper](https://arxiv.org/pdf/1905.06874.pdf)) `Alibaba`
4. [TPG-DNN: A Method for User Intent Prediction with Multi-task Learning](https://arxiv.org/abs/2008.02122) ([Paper](https://arxiv.org/pdf/2008.02122.pdf)) `Alibaba`
5. [PURS: Personalized Unexpected Recommender System for Improving User Satisfaction](https://dl.acm.org/doi/10.1145/3383313.3412238) ([Paper](https://dl.acm.org/doi/pdf/10.1145/3383313.3412238)) `Alibaba`
4. [Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939) ([Paper](https://arxiv.org/pdf/1511.06939.pdf)) `Telefonica`
5. [How 20th Century Fox uses ML to predict a movie audience](https://cloud.google.com/blog/products/ai-machine-learning/how-20th-century-fox-uses-ml-to-predict-a-movie-audience) ([Paper](https://arxiv.org/abs/1810.08189)) `20th Century Fox`
6. [Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf) `YouTube`
7. [Personalized Recommendations for Experiences Using Deep Learning](https://www.tripadvisor.com/engineering/personalized-recommendations-for-experiences-using-deep-learning/) `TripAdvisor`
8. [E-commerce in Your Inbox: Product Recommendations at Scale](https://arxiv.org/abs/1606.07154) `Yahoo`
9. [Product Recommendations at Scale](https://arxiv.org/abs/1606.07154) ([Paper](https://arxiv.org/pdf/1606.07154.pdf)) `Yahoo`
10. [Powered by AI: Instagram’s Explore recommender system](https://ai.facebook.com/blog/powered-by-ai-instagrams-explore-recommender-system/) `Facebook`
11. [Netflix Recommendations: Beyond the 5 stars (Part 1](https://netflixtechblog.com/netflix-recommendations-beyond-the-5-stars-part-1-55838468f429) ([Part 2](https://netflixtechblog.com/netflix-recommendations-beyond-the-5-stars-part-2-d9b96aa399f5)) `Netflix`
12. [Learning a Personalized Homepage](https://netflixtechblog.com/learning-a-personalized-homepage-aa8ec670359a) `Netflix`
13. [Artwork Personalization at Netflix](https://netflixtechblog.com/artwork-personalization-c589f074ad76) `Netflix`
14. [To Be Continued: Helping you find shows to continue watching on Netflix](https://netflixtechblog.com/to-be-continued-helping-you-find-shows-to-continue-watching-on-7c0d8ee4dab6) `Netflix`
14. [Calibrated Recommendations](https://dl.acm.org/doi/10.1145/3240323.3240372) ([Paper](https://dl.acm.org/doi/pdf/10.1145/3240323.3240372)) `Netflix`
15. [Food Discovery with Uber Eats: Recommending for the Marketplace](https://eng.uber.com/uber-eats-recommending-marketplace/) `Uber`
15. [Food Discovery with Uber Eats: Using Graph Learning to Power Recommendations](https://eng.uber.com/uber-eats-graph-learning/) `Uber`
16. [How Music Recommendation Works — And Doesn’t Work](https://notes.variogr.am/2012/12/11/how-music-recommendation-works-and-doesnt-work/) `Spotify`
17. [Music recommendation at Spotify](http://sigir.org/afirm2019/slides/16.%20Friday%20-%20Music%20Recommendation%20at%20Spotify%20-%20Ben%20Carterette.pdf) `Spotify`
18. [Recommending Music on Spotify with Deep Learning](https://benanne.github.io/2014/08/05/spotify-cnns.html) `Spotify`
19. [For Your Ears Only: Personalizing Spotify Home with Machine Learning](https://engineering.atspotify.com/2020/01/16/for-your-ears-only-personalizing-spotify-home-with-machine-learning/) `Spotify`
20. [Reach for the Top: How Spotify Built Shortcuts in Just Six Months](https://engineering.atspotify.com/2020/04/15/reach-for-the-top-how-spotify-built-shortcuts-in-just-six-months/) `Spotify`
21. [Explore, Exploit, and Explain: Personalizing Explainable Recommendations with Bandits](https://dl.acm.org/doi/10.1145/3240323.3240354) ([Paper](https://static1.squarespace.com/static/5ae0d0b48ab7227d232c2bea/t/5ba849e3c83025fa56814f45/1537755637453/BartRecSys.pdf)) `Spotify`
22. [Contextual and Sequential User Embeddings for Large-Scale Music Recommendation](https://dl.acm.org/doi/10.1145/3383313.3412248) ([Paper](https://dl.acm.org/doi/pdf/10.1145/3383313.3412248)) `Spotify`
22. [The Evolution of Kit: Automating Marketing Using Machine Learning](https://engineering.shopify.com/blogs/engineering/evolution-kit-automating-marketing-machine-learning) `Shopify`
23. [Using Machine Learning to Predict what File you Need Next (Part 1)](https://dropbox.tech/machine-learning/content-suggestions-machine-learning) `Dropbox`
24. [Using Machine Learning to Predict what File you Need Next (Part 2)](https://dropbox.tech/machine-learning/using-machine-learning-to-predict-what-file-you-need-next-part-2) `Dropbox`
25. [Personalized Recommendations in LinkedIn Learning](https://engineering.linkedin.com/blog/2016/12/personalized-recommendations-in-linkedin-learning) `LinkedIn`
25. [A Closer Look at the AI Behind Course Recommendations on LinkedIn Learning (Part 1)](https://engineering.linkedin.com/blog/2020/course-recommendations-ai-part-one) `LinkedIn`
26. [A Closer Look at the AI Behind Course Recommendations on LinkedIn Learning (Part 2)](https://engineering.linkedin.com/blog/2020/course-recommendations-ai-part-two) `LinkedIn`
27. [Learning to be Relevant: Evolution of a Course Recommendation System](https://dl.acm.org/doi/pdf/10.1145/3357384.3357817) (**PAPER NEEDED**)`LinkedIn`
28. [Building a Heterogeneous Social Network Recommendation System](https://engineering.linkedin.com/blog/2020/building-a-heterogeneous-social-network-recommendation-system) `LinkedIn`
28. [How TikTok recommends videos #ForYou](https://newsroom.tiktok.com/en-us/how-tiktok-recommends-videos-for-you) `ByteDance`
29. [A Meta-Learning Perspective on Cold-Start Recommendations for Items](https://papers.nips.cc/paper/7266-a-meta-learning-perspective-on-cold-start-recommendations-for-items) ([Paper](https://papers.nips.cc/paper/7266-a-meta-learning-perspective-on-cold-start-recommendations-for-items.pdf)) `Twitter`
30. [Zero-Shot Heterogeneous Transfer Learning from RecSys to Cold-Start Search Retrieval](https://arxiv.org/abs/2008.02930) ([Paper](https://arxiv.org/pdf/2008.02930.pdf)) `Google`
31. [Improved Deep & Cross Network for Feature Cross Learning in Web-scale LTR Systems](https://arxiv.org/abs/2008.13535) ([Paper](https://arxiv.org/pdf/2008.13535.pdf)) `Google`
32. [Personalized Channel Recommendations in Slack](https://slack.engineering/personalized-channel-recommendations-in-slack/) `Slack`
33. [Deep Retrieval: End-to-End Learnable Structure Model for Large-Scale Recommendations](https://arxiv.org/abs/2007.07203) ([Paper](https://arxiv.org/pdf/2007.07203.pdf)) `ByteDance`
34. [Future Data Helps Training: Modeling Future Contexts for Session-based Recommendation]() ([Paper](https://arxiv.org/pdf/1906.04473.pdf)) `Tencent`
35. [Using AI to Help Health Experts Address the COVID-19 Pandemic](https://ai.facebook.com/blog/using-ai-to-help-health-experts-address-the-covid-19-pandemic/) `Facebook`
36. [A Case Study of Session-based Recommendations in the Home-improvement Domain](https://dl.acm.org/doi/10.1145/3383313.3412235) ([Paper](https://dl.acm.org/doi/pdf/10.1145/3383313.3412235)) `Home Depot`
37. [Balancing Relevance and Discovery to Inspire Customers in the IKEA App](https://dl.acm.org/doi/10.1145/3383313.3411550) ([Paper](https://dl.acm.org/doi/pdf/10.1145/3383313.3411550)) `Ikea`
38. [How we use AutoML, Multi-task learning and Multi-tower models for Pinterest Ads](https://medium.com/pinterest-engineering/how-we-use-automl-multi-task-learning-and-multi-tower-models-for-pinterest-ads-db966c3dc99e) `Pinterest`
39. [Personalized Cuisine Filter Based on Customer Preference and Local Popularity](https://doordash.engineering/2020/01/27/personalized-cuisine-filter/) `DoorDash`

## Search & Ranking
1. [Amazon Search: The Joy of Ranking Products](https://www.amazon.science/publications/amazon-search-the-joy-of-ranking-products) ([Paper](https://assets.amazon.science/89/cd/34289f1f4d25b5857d776bdf04d5/amazon-search-the-joy-of-ranking-products.pdf), [Video](https://www.youtube.com/watch?v=NLrhmn-EZ88), [Code](https://github.com/dariasor/TreeExtra)) `Amazon`
2. [Why Do People Buy Seemingly Irrelevant Items in Voice Product Search?](https://www.amazon.science/publications/why-do-people-buy-irrelevant-items-in-voice-product-search) ([Paper](https://assets.amazon.science/f7/48/0562b2c14338a0b76ccf4f523fa5/why-do-people-buy-irrelevant-items-in-voice-product-search.pdf)) `Amazon`
3. [Semantic Product Search](https://arxiv.org/abs/1907.00937) ([Paper](https://arxiv.org/pdf/1907.00937.pdf)) `Amazon`
3. [How Lazada Ranks Products to Improve Customer Experience and Conversion](https://www.slideshare.net/eugeneyan/how-lazada-ranks-products-to-improve-customer-experience-and-conversion) `Lazada`
4. [Using Deep Learning at Scale in Twitter’s Timelines](https://blog.twitter.com/engineering/en_us/topics/insights/2017/using-deep-learning-at-scale-in-twitters-timelines.html) `Twitter`
5. [Machine Learning-Powered Search Ranking of Airbnb Experiences](https://medium.com/airbnb-engineering/machine-learning-powered-search-ranking-of-airbnb-experiences-110b4b1a0789) `Airbnb`
6. [Applying Deep Learning To Airbnb Search](https://arxiv.org/abs/1810.09591) ([Paper](https://arxiv.org/pdf/1810.09591.pdf)) `Airbnb`
7. [Managing Diversity in Airbnb Search](https://arxiv.org/abs/2004.02621) ([Paper](https://arxiv.org/pdf/2004.02621.pdf)) `Airbnb`
8. [Improving Deep Learning for Airbnb Search](https://arxiv.org/abs/2002.05515) ([Paper](https://arxiv.org/pdf/2002.05515.pdf)) `Airbnb`
8. [Ranking Relevance in Yahoo Search](https://www.kdd.org/kdd2016/subtopic/view/ranking-relevance-in-yahoo-search) ([Paper](https://www.kdd.org/kdd2016/papers/files/adf0361-yinA.pdf)) `Yahoo`
9. [An Ensemble-based Approach to Click-Through Rate Prediction for Promoted Listings at Etsy](https://arxiv.org/abs/1711.01377) ([Paper](https://arxiv.org/pdf/1711.01377.pdf)) `Etsy`
10. [Learning to Rank Personalized Search Results in Professional Networks](https://arxiv.org/abs/1605.04624) ([Paper](https://arxiv.org/pdf/1605.04624.pdf)) `LinkedIn`
11. [Entity Personalized Talent Search Models with Tree Interaction Features](https://arxiv.org/abs/1902.09041) ([Paper](https://arxiv.org/pdf/1902.09041.pdf)) `LinkedIn`
10. [In-session Personalization for Talent Search](https://arxiv.org/abs/1809.06488) ([Paper](https://arxiv.org/pdf/1809.06488.pdf)) `LinkedIn`
10. [The AI Behind LinkedIn Recruiter Search and recommendation systems](https://engineering.linkedin.com/blog/2019/04/ai-behind-linkedin-recruiter-search-and-recommendation-systems) `LinkedIn`
11. [Learning Hiring Preferences: The AI Behind LinkedIn Jobs](https://engineering.linkedin.com/blog/2019/02/learning-hiring-preferences--the-ai-behind-linkedin-jobs) `LinkedIn`
11. [Quality Matches Via Personalized AI for Hirer and Seeker Preferences](https://engineering.linkedin.com/blog/2020/quality-matches-via-personalized-ai) `LinkedIn`
12. [Understanding Dwell Time to Improve LinkedIn Feed Ranking](https://engineering.linkedin.com/blog/2020/understanding-feed-dwell-time) `LinkedIn`
13. [Ads Allocation in Feed via Constrained Optimization](https://dl.acm.org/doi/abs/10.1145/3394486.3403391) ([Paper](https://dl.acm.org/doi/pdf/10.1145/3394486.3403391), [Video](https://crossminds.ai/video/5f33697a0576dd25aef288ea/)) `LinkedIn`
14. [Talent Search and Recommendation Systems at LinkedIn](https://arxiv.org/abs/1809.06481) ([Paper](https://arxiv.org/pdf/1809.06481.pdf)) `LinkedIn`
15. [Understanding Dwell Time to Improve LinkedIn Feed Ranking](https://engineering.linkedin.com/blog/2020/understanding-feed-dwell-time) `LinkedIn`
12. [AI at Scale in Bing](https://blogs.bing.com/search/2020_05/AI-at-Scale-in-Bing) `Microsoft`
13. [Query Understanding Engine in Traveloka Universal Search](https://medium.com/traveloka-engineering/query-understanding-engine-in-traveloka-universal-search-410ad3895db7) `Traveloka`
14. [The Secret Sauce Behind Search Personalisation](https://blog.gojekengineering.com/the-secret-sauce-behind-search-personalisation-a856fb83c2f) `GoJek`
15. [Food Discovery with Uber Eats: Building a Query Understanding Engine](https://eng.uber.com/uber-eats-query-understanding/) `Uber`
16. [Neural Code Search: ML-based Code Search Using Natural Language Queries](https://ai.facebook.com/blog/neural-code-search-ml-based-code-search-using-natural-language-queries/) `Facebook`
17. [Bayesian Product Ranking at Wayfair](https://tech.wayfair.com/data-science/2020/01/bayesian-product-ranking-at-wayfair) `Wayfair`
18. [COLD: Towards the Next Generation of Pre-Ranking System](https://arxiv.org/abs/2007.16122) ([Paper](https://arxiv.org/pdf/2007.16122.pdf)) `Alibaba`
19. [Understanding Searches Better Than Ever Before](https://www.blog.google/products/search/search-language-understanding-bert/) ([Paper](https://arxiv.org/pdf/1810.04805.pdf)) `Google`
20. [Shop The Look: Building a Large Scale Visual Shopping System at Pinterest](https://dl.acm.org/doi/abs/10.1145/3394486.3403372) ([Paper](https://dl.acm.org/doi/pdf/10.1145/3394486.3403372), [Video](https://crossminds.ai/video/5f3369790576dd25aef288d7/)) `Pinterest`
21. [GDMix: A Deep Ranking Personalization Framework](https://engineering.linkedin.com/blog/2020/gdmix--a-deep-ranking-personalization-framework) ([Code](https://github.com/linkedin/gdmix)) `LinkedIn`
22. [Bringing Personalized Search to Etsy](https://codeascraft.com/2020/10/29/bringing-personalized-search-to-etsy/) `Etsy`
23. [Building a Better Search Engine for Semantic Scholar](https://medium.com/ai2-blog/building-a-better-search-engine-for-semantic-scholar-ea23a0b661e7) `Allen Institute for AI`
24. [Query Understanding for Natural Language Enterprise Search](https://arxiv.org/abs/2012.06238) ([Paper](https://arxiv.org/pdf/2012.06238.pdf)) `Salesforce`
25. [How We Used Semantic Search to Make Our Search 10x Smarter](https://medium.com/tokopedia-engineering/how-we-used-semantic-search-to-make-our-search-10x-smarter-bd9c7f601821) `Tokopedia`
26. [Powering Search & Recommendations at DoorDash](https://doordash.engineering/2017/07/06/powering-search-recommendations-at-doordash/) `DoorDash`
26. [Things Not Strings: Understanding Search Intent with Better Recall](https://doordash.engineering/2020/12/15/understanding-search-intent-with-better-recall/) `DoorDash`
27. [Query Understanding for Surfacing Under-served Music Content](https://research.atspotify.com/publications/query-understanding-for-surfacing-under-served-music-content/) ([Paper](https://labtomarket.files.wordpress.com/2020/08/cikm2020.pdf)) `Spotify`

## Embeddings
1. [Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba](https://arxiv.org/abs/1803.02349) ([Paper](https://arxiv.org/pdf/1803.02349.pdf)) `Alibaba`
2. [Embeddings@Twitter](https://blog.twitter.com/engineering/en_us/topics/insights/2018/embeddingsattwitter.html) `Twitter` 
3. [Listing Embeddings in Search Ranking](https://medium.com/airbnb-engineering/listing-embeddings-for-similar-listing-recommendations-and-real-time-personalization-in-search-601172f7603e) ([Paper](https://www.kdd.org/kdd2018/accepted-papers/view/real-time-personalization-using-embeddings-for-search-ranking-at-airbnb)) `Airbnb`
4. [Understanding Latent Style](https://multithreaded.stitchfix.com/blog/2018/06/28/latent-style/) `Stitch Fix`
5. [Towards Deep and Representation Learning for Talent Search at LinkedIn](https://arxiv.org/abs/1809.06473) ([Paper](https://arxiv.org/pdf/1809.06473.pdf)) `LinkedIn`
6. [Should we Embed? A Study on Performance of Embeddings for Real-Time Recommendations](https://arxiv.org/abs/1907.06556)([Paper](https://arxiv.org/pdf/1907.06556.pdf)) `Moshbit`
6. [Vector Representation Of Items, Customer And Cart To Build A Recommendation System](https://arxiv.org/abs/1705.06338) ([Paper](https://arxiv.org/pdf/1705.06338.pdf)) `Sears`
7. [Machine Learning for a Better Developer Experience](https://netflixtechblog.com/machine-learning-for-a-better-developer-experience-1e600c69f36c) `Netflix`
8. [Announcing ScaNN: Efficient Vector Similarity Search](https://ai.googleblog.com/2020/07/announcing-scann-efficient-vector.html) ([Paper](https://arxiv.org/pdf/1908.10396.pdf), [Code](https://github.com/google-research/google-research/tree/master/scann)) `Google`
9. [Personalized Store Feed with Vector Embeddings](https://doordash.engineering/2018/04/02/personalized-store-feed-with-vector-embeddings/) `DoorDash`

## Natural Language Processing
1. [Abusive Language Detection in Online User Content](https://dl.acm.org/doi/10.1145/2872427.2883062) ([Paper](http://www.yichang-cs.com/yahoo/WWW16_Abusivedetection.pdf)) `Yahoo`
2. [How Natural Language Processing Helps LinkedIn Members Get Support Easily](https://engineering.linkedin.com/blog/2019/04/how-natural-language-processing-help-support) `LinkedIn`
3. [Building Smart Replies for Member Messages](https://engineering.linkedin.com/blog/2017/10/building-smart-replies-for-member-messages) `LinkedIn`
4. [DeText: A deep NLP Framework for Intelligent Text Understanding](https://engineering.linkedin.com/blog/2020/open-sourcing-detext) ([Code](https://github.com/linkedin/detext)) `LinkedIn`
4. [Smart Reply: Automated Response Suggestion for Email](https://research.google/pubs/pub45189/) ([Paper](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/45189.pdf)) `Google`
5. [Gmail Smart Compose: Real-Time Assisted Writing](https://arxiv.org/abs/1906.00080) ([Paper](https://arxiv.org/pdf/1906.00080.pdf)) `Google`
5. [SmartReply for YouTube Creators](https://ai.googleblog.com/2020/07/smartreply-for-youtube-creators.html) `Google`
6. [Using Neural Networks to Find Answers in Tables](https://ai.googleblog.com/2020/04/using-neural-networks-to-find-answers.html) ([Paper](https://arxiv.org/pdf/2004.02349.pdf)) `Google`
7. [A Scalable Approach to Reducing Gender Bias in Google Translate](https://ai.googleblog.com/2020/04/a-scalable-approach-to-reducing-gender.html) `Google`
8. [Assistive AI Makes Replying Easier](https://www.microsoft.com/en-us/research/group/msai/articles/assistive-ai-makes-replying-easier-2/) `Microsoft`
9. [AI Advances to Better Detect Hate Speech](https://ai.facebook.com/blog/ai-advances-to-better-detect-hate-speech/) `Facebook`
10. [A State-of-the-Art Open Source Chatbot](https://ai.facebook.com/blog/state-of-the-art-open-source-chatbot) ([Paper](https://arxiv.org/pdf/2004.13637.pdf)) `Facebook`
11. [A Highly Efficient, Real-Time Text-to-Speech System Deployed on CPUs](https://ai.facebook.com/blog/a-highly-efficient-real-time-text-to-speech-system-deployed-on-cpus/) `Facebook`
12. [Deep Learning to Translate Between Programming Languages](https://ai.facebook.com/blog/deep-learning-to-translate-between-programming-languages/) ([Paper](https://arxiv.org/abs/2006.03511), [Code](https://github.com/facebookresearch/TransCoder)) `Facebook`
13. [Deploying Lifelong Open-Domain Dialogue Learning](https://arxiv.org/abs/2008.08076) ([Paper](https://arxiv.org/pdf/2008.08076.pdf)) `Facebook`
12. [Goal-Oriented End-to-End Conversational Models with Profile Features in a Real-World Setting](https://www.amazon.science/publications/goal-oriented-end-to-end-chatbots-with-profile-features-in-a-real-world-setting) ([Paper](https://assets.amazon.science/47/03/e0d14dc34d3eb6e0d4ec282067bd/goal-oriented-end-to-end-chatbots-with-profile-features-in-a-real-world-setting.pdf)) `Amazon`
13. [How Gojek Uses NLP to Name Pickup Locations at Scale](https://blog.gojekengineering.com/how-gojek-uses-nlp-to-name-pickup-locations-at-scale-ffdb249d1433) `GoJek`
14. [Give Me Jeans not Shoes: How BERT Helps Us Deliver What Clients Want](https://multithreaded.stitchfix.com/blog/2019/07/15/give-me-jeans/) `Stitch Fix`
15. [The State-of-the-art Open-Domain Chatbot in Chinese and English](http://research.baidu.com/Blog/index-view?id=142) ([Paper](https://arxiv.org/pdf/2006.16779.pdf)) `Baidu`
17. [PEGASUS: A State-of-the-Art Model for Abstractive Text Summarization](https://ai.googleblog.com/2020/06/pegasus-state-of-art-model-for.html) ([Paper](https://arxiv.org/pdf/1912.08777.pdf), [Code](https://github.com/google-research/pegasus)) `Google`
19. [Photon: A Robust Cross-Domain Text-to-SQL System](https://www.aclweb.org/anthology/2020.acl-demos.24/) ([Paper](https://www.aclweb.org/anthology/2020.acl-demos.24.pdf)) ([Demo](http://naturalsql.com)) `Salesforce`		
20. [GeDi: A Powerful New Method for Controlling Language Models](https://blog.einstein.ai/gedi/) ([Paper](https://arxiv.org/abs/2009.06367), [Code](https://github.com/salesforce/GeDi)) `Salesforce`
21. [Applying Topic Modeling to Improve Call Center Operations](https://www.youtube.com/watch?v=kzRR8OjF_eI&t=2s) `RICOH`
22. [WIDeText: A Multimodal Deep Learning Framework](https://medium.com/airbnb-engineering/widetext-a-multimodal-deep-learning-framework-31ce2565880c) `Airbnb`

## Sequence Modelling
1. [Practice on Long Sequential User Behavior Modeling for Click-Through Rate Prediction](https://arxiv.org/abs/1905.09248) ([Paper](https://arxiv.org/pdf/1905.09248.pdf))`Alibaba`
2. [Search-based User Interest Modeling with Sequential Behavior Data for CTR Prediction](https://arxiv.org/abs/2006.05639) ([Paper](https://arxiv.org/pdf/2006.05639.pdf)) `Alibaba`
3. [Deep Learning for Electronic Health Records](https://ai.googleblog.com/2018/05/deep-learning-for-electronic-health.html) ([Paper](https://www.nature.com/articles/s41746-018-0029-1.pdf)) `Google`
4. [Deep Learning for Understanding Consumer Histories](https://engineering.zalando.com/posts/2016/10/deep-learning-for-understanding-consumer-histories.html) ([Paper](https://doogkong.github.io/2017/papers/paper2.pdf)) `Zalando`
5. [Continual Prediction of Notification Attendance with Classical and Deep Networks](https://arxiv.org/abs/1712.07120) ([Paper](https://arxiv.org/pdf/1712.07120.pdf)) `Telefonica`
6. [Using Recurrent Neural Network Models for Early Detection of Heart Failure Onset](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5391725/) ([Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5391725/pdf/ocw112.pdf)) `Sutter Health`
7. [Doctor AI: Predicting Clinical Events via Recurrent Neural Networks](https://arxiv.org/abs/1511.05942) ([Paper](https://arxiv.org/pdf/1511.05942.pdf)) `Sutter Health`
8. [How Duolingo uses AI in every part of its app](https://venturebeat.com/2020/08/18/how-duolingo-uses-ai-in-every-part-of-its-app/) `Duolingo`
9. [Leveraging Online Social Interactions For Enhancing Integrity at Facebook](https://research.fb.com/blog/2020/08/leveraging-online-social-interactions-for-enhancing-integrity-at-facebook/) ([Paper](https://research.fb.com/wp-content/uploads/2020/08/TIES-Temporal-Interaction-Embeddings-For-Enhancing-Social-Media-Integrity-At-Facebook.pdf), [Video](https://crossminds.ai/video/5f3369780576dd25aef288cf/)) `Facebook`


## Computer Vision
1. [Categorizing Listing Photos at Airbnb](https://medium.com/airbnb-engineering/categorizing-listing-photos-at-airbnb-f9483f3ab7e3) `Airbnb`
2. [Amenity Detection and Beyond — New Frontiers of Computer Vision at Airbnb](https://medium.com/airbnb-engineering/amenity-detection-and-beyond-new-frontiers-of-computer-vision-at-airbnb-144a4441b72e) `Airbnb`
3. [Powered by AI: Advancing product understanding and building new shopping experiences](https://ai.facebook.com/blog/powered-by-ai-advancing-product-understanding-and-building-new-shopping-experiences/) `Facebook`
4. [New AI Research to Help Predict COVID-19 Resource Needs From X-rays](https://ai.facebook.com/blog/new-ai-research-to-help-predict-covid-19-resource-needs-from-a-series-of-x-rays/) ([Paper](https://arxiv.org/pdf/2101.04909.pdf), [Model](https://github.com/facebookresearch/CovidPrognosis)) `Facebook`
4. [Creating a Modern OCR Pipeline Using Computer Vision and Deep Learning](https://dropbox.tech/machine-learning/creating-a-modern-ocr-pipeline-using-computer-vision-and-deep-learning) `Dropbox`
5. [How we Improved Computer Vision Metrics by More Than 5% Only by Cleaning Labelling Errors](https://deepomatic.com/en/how-we-improved-computer-vision-metrics-by-more-than-5-percent-only-by-cleaning-labelling-errors/) `Deepomatic`
6. [A Neural Weather Model for Eight-Hour Precipitation Forecasting](https://ai.googleblog.com/2020/03/a-neural-weather-model-for-eight-hour.html) ([Paper](https://arxiv.org/pdf/2003.12140.pdf)) `Google`
7. [Machine Learning-based Damage Assessment for Disaster Relief](https://ai.googleblog.com/2020/06/machine-learning-based-damage.html) ([Paper](https://arxiv.org/pdf/1910.06444.pdf)) `Google`
8. [RepNet: Counting Repetitions in Videos](https://ai.googleblog.com/2020/06/repnet-counting-repetitions-in-videos.html) ([Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Dwibedi_Counting_Out_Time_Class_Agnostic_Video_Repetition_Counting_in_the_CVPR_2020_paper.pdf)) `Google`
9. [Converting Text to Images for Product Discovery](https://www.amazon.science/blog/converting-text-to-images-for-product-discovery) ([Paper](https://assets.amazon.science/4c/76/5830542547b7a11089ce3af943b4/scipub-972.pdf)) `Amazon`
10. [How Disney Uses PyTorch for Animated Character Recognition](https://medium.com/pytorch/how-disney-uses-pytorch-for-animated-character-recognition-a1722a182627) `Disney`
12. [Image Captioning as an Assistive Technology](https://www.ibm.com/blogs/research/2020/07/image-captioning-assistive-technology/) ([Video](https://ivc.ischool.utexas.edu/~yz9244/VizWiz_workshop/videos/MMTeam-oral.mp4)) `IBM`
13. [AI for AG: Production machine learning for agriculture](https://medium.com/pytorch/ai-for-ag-production-machine-learning-for-agriculture-e8cfdb9849a1) `Blue River`
14. [AI for Full-Self Driving at Tesla](https://youtu.be/hx7BXih7zx8?t=513) `Tesla`
15. [On-device Supermarket Product Recognition](https://ai.googleblog.com/2020/07/on-device-supermarket-product.html) `Google`
16. [Using Machine Learning to Detect Deficient Coverage in Colonoscopy Screenings](https://ai.googleblog.com/2020/08/using-machine-learning-to-detect.html) ([Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9097918)) `Google`
17. [Shop The Look: Building a Large Scale Visual Shopping System at Pinterest](https://dl.acm.org/doi/abs/10.1145/3394486.3403372) ([Paper](https://dl.acm.org/doi/pdf/10.1145/3394486.3403372), [Video](https://crossminds.ai/video/5f3369790576dd25aef288d7/)) `Pinterest`
18. [Developing Real-Time, Automatic Sign Language Detection for Video Conferencing](https://ai.googleblog.com/2020/10/developing-real-time-automatic-sign.html) ([Paper](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/2eaf0d18ec6bef00d7dd88f39dd4f9ff13eeeeb2.pdf)) `Google`
19. [Vision-based Price Suggestion for Online Second-hand Items](https://arxiv.org/abs/2012.06009) ([Paper](https://arxiv.org/pdf/2012.06009.pdf)) `Alibaba`
20. [Making machines recognize and transcribe conversations in meetings using audio and video](https://www.microsoft.com/en-us/research/blog/making-machines-recognize-and-transcribe-conversations-in-meetings-using-audio-and-video/) `Microsoft`


## Reinforcement Learning
1. [Deep Reinforcement Learning for Sponsored Search Real-time Bidding](https://arxiv.org/abs/1803.00259) ([Paper](https://arxiv.org/pdf/1803.00259.pdf)) `Alibaba`
2. [Dynamic Pricing on E-commerce Platform with Deep Reinforcement Learning](https://arxiv.org/abs/1912.02572) ([Paper](https://arxiv.org/pdf/1912.02572.pdf)) `Alibaba`
3. [Budget Constrained Bidding by Model-free Reinforcement Learning in Display Advertising](https://arxiv.org/abs/1802.08365) ([Paper](https://arxiv.org/pdf/1802.08365.pdf)) `Alibaba`
4. [Productionizing Deep Reinforcement Learning with Spark and MLflow](https://databricks.com/session_na20/productionizing-deep-reinforcement-learning-with-spark-and-mlflow) `Zynga`
5. [Deep Reinforcement Learning in Production Part1](https://towardsdatascience.com/deep-reinforcement-learning-in-production-7e1e63471e2) [Part 2](https://towardsdatascience.com/deep-reinforcement-learning-in-production-part-2-personalizing-user-notifications-812a68ce2355) `Zynga`
5. [Building AI Trading Systems](https://dennybritz.com/blog/ai-trading/) `Denny Britz`
6. [Reinforcement Learning for On-Demand Logistics](https://doordash.engineering/2018/09/10/reinforcement-learning-for-on-demand-logistics/) `DoorDash`

## Anomaly Detection
1. [Detecting Performance Anomalies in External Firmware Deployments](https://netflixtechblog.com/detecting-performance-anomalies-in-external-firmware-deployments-ed41b1bfcf46) `Netflix`
2. [Detecting and Preventing Abuse on LinkedIn using Isolation Forests](https://engineering.linkedin.com/blog/2019/isolation-forest) ([Code](https://github.com/linkedin/isolation-forest)) `LinkedIn`
3. [Preventing Abuse Using Unsupervised Learning](https://databricks.com/session_na20/preventing-abuse-using-unsupervised-learning) `LinkedIn`
4. [The Technology Behind Fighting Harassment on LinkedIn](https://engineering.linkedin.com/blog/2020/fighting-harassment) `LinkedIn`
4. [Uncovering Insurance Fraud Conspiracy with Network Learning](https://arxiv.org/abs/2002.12789) ([Paper](https://arxiv.org/pdf/2002.12789.pdf)) `Ant Financial`
5. [How Does Spam Protection Work on Stack Exchange?](https://stackoverflow.blog/2020/06/25/how-does-spam-protection-work-on-stack-exchange/) `Stack Exchange`
6. [Auto Content Moderation in C2C e-Commerce](https://www.usenix.org/conference/opml20/presentation/ueta) `Mercari`
7. [Blocking Slack Invite Spam With Machine Learning](https://slack.engineering/blocking-slack-invite-spam-with-machine-learning/) `Slack`
8. [Cloudflare Bot Management: Machine Learning and More](https://blog.cloudflare.com/cloudflare-bot-management-machine-learning-and-more/) `Cloudflare`
8. [Anomalies in Oil Temperature Variations in a Tunnel Boring Machine](https://www.youtube.com/watch?v=YV_uLLhPRAk) `SENER`
9. [Using Anomaly Detection to Monitor Low-Risk Bank Customers](https://www.youtube.com/watch?v=MExokMM_Bp4&t=3s) `Rabobank`
10. [Fighting fraud with Triplet Loss](https://tech.olx.com/fighting-fraud-with-triplet-loss-86e5f79c7a3e) `OLX Group`
11. [Facebook is Now Using AI to Sort Content for Quicker Moderation](https://www.theverge.com/2020/11/13/21562596/facebook-ai-moderation) ([Alternative](https://venturebeat.com/2020/11/13/facebooks-redoubled-ai-efforts-wont-stop-the-spread-of-harmful-content/)) `Facebook`
12. How AI is getting better at detecting hate speech [Part 1](https://ai.facebook.com/blog/how-ai-is-getting-better-at-detecting-hate-speech/), [Part 2](https://ai.facebook.com/blog/heres-how-were-using-ai-to-help-detect-misinformation/), [Part 3](https://ai.facebook.com/blog/training-ai-to-detect-hate-speech-in-the-real-world/), [Part 4](https://ai.facebook.com/blog/how-facebook-uses-super-efficient-ai-models-to-detect-hate-speech/) `Facebook`
13. [Deep Anomaly Detection with Spark and Tensorflow](https://databricks.com/session_eu19/deep-anomaly-detection-from-research-to-production-leveraging-spark-and-tensorflow) [(Hopsworks Video](https://www.youtube.com/watch?v=TgXVU8DSyCQ)) `Swedbank`, `Hopsworks`

## Graph
1. [Building The LinkedIn Knowledge Graph](https://engineering.linkedin.com/blog/2016/10/building-the-linkedin-knowledge-graph) `LinkedIn`
2. [Retail Graph — Walmart’s Product Knowledge Graph](https://medium.com/walmartlabs/retail-graph-walmarts-product-knowledge-graph-6ef7357963bc) `Walmart`
3. [Food Discovery with Uber Eats: Using Graph Learning to Power Recommendations](https://eng.uber.com/uber-eats-graph-learning/) `Uber`
4. [AliGraph: A Comprehensive Graph Neural Network Platform](https://arxiv.org/abs/1902.08730) ([Paper](https://arxiv.org/pdf/1902.08730.pdf)) `Alibaba`
5. [Scaling Knowledge Access and Retrieval at Airbnb](https://medium.com/airbnb-engineering/scaling-knowledge-access-and-retrieval-at-airbnb-665b6ba21e95) `Airbnb`
6. [Contextualizing Airbnb by Building Knowledge Graph](https://medium.com/airbnb-engineering/contextualizing-airbnb-by-building-knowledge-graph-b7077e268d5a) `Airbnb`
6. [Traffic Prediction with Advanced Graph Neural Networks](https://deepmind.com/blog/article/traffic-prediction-with-advanced-graph-neural-networks) `DeepMind`
7. [SimClusters: Community-Based Representations for Recommendations](https://dl.acm.org/doi/10.1145/3394486.3403370) ([Paper](https://dl.acm.org/doi/pdf/10.1145/3394486.3403370), [Video](https://crossminds.ai/video/5f3369790576dd25aef288d5/)) `Twitter`

## Optimization
1. [How Trip Inferences and Machine Learning Optimize Delivery Times on Uber Eats](https://eng.uber.com/uber-eats-trip-optimization/) `Uber`
2. [Next-Generation Optimization for Dasher Dispatch at DoorDash](https://doordash.engineering/2020/02/28/next-generation-optimization-for-dasher-dispatch-at-doordash/) `DoorDash`
3. [Matchmaking in Lyft Line (Part 1)](https://eng.lyft.com/matchmaking-in-lyft-line-9c2635fe62c4) [(Part 2)](https://eng.lyft.com/matchmaking-in-lyft-line-691a1a32a008) [(Part 3)](https://eng.lyft.com/matchmaking-in-lyft-line-part-3-d8f9497c0e51) `Lyft`
4. [The Data and Science behind GrabShare Carpooling](https://ieeexplore.ieee.org/document/8259801) (**PAPER NEEDED**) `Grab`
5. [Optimization of Passengers Waiting Time in Elevators Using Machine Learning](https://www.youtube.com/watch?v=vXndCC89BCw&t=4s) `Thyssen Krupp AG`
6. [Think Out of The Package: Recommending Package Types for E-commerce Shipments](https://www.amazon.science/publications/think-out-of-the-package-recommending-package-types-for-e-commerce-shipments) ([Paper](https://assets.amazon.science/0c/6c/9d0986b94bef92d148f0ac0da1ea/think-out-of-the-package-recommending-package-types-for-e-commerce-shipments.pdf)) `Amazon`
7. [Optimizing DoorDash’s Marketing Spend with Machine Learning](https://doordash.engineering/2020/07/31/optimizing-marketing-spend-with-ml/) `DoorDash`


## Information Extraction
1. [Unsupervised Extraction of Attributes and Their Values from Product Description](https://www.aclweb.org/anthology/I13-1190/) ([Paper](https://www.aclweb.org/anthology/I13-1190.pdf)) `Rakuten`
2. [Information Extraction from Receipts with Graph Convolutional Networks](https://nanonets.com/blog/information-extraction-graph-convolutional-networks/) `Nanonets`
3. [Using Machine Learning to Index Text from Billions of Images](https://dropbox.tech/machine-learning/using-machine-learning-to-index-text-from-billions-of-images) `Dropbox`
4. [Extracting Structured Data from Templatic Documents](https://ai.googleblog.com/2020/06/extracting-structured-data-from.html) ([Paper](https://www.aclweb.org/anthology/I13-1190.pdf)) `Google`
5. [AutoKnow: self-driving knowledge collection for products of thousands of types](https://www.amazon.science/publications/autoknow-self-driving-knowledge-collection-for-products-of-thousands-of-types) ([Paper](https://arxiv.org/pdf/2006.13473.pdf), [Video](https://crossminds.ai/video/5f3369730576dd25aef288a6/)) `Amazon`
6. [One-shot Text Labeling using Attention and Belief Propagation for Information Extraction](https://arxiv.org/abs/2009.04153) ([Paper](https://arxiv.org/pdf/2009.04153.pdf)) `Alibaba`

## Weak Supervision
1. [Snorkel DryBell: A Case Study in Deploying Weak Supervision at Industrial Scale](https://dl.acm.org/doi/abs/10.1145/3299869.3314036) ([Paper](https://dl.acm.org/doi/pdf/10.1145/3299869.3314036)) `Google`
2. [Osprey: Weak Supervision of Imbalanced Extraction Problems without Code](https://dl.acm.org/doi/abs/10.1145/3329486.3329492) ([Paper](https://ajratner.github.io/assets/papers/Osprey_DEEM.pdf)) `Intel`
3. [Overton: A Data System for Monitoring and Improving Machine-Learned Products](https://arxiv.org/abs/1909.05372) ([Paper](https://arxiv.org/pdf/1909.05372.pdf)) `Apple`
4. [Bootstrapping Conversational Agents with Weak Supervision](https://www.aaai.org/ojs/index.php/AAAI/article/view/5011) ([Paper](https://arxiv.org/pdf/1812.06176.pdf)) `IBM`

## Generation
1. [Better Language Models and Their Implications](https://openai.com/blog/better-language-models/) ([Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf))`OpenAI`
1. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) ([Paper](https://arxiv.org/pdf/2005.14165.pdf)) ([GPT-3 Blog post](https://openai.com/blog/openai-api/)) `OpenAI`
2. [Image GPT](https://openai.com/blog/image-gpt/) ([Paper](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf), [Code](https://github.com/openai/image-gpt)) `OpenAI`
2. [Deep Learned Super Resolution for Feature Film Production](https://graphics.pixar.com/library/SuperResolution/) ([Paper](https://graphics.pixar.com/library/SuperResolution/paper.pdf)) `Pixar`
3. [Unit Test Case Generation with Transformers](https://arxiv.org/pdf/2009.05617.pdf) `Microsoft`

## Audio
1. [Improving On-Device Speech Recognition with VoiceFilter-Lite](https://ai.googleblog.com/2020/11/improving-on-device-speech-recognition.html) ([Paper](https://arxiv.org/pdf/2009.04323.pdf))`Google`
2. [The Machine Learning Behind Hum to Search](https://ai.googleblog.com/2020/11/the-machine-learning-behind-hum-to.html) `Google`

## Validation and A/B Testing
1. [The Reusable Holdout: Preserving Validity in Adaptive Data Analysis](https://ai.googleblog.com/2015/08/the-reusable-holdout-preserving.html) ([Paper](https://science.sciencemag.org/content/sci/349/6248/636.full.pdf)) `Google`
2. [Twitter Experimentation: Technical Overview](https://blog.twitter.com/engineering/en_us/a/2015/twitter-experimentation-technical-overview.html) `Twitter`
5. [Experimenting to Solve Cramming](https://blog.twitter.com/engineering/en_us/topics/insights/2017/Experimenting-To-Solve-Cramming.html) `Twitter`
6. [Building an Intelligent Experimentation Platform with Uber Engineering](https://eng.uber.com/experimentation-platform/) `Uber`
7. [Analyzing Experiment Outcomes: Beyond Average Treatment Effects](https://eng.uber.com/analyzing-experiment-outcomes/) `Uber`
8. [Under the Hood of Uber’s Experimentation Platform](https://eng.uber.com/xp/) `Uber`
6. [Announcing a New Framework for Designing Optimal Experiments with Pyro](https://eng.uber.com/oed-pyro-release/) ([Paper](https://papers.nips.cc/paper/9553-variational-bayesian-optimal-experimental-design.pdf)) ([Paper](https://arxiv.org/pdf/1911.00294.pdf)) `Uber`
7. [Enabling 10x More Experiments with Traveloka Experiment Platform](https://medium.com/traveloka-engineering/enabling-10x-more-experiments-with-traveloka-experiment-platform-8cea13e952c) `Traveloka`
8. [Large Scale Experimentation at Stitch Fix](https://multithreaded.stitchfix.com/blog/2020/07/07/large-scale-experimentation/) ([Paper](http://proceedings.mlr.press/v89/schmit19a/schmit19a.pdf)) `Stitch Fix`
9. [Multi-Armed Bandits and the Stitch Fix Experimentation Platform](https://multithreaded.stitchfix.com/blog/2020/08/05/bandits/) `Stitch Fix`
10. [Experimentation with Resource Constraints](https://multithreaded.stitchfix.com/blog/2020/11/18/virtual-warehouse/) `Stitch Fix`
10. [Modeling Conversion Rates and Saving Millions Using Kaplan-Meier and Gamma Distributions](https://better.engineering/modeling-conversion-rates-and-saving-millions-of-dollars-using-kaplan-meier-and-gamma-distributions/) ([Code](https://github.com/better/convoys)) `Better`
11. [It’s All A/Bout Testing: The Netflix Experimentation Platform](https://netflixtechblog.com/its-all-a-bout-testing-the-netflix-experimentation-platform-4e1ca458c15) `Netflix`
11. [Computational Causal Inference at Netflix](https://netflixtechblog.com/computational-causal-inference-at-netflix-293591691c62) ([Paper](https://arxiv.org/pdf/2007.10979.pdf)) `Netflix`
12. [Key Challenges with Quasi Experiments at Netflix](https://netflixtechblog.com/key-challenges-with-quasi-experiments-at-netflix-89b4f234b852) `Netflix`
13. [Constrained Bayesian Optimization with Noisy Experiments](https://research.fb.com/publications/constrained-bayesian-optimization-with-noisy-experiments/) ([Paper](https://arxiv.org/pdf/1706.07094.pdf)) `Facebook`
16. [Detecting Interference: An A/B Test of A/B Tests](https://engineering.linkedin.com/blog/2019/06/detecting-interference--an-a-b-test-of-a-b-tests) `LinkedIn`
15. [Making the LinkedIn experimentation engine 20x faster](https://engineering.linkedin.com/blog/2020/making-the-linkedin-experimentation-engine-20x-faster) `LinkedIn`
15. [Our Evolution Towards T-REX: The Prehistory of Experimentation Infrastructure at LinkedIn](https://engineering.linkedin.com/blog/2020/our-evolution-towards-t-rex--the-prehistory-of-experimentation-i) `LinkedIn`
16. [How to Use Quasi-experiments and Counterfactuals to Build Great Products](https://engineering.shopify.com/blogs/engineering/using-quasi-experiments-counterfactuals) `Shopify`
17. [Improving Experimental Power through Control Using Predictions as Covariate](https://doordash.engineering/2020/06/08/improving-experimental-power-through-control-using-predictions-as-covariate-cupac/) `Doordash`
17. [Supporting Rapid Product Iteration with an Experimentation Analysis Platform](https://doordash.engineering/2020/09/09/experimentation-analysis-platform-mvp/) `DoorDash`
17. [Improving Online Experiment Capacity by 4X with Parallelization and Increased Sensitivity](https://doordash.engineering/2020/10/07/improving-experiment-capacity-by-4x/) `DoorDash`
18. [Leveraging Causal Modeling to Get More Value from Flat Experiment Results](https://doordash.engineering/2020/09/18/causal-modeling-to-get-more-value-from-flat-experiment-results/) `DoorDash`
25. [Iterating Real-time Assignment Algorithms Through Experimentation](https://doordash.engineering/2020/12/08/optimizing-real-time-algorithms-experimentation/) `DoorDash`
25. [Running Experiments with Google Adwords for Campaign Optimization](https://doordash.engineering/2021/02/05/google-adwords-campaign-optimization/) `DoorDash`
18. [Spotify’s New Experimentation Platform (Part 1)](https://engineering.atspotify.com/2020/10/29/spotifys-new-experimentation-platform-part-1/) [(Part 2)](https://engineering.atspotify.com/2020/11/02/spotifys-new-experimentation-platform-part-2/) `Spotify`
19. [Overlapping Experiment Infrastructure: More, Better, Faster Experimentation](https://research.google/pubs/pub36500/) ([Paper](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/36500.pdf)) `Google`
20. [Experimentation Platform at Zalando: Part 1 - Evolution](https://engineering.zalando.com/posts/2021/01/experimentation-platform-part1.html) `Zalando`
21. [Scaling Airbnb’s Experimentation Platform](https://medium.com/airbnb-engineering/https-medium-com-jonathan-parks-scaling-erf-23fd17c91166) `Airbnb`
22. [Designing Experimentation Guardrails](https://medium.com/airbnb-engineering/designing-experimentation-guardrails-ed6a976ec669) `Airbnb`
22. [Reliable and Scalable Feature Toggles and A/B Testing SDK at Grab](https://engineering.grab.com/feature-toggles-ab-testing) `Grab`
23. [Meet Wasabi, an Open Source A/B Testing Platform](https://www.intuit.com/blog/technology/engineering/meet-wasabi-an-open-source-ab-testing-platform/) ([Code](https://github.com/intuit/wasabi)) `Intuit`
24. [Building Pinterest’s A/B Testing Platform](https://medium.com/pinterest-engineering/building-pinterests-a-b-testing-platform-ab4934ace9f4) `Pinterest`

## Model Management
1. [Runway - Model Lifecycle Management at Netflix](https://www.usenix.org/conference/opml20/presentation/cepoi) `Netflix`
2. [Overton: A Data System for Monitoring and Improving Machine-Learned Products](https://arxiv.org/abs/1909.05372) ([Paper](https://arxiv.org/pdf/1909.05372.pdf)) `Apple`
3. [Managing ML Models @ Scale - Intuit’s ML Platform](https://www.usenix.org/conference/opml20/presentation/wenzel) `Intuit`
4. [Operationalizing Machine Learning—Managing Provenance from Raw Data to Predictions](https://vimeo.com/274396495) `Comcast`

## Efficiency
1. [GrokNet: Unified Computer Vision Model Trunk and Embeddings For Commerce](https://ai.facebook.com/research/publications/groknet-unified-computer-vision-model-trunk-and-embeddings-for-commerce/) ([Paper](https://scontent-sea1-1.xx.fbcdn.net/v/t39.8562-6/99353320_565175057533429_3886205100842024960_n.pdf?_nc_cat=110&_nc_sid=ae5e01&_nc_ohc=WQBaZy1gnmUAX8Ecqtt&_nc_ht=scontent-sea1-1.xx&oh=cab2f11dd9154d817149cb73e8b692a8&oe=5F5A3778)) `Facebook`
2. [Permute, Quantize, and Fine-tune: Efficient Compression of Neural Networks](https://arxiv.org/abs/2010.15703) ([Paper](https://arxiv.org/pdf/2010.15703.pdf)) `Uber`

## Ethics
1. [Building Inclusive Products Through A/B Testing](https://engineering.linkedin.com/blog/2020/building-inclusive-products-through-a-b-testing) ([Paper](https://arxiv.org/pdf/2002.05819.pdf)) `LinkedIn`
2. [LiFT: A Scalable Framework for Measuring Fairness in ML Applications](https://engineering.linkedin.com/blog/2020/lift-addressing-bias-in-large-scale-ai-applications) ([Paper](https://arxiv.org/pdf/2008.07433.pdf)) `LinkedIn`

## Infra
1. [Reengineering Facebook AI’s Deep Learning Platforms for Interoperability](https://ai.facebook.com/blog/reengineering-facebook-ais-deep-learning-platforms-for-interoperability) `Facebook`

## Platforms
1. [Managing ML Models @ Scale - Intuit’s ML Platform](https://www.usenix.org/conference/opml20/presentation/wenzel) `Intuit`
2. [Operationalizing Machine Learning—Managing Provenance from Raw Data to Predictions](https://vimeo.com/274396495) `Comcast`
3. [Big Data Machine Learning Platform at Pinterest](https://www.slideshare.net/Alluxio/pinterest-big-data-machine-learning-platform-at-pinterest) `Pinterest`
4. [Real-time Machine Learning Inference Platform at Zomato](https://www.youtube.com/watch?v=0-3ES1vzW14) `Zomato`
5. [Meet Michelangelo: Uber’s Machine Learning Platform](https://eng.uber.com/michelangelo-machine-learning-platform/) `Uber`
6. [Building Flexible Ensemble ML Models with a Computational Graph](https://doordash.engineering/2021/01/26/computational-graph-machine-learning-ensemble-model-support/) `DoorDash`


## Practices
1. [Practical Recommendations for Gradient-Based Training of Deep Architectures](https://arxiv.org/abs/1206.5533) ([Paper](https://arxiv.org/pdf/1206.5533.pdf)) `Yoshua Bengio`
2. [Machine Learning: The High Interest Credit Card of Technical Debt](https://research.google/pubs/pub43146/) ([Paper](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/43146.pdf)) ([Paper](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf)) `Google`
3. [Rules of Machine Learning: Best Practices for ML Engineering](https://developers.google.com/machine-learning/guides/rules-of-ml) `Google`
5. [On Challenges in Machine Learning Model Management](http://sites.computer.org/debull/A18dec/p5.pdf) `Amazon`
6. [Machine Learning in Production: The Booking.com Approach](https://booking.ai/https-booking-ai-machine-learning-production-3ee8fe943c70) `Booking`
7. [150 Successful Machine Learning Models: 6 Lessons Learned at Booking.com](https://www.kdd.org/kdd2019/accepted-papers/view/150-successful-machine-learning-models-6-lessons-learned-at-booking.com) ([Paper](https://dl.acm.org/doi/pdf/10.1145/3292500.3330744)) `Booking`
8. [Successes and Challenges in Adopting Machine Learning at Scale at a Global Bank](https://www.youtube.com/watch?v=QYQKG5OcwEI) `Rabobank`
9. [Challenges in Deploying Machine Learning: a Survey of Case Studies](https://arxiv.org/abs/2011.09926) ([Paper](https://arxiv.org/pdf/2011.09926.pdf)) `Cambridge`

## Team structure
1. [Engineers Shouldn’t Write ETL: A Guide to Building a High Functioning Data Science Department](https://multithreaded.stitchfix.com/blog/2016/03/16/engineers-shouldnt-write-etl/) `Stitch Fix`
2. [Beware the Data Science Pin Factory: The Power of the Full-Stack Data Science Generalist](https://multithreaded.stitchfix.com/blog/2019/03/11/FullStackDS-Generalists/) `Stitch Fix`
3. [Cultivating Algorithms: How We Grow Data Science at Stitch Fix](https://cultivating-algos.stitchfix.com) `StitchFix`
4. [Analytics at Netflix: Who We Are and What We Do](https://netflixtechblog.com/analytics-at-netflix-who-we-are-and-what-we-do-7d9c08fe6965) `Netflix`

## Fails
1. [160k+ High School Students Will Graduate Only If a Model Allows Them to](http://positivelysemidefinite.com/2020/06/160k-students.html) `International Baccalaureate`
2. [When It Comes to Gorillas, Google Photos Remains Blind](https://www.wired.com/story/when-it-comes-to-gorillas-google-photos-remains-blind/) `Google`
3. [An Algorithm That ‘Predicts’ Criminality Based on a Face Sparks a Furor](https://www.wired.com/story/algorithm-predicts-criminality-based-face-sparks-furor/) `Harrisburg University`
4. [It's Hard to Generate Neural Text From GPT-3 About Muslims](https://twitter.com/abidlabs/status/1291165311329341440) `OpenAI`
5. [A British AI Tool to Predict Violent Crime Is Too Flawed to Use](https://www.wired.co.uk/article/police-violence-prediction-ndas) `United Kingdom`
6. More in [awful-ai](https://github.com/daviddao/awful-ai)

<br>

**P.S., Want a summary of ML advancements?** Get up to speed with survey papers 👉[`ml-surveys`](https://github.com/eugeneyan/ml-surveys)
