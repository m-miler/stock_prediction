# Content of Project
* [General info](#general-info)
* [Technologies](#technologies)
* [Environment](#environment)
* [Setup](#setup)

## General Info
Stock Prediction is an application for creating simple machine learning and deep learning models for time series prediction. 

## Technologies
<ul>
<li>Python 3.10</li>
<li>FastAPI 0.110.0</li>
<li>SQLAlchemy</li>
<li>Docker</li>
<li>PostgreSQL</li>
<li>Pytest</li>
</ul>

## Setup
1. Clone GitHub repository 
``` 
git clone https://github.com/m-miler/investment_portfolios.git
```
2. Install docker and docker-compose then run in project file.
```
  docker compose -f .\devOps\docker-compose-dev.yml up -d --build
```

> [!NOTE]
> This application works together with stock_web_scrapper app. 
> Please see [stock_web_scrapper setup](https://github.com/m-miler/stock_web_scrapper)