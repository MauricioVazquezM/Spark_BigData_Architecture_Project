# Spark BigData Architecture Project

<p align="center">
  <img width="800" height="500" src="https://double.cloud/assets/blog/articles/what-is-data-stream-shema1.png.webp">
</p>

## Team
- [Santiago Olver Moreno](https://github.com/SantiOlvera), Data Science and Actuarial Science Double Bachelor Program student at ITAM.
- [Mauricio Vázquez Moran](https://github.com/MauricioVazquezM), Data Science and Actuarial Science Double Bachelor Program student at ITAM.

## Problem Definition and Motivation
In the field of Data Science, the processing of massive batch information for training machine learning models is crucial. While the architecture is predetermined, many factors, particularly those related to runtime, significantly influence the response time necessary to meet client expectations. Consequently, mastering aspects related to the architecture of the platform where our big data applications are executed becomes paramount.
<br>
The motivation for this project is to develop an application using Spark Structured Streaming and/or Kafka to capture real-time data. Once sufficient information is retrieved from the input stream, the goal is to train either a classification or linear regression model and test the model's quality with data that continues to arrive in real time. The challenge involves not only capturing data in real time for training but also receiving new data for making predictions using the built model. Data from disk will be accepted with a certain capture periodicity, but applications that manage real-time data capture will be evaluated more favorably.
<br>
Additionally, the project requires performing comparisons of execution times using the local ITAM cluster, Spark standalone, and cloud infrastructure (Databricks) to demonstrate, with the same application, the performance differences across various platforms.