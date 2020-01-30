## Advanced Customer Analytics at Capernaum Ltd using Machine learning:
## End to End pipeline
(Intricacies not included as it is confidential)


### Objective:
**Customer behavioural segmentation:** To understand customer behaviour of Capernaum Ltd (User base: 13 million approx.) and categorize them based on **demography**, **buying behaviour to unearth critical insights**, facilitate **capture/retention** and build data infrastructure(**Customer analytics record/Customer Data Platform**) from scratch.



### Stakeholders:

-  **CEO**
-  **CTO**
-  **Manager**
-  **Marketing team**

### Implementation Team:

-  **CTO**
-  **Manager**
-  **Myself**

### Business questions: 

What are the signals of a 
-  **Useful transactional user**
-  **Unusable transactional user**
-  **Engaged user**
-  **Multi-course user**
-  **Graduate user**


### Stages:

1. **Business Understanding** (Deciding effective KPIs)
2. **Data Understanding** (Transactional DB/Google Analytics/Exploratory Data Analysis)
3. **Data Extraction** (SQL/API calls)
4. **Data Preparation/Feature Engineering** (Google sheets/Advanced Excel)
5. **Modelling** (Machine learning: K-Means/Agglomerative hierarchical/Gaussian mixture model)
6. **Evaluation** (Elbow/Silhouette)
7. **Data visualization/Presentation** (Insights/funnel metrics)
8. **Deployment** (Dendrograms/Customer Analytics record)


### KPI selection process:
1. **Determine the key drivers of the business**
2. **Define qualitative objective**
3. **Define metric**
4. **Decompose that metric into qualitative problem areas**
5. **Iteratively solve/measure progress against that metric**


**eg:**  
**Goal:** Increase Site traffic  
**Indicator**---> Loyalty  
**Quantifiable measurement/data point:** Count of returns/ Avg time taken to return to player pages



### Approach:



1. **Determination of Macro segments** 

We first decided the macro segments as mentioned below

a. **Loyalty**

b. **Engagement**

c. **Interest**

d. **Value**


Brainstorming and selecting suitable/effective KPIs (base/derived) under each segment was carried out.

Eg: **Loyalty**

**KPI selected:** 
1. Count of returns
2. Average time to return to player pages
 
Each of the above segments was broken down into several clusters and further into micro clusters if required (to attain more visibility/insights)





2. **Data Extraction**
Data Extraction was performed from MySQL hosted on AWS and Google Analytics API

Total KPIs selected: 15

Dataset: 10s of GB


3. **Modelling**
I used k-means clustering to create and derive the optimal number of clusters and understand the underlying customer segments.  
	
<!-- 	
**Algorithm Overview**
![Image of flowchart](https://github.com/Tanay7/Machine-Learning/blob/master/Customer_segmentation/Images/ggg.png)
 --> 


4. **Visualization:**
Python Clustering Screenshot:

![Image of flowchart](https://github.com/Tanay7/Machine-Learning/blob/master/Customer_segmentation/Images/5a.PNG)

Cluster/Class Description:
![Image of flowchart](https://github.com/Tanay7/Machine-Learning/blob/master/Customer_segmentation/Images/7a.PNG)
<!-- 
![Image of flowchart](https://github.com/Tanay7/Machine-Learning/blob/master/Customer_segmentation/Images/8a.PNG)
 --> 

**Customer Data Platform**  

![Image of flowchart](https://github.com/Tanay7/Machine-Learning/blob/master/Customer_segmentation/Images/13a.PNG)

 
![Image of flowchart](https://github.com/Tanay7/Machine-Learning/blob/master/Customer_segmentation/Images/9a.PNG)


<!-- 
**Clustered User Ids:**

![Image of flowchart](https://github.com/Tanay7/Machine-Learning/blob/master/Customer_segmentation/Images/2a.png)

**Dendrogram sample:**
![Image of flowchart](https://github.com/Tanay7/Machine-Learning/blob/master/Customer_segmentation/Images/6a.jpg)
 --> 
 
 	
**Preliminary code link:**

[K means](https://github.com/Tanay7/Machine-Learning/blob/master/Customer_segmentation/K%20Means%20code.ipynb)


 
 
## Insights: 


Eg **Value** (refer Value graphs.pdf) (Dummy data)

![Image of flowchart](https://github.com/Tanay7/Machine-Learning/blob/master/Customer_segmentation/Images/11a.PNG)

![Image of flowchart](https://github.com/Tanay7/Machine-Learning/blob/master/Customer_segmentation/Images/12a.PNG)

As you can see from the pdf, the users with no Value (no propensity to spend) on the site remains the same which is justifiable as Capernaum is dedicated to providing free education and hence targets the users with negligible tendency to spend/purchase.  




## Challenges encountered during my journey
	
**Difficulties in scoping the project during the initial stages.**\
eg Changes in business expectations, fickleness in the selection of KPI's 
 

**Simply throwing a standard Machine learning algo at a particular business problem is not enough.  Each business scenario is different.**\
eg: In our business, there is a huge difference between users who made '0' product purchase than the users who made just '1' purchase which are in turn way different than the ones who made '>2-3' purchases. A standard machine learning algo (Clustering) cannot take this into account.

**Market dynamics: Instability of segments over time**\
eg: Structural change which leads to segment creep and membership migration as the Users move from one segment to another.


**Frequent Google sheet crash:**\
eg Cannot paste 10 million cells in one sheet.

         
**Extensive manual work involved:**\
eg: Copy/pasting millions of rows/slow loading times while performing daily/weekly/monthly segmentation.


**Limitations of MYSQL database version (Ver 5.0) on AWS**\
eg. Some analytical functions (LEAD/LAG) could not be used out of the box.


**Frequent MYSQL database memory table crash/lag which caused slow query execution to fetch results:**\
eg 2-3 hour wait period 

**Incomplete inclusion of Demographic data:**\
eg : Age, gender were not fully included due to data export limitations of Google Analytics


## Impact:  

1. Implemented entirely using Open source tools hence negligible implementation cost.  
2. Offered Capernaum Ltd the ability to leverage data to discover consumers’ intent.  
3. Personalized/timely marketing campaigns for optimal customer experience now possible.     





