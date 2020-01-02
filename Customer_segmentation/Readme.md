## Customer behavioural segmentation at Capernaum Ltd using Machine learning: End to End pipeline
(Intricacies not included as it is confidential)


### Objective:
 To understand customer behaviour of Capernaum Ltd (User base: 13 million approx.) and categorize them based on **demography**, **buying behaviour to unearth critical insights**, facilitate **capture/retention** and build data infrastructure(**Customer analytics record/Customer Data Platform**) from scratch.

### Stakeholders:

1. **CEO**
2. **CTO**
3. **Manager**
4. **Marketing team**

### Business questions: 

What are the signals of a 
1. **Useful transactional user**
2. **Unusable transactional user**
3. **Engaged user**
4. **Multi-course user**
5. **Graduate user**


### Stages:

1. **Business Understanding** (Brainstorming with senior management /Deciding effective KPIs)
2. **Data Understanding** (DB tables/ Google Analytics metrics)
3. **Data Extraction** (SQL for transactional data/Google Analytics API for demographic data)
4. **Data Preparation/Feature Engineering** (Google sheets/Advanced Excel: Pivots/VLOOKUPS)
5. **Modelling** (Machine learning)
6. **Evaluation** (Fine Tuning/Tinkering/Elbow/Silhouette/Gap Statistic Methods)
7. **Data visualization/Presentation** (To uncover insights/study marketing funnel metrics in detail)
8. **Deployment** (Creating dendrograms/Customer Analytics record to be used by the marketing team for effective Email marketing to facilitate capture and retention.)


### KPI selection process:
1. **Some Marketing goal** 
2. **Quantifiable measurements/ data points used to gauge Capernaum's performance relative to that goal**


**eg:**  
**Goal 1:** Increase Site traffic  
**Indicator**---> Loyalty  
**Quantifiable measurement/data point:** Count of returns/ Avg time taken to return to player pages

**Goal 2:** Increase Time on site  
**Indicator**---> Engagement  
**Quantifiable measurement/data point:** Completion rate/Dwell Time

**Goal 3:** Increase Propensity to spend/Revenue per visitor(RPV)/Average order size  
**Indicator**---> Value  
**Quantifiable measurement/data point:** Cart abandonments/Purchases

**Goal 4:** Diversify Product affinity as much as possible  
**Indicator**---> Interest  
**Quantifiable measurement/data point:** Course enrols (product preference)


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
Specific filtering 
	
<!-- 	
**Algorithm Overview**
![Image of flowchart](https://github.com/Tanay7/Machine-Learning/blob/master/Customer_segmentation/Images/ggg.png)
 --> 
	
**Preliminary code link:**

[K means](https://github.com/Tanay7/Machine-Learning/blob/master/Customer_segmentation/K%20Means%20code.ipynb)



4. **Visualization:**
Python Clustering Screenshot:

![Image of flowchart](https://github.com/Tanay7/Machine-Learning/blob/master/Customer_segmentation/Images/5a.PNG)

Cluster/Class Description:
![Image of flowchart](https://github.com/Tanay7/Machine-Learning/blob/master/Customer_segmentation/Images/7a.PNG)
<!-- 
![Image of flowchart](https://github.com/Tanay7/Machine-Learning/blob/master/Customer_segmentation/Images/8a.PNG)
 --> 
![Image of flowchart](https://github.com/Tanay7/Machine-Learning/blob/master/Customer_segmentation/Images/9a.PNG)

<!-- 
**Clustered User Ids:**

![Image of flowchart](https://github.com/Tanay7/Machine-Learning/blob/master/Customer_segmentation/Images/2a.png)

**Dendrogram sample:**
![Image of flowchart](https://github.com/Tanay7/Machine-Learning/blob/master/Customer_segmentation/Images/6a.jpg)
 --> 
## Insights: 

1. **Loyalty** (refer Loyalty graphs.pdf)
![Image of flowchart](https://github.com/Tanay7/Machine-Learning/blob/master/Customer_segmentation/Images/3a.png)

![Image of flowchart](https://github.com/Tanay7/Machine-Learning/blob/master/Customer_segmentation/Images/4a.png)


2. **Engagement** (refer Engagement graphs.pdf)

As you can see from the pdf, the users with no engagement on the site increase from 27% to 95% in the next month (April) due to email marketing efforts by the team.However, the completion rate seems to be falling drastically no matter what.
  
3. **Interest** (refer Interest graphs.pdf)

As you can see from the pdf, the users with no Interest (in various courses) on the site increase from 5% to 95% in the next month (April).The users with Narrow focus (No. of different courses enrolled) decreases from 55% to 5%

4. **Value** (refer Value graphs.pdf)

As you can see from the pdf, the users with no Value (no propensity to spend) on the site remains the same which is justifiable as Capernaum is dedicated to providing free education and hence targets the users with negligible tendency to spend/purchase.
  
  
## Outcome: 
You can see that due to the inclusion of new courses and the arduous efforts by our Marketing team the percentage of High purchasers do spike up.


## Challenges encountered during my journey
	
### Problem 1:
**Difficulties in scoping the project during the initial stages.**\  
eg Changes in business expectations, fickleness in the selection of KPI's 

**Mitigation measure:**\  
Tried to adapt as the problem evolved.\
Took several iterations of modelling, trial and error to finalise the methodology.\
Nevertheless it was exciting to be on my toes.



### Problem 2: 
**Simply throwing a standard Machine learning algo at a particular business problem is not enough.  Each business scenario is different.**\
eg: In our business, there is a huge difference between users who made '0' product purchase than the users who made just '1' purchase which are in turn way different than the ones who made '>2-3' purchases. A standard machine learning algo (Clustering) cannot take this into account.

**Mitigation measure:**\
Tinkering and filtering was carried out with the data at varying levels (Led to changes in scoping of the project at regular intervals)

### Problem 3:
**Market dynamics: Instability of segments over time**
eg: Structural change which leads to segment creep and membership migration as the Users move from one segment to another.

**Mitigation measure:**\
Carried out automation (results fetch from SQL Db,Google sheets) to drastically reduce manual labour and time when Daily/ weekly/monthly/6 monthly segmentations were carried out.

### Problem 4:
**Frequent Google sheet crash:**\
eg Cannot paste 10 million cells in one sheet.

**Mitigation measure:** Divided and analysed 1.3 million users at a time (Total: 10 segments )
         
### Problem 5:
**Extensive manual work involved:**\
eg: Copy/pasting millions of rows/slow loading times while performing daily/weekly/monthly segmentation.

**Mitigation measure:**\
Extensive automation in Google cloud during the intermediate stages of the project.



### Problem 6:
**Limitations of Alison MYSQL database version (Ver 5.0) on AWS**\
eg. Some analytical functions (LEAD/LAG) could not be used out of the box.

**Mitigation measure:**\
Wrote custom analytical functions from scratch.



### Problem 7:
**Frequent MYSQL database memory table crash/lag which caused slow query execution to fetch results:**\
eg 2-3 hour wait period 

**Mitigation measure:**\
Broke down the complex query into several simple ones and later collated into a single sheet for analysis using iterator.


## Impact:  

1. Implemented entirely using Open source tools hence negligible implementation cost.  
2. Offered Capernaum Ltd (Alison) the ability to leverage data to discover consumers’ intent.  
3. Personalized/timely marketing campaigns for optimal customer experience now possible.     





