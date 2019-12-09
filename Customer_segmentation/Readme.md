## Customer behavioural segmentation at Capernaum Ltd using Machine learning: End to End Pipeline
(Intricacies not included as it is confidential)


### Objective:
 To understand customer behaviour of Capernaum(Alison) user base (13 million approx.) and categorize them based on demography and buying behaviour to uncover insights and facilitate capture/retention.

### Business questions 

What are the signals of a 
1. **Useful transactional user?**
2. **Unusable transactional user?**
3. **Engaged user?**
4. **Multi-course user?**
5. **Graduate user?**

### Stakeholders

1. **CEO**
2. **CTO**
3. **Manager**
4. **Marketing team**

### Stages:

1. **Business Understanding** (Brainstorming with senior management /Deciding effective KPIs)
2. **Data Understanding** (DB tables/ Google Analytics metrics)
3. **Data Extraction** (SQL for transactional data/Google Analytics API for demographic data)
4. **Data Preparation** (Google sheets/Advanced Excel: Pivots/VLOOKUPS)
5. **Modelling** (Machine learning)
6. **Evaluation** (Fine Tuning/Tinkering the algorithm according to specific business (Alison) scenario)
7. **Data visualization/Presentation** (To uncover insights)
8. **Deployment** (Creating dendrograms/Customer Analytics record to be used by the marketing team for effective Email marketing to facilitate capture and retention.)


![Image of flowchart](https://github.com/Tanay7/Machine-Learning/blob/master/Customer_segmentation/Images/ggg.png)


## Key findings : 

1. Loyalty (refer Loyalty graphs.pdf) (The pdfs need to be downloaded to be magnified)

  As you can see from the pdf, the users with no activity on the site increase from 26% to 98% in the next month (April).
Concurrently, the medium-low value users also tend to decrease from 60% to 5% 

2. Engagement (refer Engagement graphs.pdf)

  As you can see from the pdf, the users with no engagement on the site increase from 27% to 95% in the next month (April) due to email marketing efforts by the team.
  However, the completion rate seems to be falling drastically no matter what.
  
3. Interest (refer Interest graphs.pdf)

  As you can see from the pdf, the users with no Interest (in various courses) on the site increase from 5% to 95% in the next month (April).
  The users with Narrow focus (No. of different courses enrolled) decreases from 55% to 5%

4. Value (refer Value graphs.pdf)

  As you can see from the pdf, the users with no Value (no propensity to spend) on the site remains the same which is justifiable as Capernaum is dedicated to providing free education
  and hence targets the users with negligible tendency to spend/purchase.
  However, you can see that due to the inclusion of new courses and the arduous efforts by our Marketing team the percentage of High purchasers do spikes up.
  
Overall, I was responsible for creating different personas via dendrograms for effective email marketing/push notifications at Capernaum.  


## Problems Encountered during my journey

	
### Problem 1: 
Difficulties in scoping the project during the initial stages eg Changes in business expectations 

**Mitigation measure:** Tried to adapt as the problem evolved.
Took several iterations of modelling, trial and error to finalise the methodology.
Nevertheless it was exciting to be on my toes.
	
### Problem 2:
Frequent Google sheet crash: eg Cannot paste 10 million cells in one sheet.

**Mitigation measure:** Divided and analysed 1.3 million users at a time (Total: 10 segments )
         
### Problem 3:
Extensive manual work involved : eg: Copy/pasting millions of rows/slow loading times. 

**Mitigation measure:** Extensive automation in Google cloud during the middle stages of the project.

### Problem 4:
Limitations of Alison MYSQL database version (Ver 5.0) on AWS eg : Some analytical functions (LEAD/LAG) could not be used out of the box.

**Mitigation measure:** Wrote custom analytical functions from scratch.

### Problem 5:
Frequent MYSQL database memory table crash/lag which caused slow query execution to fetch results: eg 2-3 hour wait period 

**Mitigation measure:** Broke down the complex query into several simple ones and later collated into a single sheet for analysis using iterator.
 
