#!/usr/bin/env python
# coding: utf-8

# # Question 1: Performance of schools with art programs versus no art programs
# 
# Question 1: How does art program staffing affect state test results (ELA & Math) in New York City schools?
# 
# Hypothesis: IF art programs have an affect on state test results THEN there will be a statistically significant difference in ELA and Math state test scores between schools with a full time arts supervisor and schools without a full time arts supervisor.
#     
# Null Hypothesis: IF art programs do not have an affect on state test results THEN there will be no statistically significant difference in ELA and Math state test scores between schools with a full time arts supervisor and schools without a full time arts supervisor.
# 

# GET AND MERGE DATA

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


#import dependencies
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats


# In[6]:


#load data 
#resource for converting:https://medium.com/better-programming/using-python-to-convert-worksheets-in-an-excel-file-to-separate-csv-files-7dd406b652d7

#survey_2014_2015 = "2014-2015_Arts_Survey_Data.csv"
#survey_2015_2016 = "2015-2016_Arts_Survey_Data.csv"
survey_2016_2017 = "../data/2016-2017_Arts_Survey_Data.csv"
survey_2017_2018 = "../data/2017-2018_Arts_Survey_Data.csv"
survey_2018_2019 = "../data/2018-2019_Arts_Survey_Data.csv"
ELA_2013_2019 = "../data/2013-2019_school_ela_results.csv"
Math_2013_2019 = "../data/2013-2019_school_ela_results.csv"

#survey_2014_2015_df = pd.read_csv(survey_2014_2015, encoding='utf-8', dtype = 'unicode')
#survey_2015_2016_df = pd.read_csv(survey_2015_2016, encoding='utf-8', dtype = 'unicode')
survey_2016_2017_df = pd.read_csv(survey_2016_2017, encoding='utf-8', dtype = 'unicode')
survey_2017_2018_df = pd.read_csv(survey_2017_2018, encoding='utf-8', dtype = 'unicode')
survey_2018_2019_df = pd.read_csv(survey_2018_2019, encoding='utf-8', dtype = 'unicode')
ELA_2013_2019_df = pd.read_csv(ELA_2013_2019, encoding='utf-8', dtype = 'unicode')
Math_2013_2019_df = pd.read_csv(Math_2013_2019, encoding='utf-8', dtype = 'unicode')


# In[7]:


#Merge data by DBN and SchoolYear

#Add SchoolYear columns and SchoolYear to each survey file
survey_2016_2017_df["Year"] = '2017'
survey_2017_2018_df["Year"] = '2018'
survey_2018_2019_df["Year"] = '2019'

#Add subject to state test score files
ELA_2013_2019_df["Subject"] = 'ELA'
Math_2013_2019_df["Subject"] = 'Math'

#Change Q0_DBN column to just DBN in survey data
survey_2016_2017_dbn = survey_2016_2017_df.rename(columns = {'Q0_DBN':'DBN'})
survey_2016_2017_dbn
survey_2017_2018_dbn = survey_2017_2018_df.rename(columns = {'Q0_DBN':'DBN'})
survey_2017_2018_dbn
survey_2018_2019_dbn = survey_2018_2019_df.rename(columns = {'Q0_DBN':'DBN'})
survey_2018_2019_dbn

#Append survey results
survey_2016_2018 = survey_2016_2017_dbn.append(survey_2017_2018_dbn)
survey_2016_2019 = survey_2016_2018.append(survey_2018_2019_dbn)
survey_2016_2019
#append state test files
TestResults_2013_2019 = ELA_2013_2019_df.append(Math_2013_2019_df)
TestResults_2013_2019

#Merge survey and state test results files on Year and DBN
combined_df = pd.merge(TestResults_2013_2019, survey_2016_2019, how='inner', on=['Year', 'DBN'])
#combined_df.head(1)


# EXPLORE AND CLEAN DATA

# In[8]:


#explore combined_df should have 27175 rows and 1840 columns
combined_df.shape


# In[9]:


#explore column headers
#list(combined_df.columns)


# In[10]:


#only choose columns necessary for analysis df1 = df[['a','b']]
cleaned_df = combined_df[["DBN", "School Name", "Grade", "Year", 
                          "Subject", "# Level 3+4", "% Level 3+4", 
                          "Q3_1", "Q3_2", "Q3_3", "Q3_4"]]
cleaned_df.head(1)


# In[11]:


#rename columns
   
col_rename_dict = {
    "Q3_1": "Full Time",
    "Q3_2": "Full Time Plus",
    "Q3_3": "Part Time",
    "Q3_4": "None",

}
renamed_df = cleaned_df.rename(columns=col_rename_dict)
renamed_df.head(1)


# In[12]:


#68 rows have an 's' in the % of Level 3+4 and need to be removed
sorted_df = renamed_df.sort_values("% Level 3+4", ascending = True)
sorted_df["% Level 3+4"]
sorted_df.groupby(["% Level 3+4"]).count()


# Get names of indexes for which % Level 3+4 has value 's'
indexNames = renamed_df[renamed_df['% Level 3+4'] == 's' ].index
 
# Delete 's' row indexes from dataFrame
renamed_df.drop(indexNames , inplace=True)
renamed_df.head(1)


# In[13]:


#Only include ALL Grades rows to summarize data by the school level
cond = renamed_df["Grade"] == "All Grades"
renamed_df =renamed_df[cond]
renamed_df.head(5)


# In[14]:


#explore data- get # of rows
total = len(renamed_df)
total


# In[15]:


#Unique schools
renamed_df["DBN"].nunique()


# In[16]:


#change type to perform analyses
renamed_df['Full Time'] = renamed_df['Full Time'].astype(float)
renamed_df['Full Time Plus'] = renamed_df['Full Time Plus'].astype(float)
renamed_df['Part Time'] = renamed_df['Part Time'].astype(float)
renamed_df['None'] = renamed_df['None'].astype(float)
renamed_df['% Level 3+4'] = renamed_df['% Level 3+4'].astype(float)


# In[17]:


#Write file to csv
#renamed_df.to_csv('renamed1_df.csv', index=False)


# Analyze Data- Full Time vs. Others
# 1. Math and ELA together
# 2. Math only
# 3. ELA only

# In[18]:


#1. Create a summary table with frequencies of responses- ELA and Math together

fulltime = renamed_df["Full Time"].sum()
fulltime
fulltimeplus = renamed_df["Full Time Plus"].sum()
fulltimeplus
parttime = renamed_df["Part Time"].sum()
parttime
none = renamed_df["None"].sum()
none
count_total = fulltime + fulltimeplus + parttime + none
alltheothers = fulltimeplus + parttime + none

#8 rows of data did not answer question.
unaccounted = total - count_total

summary_dict = {'Full Time Only': [fulltime],
        'Full Time Plus Count': [fulltimeplus],
          'Part Time Count': [parttime],
          'None Count': [none],
                 'Not Full Time Only': [alltheothers],
               'Total of these measures': [count_total],
                 'Total': [total],
               'Did Not Answer this Question': [unaccounted]}
summary = pd.DataFrame.from_dict(summary_dict) #, orient='index')
summary


# In[19]:


#2. Create a summary table with frequencies of responses- Math only
m_fulltime = math_renamed_df["Full Time"].sum()
m_fulltime
m_fulltimeplus = math_renamed_df["Full Time Plus"].sum()
m_fulltimeplus
m_parttime = math_renamed_df["Part Time"].sum()
m_parttime
m_none = math_renamed_df["None"].sum()
m_none
m_count_total = m_fulltime + m_fulltimeplus + m_parttime + m_none
m_alltheothers = m_fulltimeplus + m_parttime + m_none

#What to do with 8 rows of unanswered data?
m_unaccounted = total - m_count_total

summary_dict = {'Full Time Only': [m_fulltime],
        'Full Time Plus Count': [m_fulltimeplus],
          'Part Time Count': [m_parttime],
          'None Count': [m_none],
                 'Not Full Time Only': [m_alltheothers],
               'Total of these measures': [m_count_total],
                 'Total': [total],
               'Did Not Answer this Question': [m_unaccounted]}
m_summary = pd.DataFrame.from_dict(summary_dict) #, orient='index')
m_summary


# In[20]:


#3. Create a summary table with frequencies of responses- ELA only
ela_fulltime = ela_renamed_df["Full Time"].sum()
ela_fulltime
ela_fulltimeplus = ela_renamed_df["Full Time Plus"].sum()
ela_fulltimeplus
ela_parttime = ela_renamed_df["Part Time"].sum()
ela_parttime
ela_none = ela_renamed_df["None"].sum()
ela_none
ela_count_total = ela_fulltime + ela_fulltimeplus + ela_parttime + ela_none
ela_alltheothers = ela_fulltimeplus + ela_parttime + ela_none

#What to do with 8 rows of unanswered data?
ela_unaccounted = total - ela_count_total

summary_dict = {'Full Time Only': [ela_fulltime],
        'Full Time Plus': [ela_fulltimeplus],
          'Part Time': [ela_parttime],
          'None': [ela_none],
              #   'Not Full Time Only': [ela_alltheothers],
               'Total': [ela_count_total]}
ela_summary = pd.DataFrame.from_dict(summary_dict) #, orient='index')
ela_summary


# In[21]:


#1. Math and ELA together frequencies
renamed_df.groupby(["Full Time"])["% Level 3+4"].count()


# In[22]:


#2. Math only frequencies
cond1 = renamed_df['Subject'] == 'Math'
math_renamed_df = renamed_df[cond1]
math_renamed_df.groupby(["Full Time"])["% Level 3+4"].count()


# In[23]:


#3. ELA only frequencies
cond2 = renamed_df['Subject'] == 'ELA'
ela_renamed_df = renamed_df[cond2]
ela_renamed_df.groupby(["Full Time"])["% Level 3+4"].count()


# In[26]:


#1 Math and ELA together histograms
#plt.hist(renamed_df["% Level 3+4"], bins=None, histtype='bar', align='mid', orientation='vertical')
#plt.savefig("../images/MathandELAHistogram.png")


# In[139]:


#2. Math histogram
plt.hist(math_renamed_df["% Level 3+4"], bins=None, histtype='bar', align='mid', orientation='vertical')
plt.title("Math Pass Rate Histogram")
plt.xlabel("Pass Rate")
plt.savefig("../images/MathStateTestScores_Histogram.png")


# In[27]:


#3. ELA histogram
plt.hist(ela_renamed_df["% Level 3+4"], bins=None, histtype='bar', align='mid', orientation='vertical')
plt.title("ELA Pass Rate Histogram")
plt.xlabel("Pass Rate")
plt.savefig("../images/ELAStateTestScores_Histogram.png")


# In[28]:


#1. Math and ELA descriptives pd.concat([s1, s2], ignore_index=True), df.columns = ['a', 'b', 'c']
MandELA_mean = renamed_df.groupby(["Full Time"])["% Level 3+4"].mean()
MandELA_median = renamed_df.groupby(["Full Time"])["% Level 3+4"].median()
MandELA_var = renamed_df.groupby(["Full Time"])["% Level 3+4"].var()
MandELA_std = renamed_df.groupby(["Full Time"])["% Level 3+4"].std()
MandELA_max = renamed_df.groupby(["Full Time"])["% Level 3+4"].max()
MandELA_min = renamed_df.groupby(["Full Time"])["% Level 3+4"].min()

summary = pd.concat([MandELA_mean, MandELA_median, MandELA_var, MandELA_std, 
                     MandELA_max, MandELA_min], axis=1, join='inner')

summary.columns = ['Mean', 'Median', 'Variance', "Standard Deviation", "Maximum", "Minimum"]
summary


# In[29]:


#2. Math descriptives pd.concat([s1, s2], ignore_index=True), df.columns = ['a', 'b', 'c']
M_mean = math_renamed_df.groupby(["Full Time"])["% Level 3+4"].mean()
M_median = math_renamed_df.groupby(["Full Time"])["% Level 3+4"].median()
M_var = math_renamed_df.groupby(["Full Time"])["% Level 3+4"].var()
M_std = math_renamed_df.groupby(["Full Time"])["% Level 3+4"].std()
M_max = math_renamed_df.groupby(["Full Time"])["% Level 3+4"].max()
M_min = math_renamed_df.groupby(["Full Time"])["% Level 3+4"].min()

m_summary = pd.concat([M_mean, M_median, M_var, M_std, 
                     M_max, M_min], axis=1, join='inner')

m_summary.columns = ['Mean', 'Median', 'Variance', "Standard Deviation", "Maximum", "Minimum"]
m_summary


# In[30]:


#3. ELA descriptives pd.concat([s1, s2], ignore_index=True), df.columns = ['a', 'b', 'c']
ela_mean = ela_renamed_df.groupby(["Full Time"])["% Level 3+4"].mean()
ela_median = ela_renamed_df.groupby(["Full Time"])["% Level 3+4"].median()
ela_var = ela_renamed_df.groupby(["Full Time"])["% Level 3+4"].var()
ela_std = ela_renamed_df.groupby(["Full Time"])["% Level 3+4"].std()
ela_max = ela_renamed_df.groupby(["Full Time"])["% Level 3+4"].max()
ela_min = ela_renamed_df.groupby(["Full Time"])["% Level 3+4"].min()

ela_summary = pd.concat([ela_mean, ela_median, ela_var, ela_std, 
                     ela_max, ela_min], axis=1, join='inner')

ela_summary.columns = ['Mean', 'Median', 'Variance', "Standard Deviation", "Maximum", "Minimum"]
ela_summary


# In[31]:


#1. Math and ELA together Welch's independent ttest

#perform independent t-test to see if no art programs is different from 1 or more art programs group both Math and ELA
#(Welch's t-test) equal_var = False
fulltime_test = renamed_df[renamed_df['Full Time'] == 1]
none_test = renamed_df[renamed_df['Full Time'] == 0]

stats.ttest_ind(fulltime_test['% Level 3+4'], none_test['% Level 3+4'], equal_var=False)


# In[32]:


#2. Math Welch's independent ttest

#perform independent t-test to see if no art programs is different from 1 or more art programs group both Math and ELA
#(Welch's t-test) equal_var = False
fulltime_test = math_renamed_df[math_renamed_df['Full Time'] == 1]
none_test = math_renamed_df[math_renamed_df['Full Time'] == 0]

stats.ttest_ind(fulltime_test['% Level 3+4'], none_test['% Level 3+4'], equal_var=False)


# In[58]:


#2. Math boxplot
boxplot = math_renamed_df.boxplot("% Level 3+4", by="Full Time", figsize=(9, 6), meanline=False, showmeans=True)
boxplot.set_title("Math Pass Rate by Full Time Supervisor")
print(boxplot)
plt.savefig("../images/Math_boxplot_fulltimesupervisor.png")


# In[33]:


#3. ELA Welch's independent ttest

#perform independent t-test to see if no art programs is different from 1 or more art programs group both Math and ELA
#(Welch's t-test) equal_var = False
fulltime_test = ela_renamed_df[ela_renamed_df['Full Time'] == 1]
none_test = ela_renamed_df[ela_renamed_df['Full Time'] == 0]

stats.ttest_ind(fulltime_test['% Level 3+4'], none_test['% Level 3+4'], equal_var=False)


# In[59]:


#3. ELA boxplot
boxplot = ela_renamed_df.boxplot("% Level 3+4", by="Full Time", figsize=(9, 6), meanline=False, showmeans=True)
boxplot.set_title("ELA Pass Rate by Full Time Supervisor")
print(boxplot)
plt.savefig("../images/ELA_boxplot_fulltimesupervisor.png")


# There isn't a statistically significant difference between schools with full time art supervisor and those without for ELA and ELA & Math combined or ELA.

# Use ANOVA to check for differences by year

# In[34]:


#create conditions for each year
cond1 = renamed_df['Year'] == '2017'
cond2 = renamed_df['Year'] == '2018'
cond3 = renamed_df['Year'] == '2019'

#create data frame for each year
df_2017 = renamed_df[cond1]
df_2018 = renamed_df[cond2]
df_2019 = renamed_df[cond3]

group1 = df_2017["% Level 3+4"]
group2 = df_2018["% Level 3+4"]
group3 = df_2019["% Level 3+4"]


# In[35]:


#boxplot
renamed_df.boxplot("% Level 3+4", by="Year", figsize=(9, 6), meanline=False, showmeans=True)


# In[36]:


#ANOVA to see if school years are significantly different
stats.f_oneway(group1, group2, group3)
#stats.f_oneway(group2, group3)
#stats.f_oneway(group1, group2)


# Look for schools who went from having no full time arts supervisor to having full time supervisor
# 
# 36 schools went from having no full time art supervisor to having full time supervisor.

# In[37]:


#Merge to identify schools that changed from no full time to full time
cond1 = df_2017['Full Time'] == 0
cond2 = df_2018['Full Time'] == 1
cols = ["DBN", "School Name", 'Year', "% Level 3+4", "Full Time", "Subject"]

df_2017_noft = df_2017[cond1][cols]
df_2018_ft = df_2018[cond2][cols]


#merge on DBN DataFrame.merge(self, right, how='inner', on=None,
change_table = df_2017_noft.merge(df_2018_ft, how = 'inner', on='DBN')
change_table


# In[38]:


#Write file to csv to check that table merged correctly
#change_table.to_csv('tableofchange.csv', index=False)


# In[39]:


#How many schools is this? 2 years * 2 subjects = 4 rows per school
change_table["DBN"].value_counts()


# In[40]:


#separate and append to be able to run tests and create box plot
cols1 = ["DBN", "School Name_x", "Year_x", "% Level 3+4_x", "Full Time_x", "Subject_x"]
cols2 = ["DBN", "School Name_y", "Year_y", "% Level 3+4_y", "Full Time_y", "Subject_y"]

firstyear = change_table[cols1]
secondyear = change_table[cols2]

#rename columns df.rename(columns={"A": "a", "B": "c"})
first_clean = firstyear.rename(columns={"School Name_x": "School_Name", 
                          "Year_x": "Year", 
                          "% Level 3+4_x": "% Level 3+4",
                        "Subject_x": "Subject",
                          "Full Time_x": "Full Time"})
second_clean = secondyear.rename(columns={"School Name_y": "School_Name", 
                          "Year_y": "Year", 
                          "% Level 3+4_y": "% Level 3+4",
                        "Subject_y": "Subject",
                          "Full Time_y": "Full Time"})

appended_df = first_clean.append(second_clean)
appended_df


# In[41]:


#there is an increase is average for schools who hired a full time arts supervisor
appended_df.groupby(["Year", "Subject"])["% Level 3+4"].mean()


# In[42]:


appended_df.boxplot("% Level 3+4", by="Year", figsize=(9, 6), meanline=False, showmeans=True)


# In[43]:


grouped_append = appended_df.groupby(["DBN", "Year"], as_index=False)["% Level 3+4"].mean()
grouped_append.head(5)


# In[44]:


nofulltime_test = appended_df[appended_df['Year'] == '2017']
fulltime_test = appended_df[appended_df['Year'] == '2018']

stats.ttest_ind(nofulltime_test['% Level 3+4'], fulltime_test['% Level 3+4'], equal_var=False)


# Analyze Data- Full Time + Full Time Plus vs. Part Time/ None
# 1. Math and ELA together
# 2. Math only
# 3. ELA only

# In[45]:


#Create a column that combines the Full Time and Full Time Plus columns
renamed_df['Any Full Time'] = renamed_df["Full Time"] + renamed_df["Full Time Plus"]
anyfulltime_df = renamed_df
anyfulltime_df.head()


# In[46]:


#1. Math and ELA together frequencies
anyfulltime_df.groupby(["Any Full Time"])["% Level 3+4"].count()


# In[47]:


#2. Math only frequencies
cond1 = anyfulltime_df['Subject'] == 'Math'
math_anyfulltime_renamed_df = anyfulltime_df[cond1]
math_anyfulltime_renamed_df.groupby(["Any Full Time"])["% Level 3+4"].count()


# In[48]:


#3. ELA only frequencies
cond1 = anyfulltime_df['Subject'] == 'ELA'
ela_anyfulltime_renamed_df = anyfulltime_df[cond1]
ela_anyfulltime_renamed_df.groupby(["Any Full Time"])["% Level 3+4"].count()


# In[49]:


#1. Math and ELA descriptives pd.concat([s1, s2], ignore_index=True), df.columns = ['a', 'b', 'c']
MandELA_mean = anyfulltime_df.groupby(["Any Full Time"])["% Level 3+4"].mean()
MandELA_median = anyfulltime_df.groupby(["Any Full Time"])["% Level 3+4"].median()
MandELA_var = anyfulltime_df.groupby(["Any Full Time"])["% Level 3+4"].var()
MandELA_std = anyfulltime_df.groupby(["Any Full Time"])["% Level 3+4"].std()
MandELA_max = anyfulltime_df.groupby(["Any Full Time"])["% Level 3+4"].max()
MandELA_min = anyfulltime_df.groupby(["Any Full Time"])["% Level 3+4"].min()

summary = pd.concat([MandELA_mean, MandELA_median, MandELA_var, MandELA_std, 
                     MandELA_max, MandELA_min], axis=1, join='inner')

summary.columns = ['Mean', 'Median', 'Variance', "Standard Deviation", "Maximum", "Minimum"]
summary


# In[50]:


#2. Math descriptives pd.concat([s1, s2], ignore_index=True), df.columns = ['a', 'b', 'c']
M_mean = math_anyfulltime_renamed_df.groupby(["Any Full Time"])["% Level 3+4"].mean()
M_median = math_anyfulltime_renamed_df.groupby(["Any Full Time"])["% Level 3+4"].median()
M_var = math_anyfulltime_renamed_df.groupby(["Any Full Time"])["% Level 3+4"].var()
M_std = math_anyfulltime_renamed_df.groupby(["Any Full Time"])["% Level 3+4"].std()
M_max = math_anyfulltime_renamed_df.groupby(["Any Full Time"])["% Level 3+4"].max()
M_min = math_anyfulltime_renamed_df.groupby(["Any Full Time"])["% Level 3+4"].min()

m_summary = pd.concat([M_mean, M_median, M_var, M_std, 
                     M_max, M_min], axis=1, join='inner')

m_summary.columns = ['Mean', 'Median', 'Variance', "Standard Deviation", "Maximum", "Minimum"]
m_summary


# In[51]:


#3. ELA descriptives pd.concat([s1, s2], ignore_index=True), df.columns = ['a', 'b', 'c']
ela_mean = ela_anyfulltime_renamed_df.groupby(["Any Full Time"])["% Level 3+4"].mean()
ela_median = ela_anyfulltime_renamed_df.groupby(["Any Full Time"])["% Level 3+4"].median()
ela_var = ela_anyfulltime_renamed_df.groupby(["Any Full Time"])["% Level 3+4"].var()
ela_std = ela_anyfulltime_renamed_df.groupby(["Any Full Time"])["% Level 3+4"].std()
ela_max = ela_anyfulltime_renamed_df.groupby(["Any Full Time"])["% Level 3+4"].max()
ela_min = ela_anyfulltime_renamed_df.groupby(["Any Full Time"])["% Level 3+4"].min()

ela_summary = pd.concat([ela_mean, ela_median, ela_var, ela_std, 
                     ela_max, ela_min], axis=1, join='inner')

ela_summary.columns = ['Mean', 'Median', 'Variance', "Standard Deviation", "Maximum", "Minimum"]
ela_summary


# In[52]:


#1. Math and ELA together Welch's independent ttest

#perform independent t-test to see if any full time supervisor is different from 1 not in both Math and ELA
#(Welch's t-test) equal_var = False
fulltime_test = anyfulltime_df[anyfulltime_df['Any Full Time'] == 1]
none_test = anyfulltime_df[anyfulltime_df['Any Full Time'] == 0]

stats.ttest_ind(fulltime_test['% Level 3+4'], none_test['% Level 3+4'], equal_var=False)


# In[53]:


#2. Math Welch's independent ttest

#perform independent t-test to see if no art programs is different from 1 or more art programs group both Math and ELA
#(Welch's t-test) equal_var = False
fulltime_test = math_anyfulltime_renamed_df[math_anyfulltime_renamed_df['Full Time'] == 1]
none_test = math_anyfulltime_renamed_df[math_anyfulltime_renamed_df['Full Time'] == 0]

stats.ttest_ind(fulltime_test['% Level 3+4'], none_test['% Level 3+4'], equal_var=False)


# In[54]:


#2. Math boxplot
boxplot = math_anyfulltime_renamed_df.boxplot("% Level 3+4", by="Full Time", figsize=(9, 6), meanline=False, showmeans=True)
boxplot.set_title("Math Pass Rate by Full Time Supervisor")
print(boxplot)


# In[55]:


#3. ELA Welch's independent ttest

#perform independent t-test to see if no art programs is different from 1 or more art programs group both Math and ELA
#(Welch's t-test) equal_var = False
fulltime_test = ela_anyfulltime_renamed_df[ela_anyfulltime_renamed_df['Full Time'] == 1]
none_test = ela_anyfulltime_renamed_df[ela_anyfulltime_renamed_df['Full Time'] == 0]

stats.ttest_ind(fulltime_test['% Level 3+4'], none_test['% Level 3+4'], equal_var=False)


# In[56]:


#3. ELA boxplot
boxplot = ela_anyfulltime_renamed_df.boxplot("% Level 3+4", by="Full Time", figsize=(9, 6), meanline=False, showmeans=True)
boxplot.set_title("ELA Pass Rate by Full Time Supervisor")
print(boxplot)


# In[2]:


jupyter nbconvert --to script [aforshee_question1].ipynb

