#Dominic Callan
#Student number 17164605
#CA 5 - Programming for Big Data - Python coding
#Import the following packages using the import command:

import matplotlib.pyplot as plt; plt.rcdefaults()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison

#Open and read CSV file:

df = pd.read_csv('CA5 - 17164605\winemag-data_first150k.csv', index_col=0)

#Clean data by changing blanks to NA's then delete any rows with price or points that are blank

df.fillna(0) # How to remove rows with blanks -  https://stackoverflow.com/questions/29314033/python-pandas-dataframe-remove-empty-cells -  - Accessed 4 May 2018

df['price'].replace(0, np.nan, inplace=True) 
df.dropna(subset=['price'], inplace=True)

df['points'].replace(0, np.nan, inplace=True)
df.dropna(subset=['points'], inplace=True)

#Create a variable mycolumns containing a dataframe with all rows and the columns 'country', 'points', and 'variety', 'province' and 'price' column from this dataframe

mycolumns = df[['country', 'points', 'variety', 'price']]

#Check how many of each variety of each wine are in the dataset

df['variety'].value_counts()

#Count how many reviews we have for each country

df['country'].value_counts()

#Visualisation bar chart of the top ten most reviewed countries:

mycolumns['country'].value_counts().head(10).plot.bar()


#Create a variable Chardonnay to filter my columns dataframe so that it now shows us only the Chardonnay variety of wine (excluding blends)

Chardonnay = df[mycolumns['variety'] == 'Chardonnay']

#Create a bar chart of the total number of Chardonnay reviews by country

Chardonnay['country'].value_counts().head(10).plot.bar()

#Find the mean points score all chardonnays (excluding blends) reviewed in the dataset:

ChardAVG = Chardonnay[['points']].mean()

#find mean price

ChardAVGPrice = Chardonnay[['price']].mean()

#Create a dataframes of all Chardonnay variety wines (excluding blends) from the US,France, New Zealand,
#Australia and Italy and get the mean points score and mean price for these wine reviews

USchardonnay = mycolumns[(mycolumns.variety == 'Chardonnay') & (mycolumns.country == 'US')]
USAVG = USchardonnay[['points']].mean()
USAVGPrice = USchardonnay[['price']].mean()

Francechardonnay = mycolumns[(mycolumns.variety == 'Chardonnay') & (mycolumns.country == 'France')]
FranceAVG = Francechardonnay[['points']].mean()
FranceAVGPrice = Francechardonnay[['price']].mean()

NZchardonnay = mycolumns[(mycolumns.variety == 'Chardonnay') & (mycolumns.country == 'New Zealand')]
NZAVG = NZchardonnay[['points']].mean()
NZAVGPrice = NZchardonnay[['price']].mean()

Auschardonnay = mycolumns[(mycolumns.variety == 'Chardonnay') & (mycolumns.country == 'Australia')]
AUSAVG = Auschardonnay[['points']].mean()
AUSAVGPrice = Auschardonnay[['price']].mean()

Italychardonnay = mycolumns[(mycolumns.variety == 'Chardonnay') & (mycolumns.country == 'Italy')]
ItalyAVG = Italychardonnay[['points']].mean()
ItalyAVGPrice = Italychardonnay[['price']].mean()

Chilechardonnay = mycolumns[(mycolumns.variety == 'Chardonnay') & (mycolumns.country == 'Chile')]
ChileAVG = Chilechardonnay[['points']].mean()
ChileAVGPrice = Chilechardonnay[['price']].mean()

ARGchardonnay = mycolumns[(mycolumns.variety == 'Chardonnay') & (mycolumns.country == 'Argentina')]
ARGAVG = ARGchardonnay[['points']].mean()
ARGAVGPrice = ARGchardonnay[['price']].mean()




#Create a Bar chart showing average ratings points scores for each Chardonnay by the top 7 most reviewed countries:

ChardMeans = [float(USAVG), float(NZAVG), float(AUSAVG), float(FranceAVG), float(ItalyAVG), float(ChileAVG), float(ARGAVG),]
y_pos = np.arange(len(ChardMeans))
Chardlabels = ('US Chardonnay', 'New Zealand Chardonnay', 'Australian Chardonnay', 'French Chardonnay', 'Italian Chardonnay', 'Chilean Chardonnay', 'Argentinian Chardonnay')
yp = [85, 86, 87, 88]
low = min(yp)
high = max(yp)
# setting the y axis ranges, gives a bit of wiggle room at each end - Ref https://stackoverflow.com/questions/11216319/automatically-setting-y-axis-limits-for-bar-graph-using-matplotlib/11217803 - Accessed 4 May 2018
plt.ylim([math.ceil(low-0.5*(high-low)), math.ceil(high+0.5*(high-low))]) # setting the y axis ranges
plt.bar(y_pos, ChardMeans)
#plt.ylabel('Score out of 100') # adding label to the y-axis for clarity
plt.xticks(y_pos,Chardlabels, rotation='vertical') #rotating the labels 90 degrees so that they are readable
plt.title('Average Chardonnay review scores by country')
plt.axhline(float(ChardAVG), color = 'red') #line showing the mean of all Chardonnay scores globally 
plt.grid(zorder=0) #adding gridlines to background
plt.show()


#Bar chart to show mean price per bottle of chardonnay by country, red line indicates global mean price as per our data set

ChardPrice = [float(USAVGPrice), float(NZAVGPrice), float(AUSAVGPrice), float(FranceAVGPrice), float(ItalyAVGPrice), float(ChileAVGPrice), float(ARGAVGPrice)]
y_pos = np.arange(len(ChardPrice))
ChardPricelabels = ('US Chardonnay', 'New Zealand Chardonnay', 'Australian Chardonnay', 'French Chardonnay', 'Italian Chardonnay', 'Chilean Chardonnay', 'Argentinian Chardonnay')
y3 = [20, 30, 40, 50, 60]
low = min(y3) 
high = max(y3)
# setting the y axis ranges, gives a bit of wiggle room at each end - Ref https://stackoverflow.com/questions/11216319/automatically-setting-y-axis-limits-for-bar-graph-using-matplotlib/11217803  - Accessed 4 May 2018
plt.ylim([math.ceil(low-0.5*(high-low)), math.ceil(high+0.5*(high-low))]) 
plt.bar(y_pos, ChardPrice)
plt.xticks(y_pos,ChardPricelabels, rotation='vertical') #rotating the labels 90 degrees so that they are readable
plt.title('Average Chardonnay price by country')
plt.axhline(float(ChardAVGPrice), color = 'red') #line showing the mean of all Chardonnay scores globally 
plt.grid(zorder=0) #adding gridlines to background
plt.show()


# Scatter plot comparing the mean score vs mean price of chardonnays by country

x = ChardMeans
y = ChardPrice
plt.scatter(x, y, label = 'Countries', color = 'r')
plt.title('Average Chardonnay price by country')
plt.xlabel("Mean review score")
plt.text(89.315353 ,64.648614 , "France")
plt.text(87.729543, 27.676953, 'US')
plt.text(87.810573, 25.491765, 'New Zealand')
plt.text(86.727952, 20.409295, 'Australia')
plt.text(88.367906, 32.188889, 'Italy')
plt.text(85.246011, 14.592742, 'Chile' )
plt.text(84.177489, 13.812636, 'Argentina')
plt.plot()


#Look for outliers that are effecting our average = check this visually with a scatter plot

Chardonnay.plot(kind="scatter", x="price", y="points", color = 'purple')

#Let's run our charts again, this time removing the outlier wines over 750 dollars

#Create a dataframe removing any bottles of CHardonnay costing over 750 US dollars
Chardonnay2 = Chardonnay[(Chardonnay.price <= 750)]             

Chardonnay2.plot(kind="scatter", x="price", y="points", color = 'orange')

#Find the mean score and price with the $750 + outliers removed
ChardAVG2 = Chardonnay2[['points']].mean()

#Get mean price too
ChardAVGPrice2 = Chardonnay2[['price']].mean()

#Create new variables for the same data, but at a maximum price of $750 per bottle (using Find and replace function, replacing the dataframe 'mycolumns' with 'Chardonnay2')

USchardonnay2 = Chardonnay2[(Chardonnay2.variety == 'Chardonnay') & (Chardonnay2.country == 'US')]
USAVG2 = USchardonnay2[['points']].mean()
USAVGPrice2 = USchardonnay2[['price']].mean()

Francechardonnay2 = Chardonnay2[(Chardonnay2.variety == 'Chardonnay') & (Chardonnay2.country == 'France')]
FranceAVG2 = Francechardonnay2[['points']].mean()
FranceAVGPrice2 = Francechardonnay2[['price']].mean()

NZchardonnay2 = Chardonnay2[(Chardonnay2.variety == 'Chardonnay') & (Chardonnay2.country == 'New Zealand')]
NZAVG2 = NZchardonnay2[['points']].mean()
NZAVGPrice2 = NZchardonnay2[['price']].mean()

Auschardonnay2 = Chardonnay2[(Chardonnay2.variety == 'Chardonnay') & (Chardonnay2.country == 'Australia')]
AUSAVG2 = Auschardonnay2[['points']].mean()
AUSAVGPrice2 = Auschardonnay2[['price']].mean()

Italychardonnay2 = Chardonnay2[(Chardonnay2.variety == 'Chardonnay') & (Chardonnay2.country == 'Italy')]
ItalyAVG2 = Italychardonnay2[['points']].mean()
ItalyAVGPrice2 = Italychardonnay2[['price']].mean()

Chilechardonnay2 = Chardonnay2[(Chardonnay2.variety == 'Chardonnay') & (Chardonnay2.country == 'Chile')]
ChileAVG2 = Chilechardonnay2[['points']].mean()
ChileAVGPrice2 = Chilechardonnay2[['price']].mean()

ARGchardonnay2 = Chardonnay2[(Chardonnay2.variety == 'Chardonnay') & (Chardonnay2.country == 'Argentina')]
ARGAVG2 = ARGchardonnay2[['points']].mean()
ARGAVGPrice2 = ARGchardonnay2[['price']].mean()

#Create new bar chart of mean scores wines at $750 per bottle

ChardMeans2 = [float(USAVG2), float(NZAVG2), float(AUSAVG2), float(FranceAVG2), float(ItalyAVG2), float(ChileAVG2), float(ARGAVG2)]
y_pos = np.arange(len(ChardMeans2))
Chardlabels2 = ('US Chardonnay', 'New Zealand Chardonnay', 'Australian Chardonnay', 'French Chardonnay', 'Italian Chardonnay', 'Chilean Chardonnay', 'Argentinian Chardonnay')
y5 = [85, 86, 87, 88]
low = min(y5)
high = max(y5)
plt.ylim([math.ceil(low-0.5*(high-low)), math.ceil(high+0.5*(high-low))]) # setting the y axis ranges
plt.bar(y_pos, ChardMeans2)
plt.title('Average Chardonnay review scores by country maximum price $750 per bottle')
plt.xticks(y_pos,Chardlabels, rotation='vertical') #rotating the labels 90 degrees so that they are readable
plt.axhline(float(ChardAVG2), color = 'red') #line showing the mean of all Chardonnay scores globally 
plt.grid(zorder=0) #adding gridlines to background
plt.show()



#Create a new bar chart of average price per bottle of Chardonnay by country with outlier prices above 750 removed

ChardPrice2 = [float(USAVGPrice2), float(NZAVGPrice2), float(AUSAVGPrice2), float(FranceAVGPrice2), float(ItalyAVGPrice2), float(ChileAVGPrice2), float(ARGAVGPrice2)]
y_pos = np.arange(len(ChardPrice2))
ChardPricelabels2 = ('US Chardonnay', 'New Zealand Chardonnay', 'Australian Chardonnay', 'French Chardonnay', 'Italian Chardonnay', 'Chilean Chardonnay', 'Argentinian Chardonnay')
y4 = [20, 30, 40, 50, 60]
low = min(y4)
high = max(y4)
plt.ylim([math.ceil(low-0.5*(high-low)), math.ceil(high+0.5*(high-low))]) # setting the y axis ranges
plt.bar(y_pos, ChardPrice2)

plt.xticks(y_pos,ChardPricelabels2, rotation='vertical') #rotating the labels 90 degrees so that they are readable
plt.title('Average Chardonnay price by country maximum price $750 per bottle')
plt.axhline(float(ChardAVGPrice2), color = 'red') #line showing the mean of all Chardonnay scores globally 
plt.grid(zorder=0) #adding gridlines to background
plt.show()


#Note that French price has reduced slightly



#A new bar chart of the mean scores max price 750 per bottle

Chardonnay2['country'].value_counts().head(10).plot.bar()


#Scatter plot comparing the mean score vs mean price of chardonnays by country max price 750 per bottle

x = ChardMeans2
y = ChardPrice2
plt.xlabel('Mean review score')
plt.title('Average Chardonnay price by country')
plt.scatter(x, y)
plt.text(89.315353 ,64.648614 , "France") # set labels next to the plot points
plt.text(87.729543, 27.676953, 'US')
plt.text(87.810573, 25.491765, 'New Zealand')
plt.text(86.727952, 20.409295, 'Australia')
plt.text(88.367906, 32.188889, 'Italy')
plt.text(85.246011, 14.592742, 'Chile' )
plt.text(84.177489, 13.812636, 'Argentina')
plt.plot()


#Create dataframes and visualisations with maximum price chardonnay of $50, same as above

Chardonnay3 = Chardonnay[(Chardonnay.price <= 50.0)]             

#Check visually with scatter plot - show more even spread than before

Chardonnay3.plot(kind="scatter", x="price", y="points", color = 'green')

#Find the mean score and price with the outliers removed

ChardAVG3 = Chardonnay3[['points']].mean()

ChardAVGPrice3 = Chardonnay3[['price']].mean()

#Create new variables for the same data, but at a maximum price of $50 per bottle (using Find and replace function, replacing the dataframe 'Chardonnay2' with 'Chardonnay3')


USchardonnay3 = Chardonnay3[(Chardonnay3.variety == 'Chardonnay') & (Chardonnay3.country == 'US')]
USAVG3 = USchardonnay3[['points']].mean()
USAVGPrice3 = USchardonnay3[['price']].mean()

Francechardonnay3 = Chardonnay3[(Chardonnay3.variety == 'Chardonnay') & (Chardonnay3.country == 'France')]
FranceAVG3 = Francechardonnay3[['points']].mean()
FranceAVGPrice3 = Francechardonnay3[['price']].mean()

NZchardonnay3 = Chardonnay3[(Chardonnay3.variety == 'Chardonnay') & (Chardonnay3.country == 'New Zealand')]
NZAVG3 = NZchardonnay3[['points']].mean()
NZAVGPrice3 = NZchardonnay3[['price']].mean()

Auschardonnay3 = Chardonnay3[(Chardonnay3.variety == 'Chardonnay') & (Chardonnay3.country == 'Australia')]
AUSAVG3 = Auschardonnay3[['points']].mean()
AUSAVGPrice3 = Auschardonnay3[['price']].mean()

Italychardonnay3 = Chardonnay3[(Chardonnay3.variety == 'Chardonnay') & (Chardonnay3.country == 'Italy')]
ItalyAVG3 = Italychardonnay3[['points']].mean()
ItalyAVGPrice3 = Italychardonnay3[['price']].mean()

Chilechardonnay3 = Chardonnay3[(Chardonnay3.variety == 'Chardonnay') & (Chardonnay3.country == 'Chile')]
ChileAVG3 = Chilechardonnay3[['points']].mean()
ChileAVGPrice3 = Chilechardonnay3[['price']].mean()

ARGchardonnay3 = Chardonnay3[(Chardonnay3.variety == 'Chardonnay') & (Chardonnay3.country == 'Argentina')]
ARGAVG3 = ARGchardonnay3[['points']].mean()
ARGAVGPrice3 = ARGchardonnay3[['price']].mean()


ChardMeans3 = [float(USAVG3), float(NZAVG3), float(AUSAVG3), float(FranceAVG3), float(ItalyAVG3), float(ChileAVG3), float(ARGAVG3)]
y_pos = np.arange(len(ChardMeans3))
Chardlabels3 = ('US Chardonnay', 'New Zealand Chardonnay', 'Australian Chardonnay', 'Franch Chardonnay', 'Italian Chardonnay', 'Chilean Chardonnay', 'Argentinian Chardonnay')
y6 = [85, 86, 87, 88]
low = min(y6)
high = max(y6)
plt.ylim([math.ceil(low-0.5*(high-low)), math.ceil(high+0.5*(high-low))]) # setting the y axis ranges
plt.bar(y_pos, ChardMeans3)
plt.title('Average Chardonnay review scores by country max price $50 per bottle')
plt.xticks(y_pos,Chardlabels, rotation='vertical') #rotating the labels 90 degrees so that they are readable
plt.axhline(float(ChardAVG3), color = 'red') #line showing the mean of all Chardonnay scores globally 
plt.grid(zorder=0) #adding gridlines to background
plt.show()


#Create a new bar chart of average price per bottle of Chardonnay by country with outlier prices above 750 removed

ChardPrice3 = [float(USAVGPrice3), float(NZAVGPrice3), float(AUSAVGPrice3), float(FranceAVGPrice3), float(ItalyAVGPrice3), float(ChileAVGPrice3), float(ARGAVGPrice3)]
y_pos = np.arange(len(ChardPrice3))
ChardPricelabels3 = ('US Chardonnay', 'New Zealand Chardonnay', 'Australian Chardonnay', 'French Chardonnay', 'Italian Chardonnay', 'Chilean Chardonnay', 'Argentinian Chardonnay')
y7 = [20, 30, 40, 50, 60]
low = min(y7)
high = max(y7)
plt.ylim([math.ceil(low-0.5*(high-low)), math.ceil(high+0.5*(high-low))]) # setting the y axis ranges
plt.bar(y_pos, ChardPrice3)
plt.xticks(y_pos,ChardPricelabels3, rotation='vertical') #rotating the labels 90 degrees so that they are readable
plt.title('Average Chardonnay price by country max price $50 per bottle')
plt.axhline(float(ChardAVGPrice3), color = 'red') #line showing the mean of all Chardonnay scores globally 
plt.grid(zorder=0) #adding gridlines to background
plt.show()


#Scatter plot showing mean rating for max $50 per bottle sample, as before

x = ChardMeans3
y = ChardPrice3
plt.xlabel('Mean review score')
plt.title('Average Chardonnay price by country maximum price $50 per bottle')
plt.scatter(x, y, color = 'red')
plt.text(87.578402 ,28.088018 , "France")
plt.text(87.522819, 25.394877, 'US')
plt.text(87.712531, 24.034398, 'New Zealand')
plt.text(86.518809, 17.902821, 'Australia')
plt.text(88.123426, 26.634761, 'Italy')
plt.text(85.223275, 14.239513, 'Chile' )
plt.text(84.159389, 13.591703, 'Argentina')
plt.plot()


#Create two datasets with only the top seven countries and only chardonnay variety for the original and max price $50 sets

Chardonnay7 = [USchardonnay, Francechardonnay, NZchardonnay, Auschardonnay, Italychardonnay, Chilechardonnay, ARGchardonnay]

Chardonnay9 = [USchardonnay3, Francechardonnay3, NZchardonnay3, Auschardonnay3, Italychardonnay3, Chilechardonnay3, ARGchardonnay3]

#Concatanate into single data frames
result = pd.concat(Chardonnay7)

result3 = pd.concat(Chardonnay9)

#Do Tukey tests to compare means between all of the pairs between the seven countries analysed, price and ratings points:
#Ref http://www.statsmodels.org/dev/generated/statsmodels.stats.multicomp.pairwise_tukeyhsd.html - Accessed 4 May 2018

print pairwise_tukeyhsd(endog=result['price'], groups= result['country'], alpha = 0.05)

print pairwise_tukeyhsd(endog=result['points'], groups= result['country'], alpha = 0.05)

print pairwise_tukeyhsd(endog=result3['price'], groups= result3['country'], alpha = 0.05)

print pairwise_tukeyhsd(endog=result3['points'], groups= result3['country'], alpha = 0.05)






