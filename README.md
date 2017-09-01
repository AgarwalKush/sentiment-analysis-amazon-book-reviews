# sentiment-analysis-amazon-book-reviews
R Program implementing sentiment analysis of Amazon book reviews, using Naive Bayes Classifier and SVM

#####################################\
# Group 2 - Sentiment Analysis for Amazon Book Reviews

#####
# Prerequisites

	-> Install R:
		For Linux: https://cran.r-project.org/bin/linux/
		For Mac OS X: https://cran.r-project.org/bin/macosx/
		For Windows: https://cran.r-project.org/bin/windows/

#####




#####
# Required Software:

	-> Install RStudio:
		For Ubuntu 12.04+ (64-bit): https://download1.rstudio.org/rstudio-1.0.143-amd64.deb
		For MacOS X 10.6+: https://download1.rstudio.org/RStudio-1.0.143.dmg
		For Windows Vista/7/8/10: https://download1.rstudio.org/RStudio-1.0.143.exe
#####




#####
# Contents:
	Train2.R // R Script that runs sentiment analysis (recommend using RStudio to run this)
	reviewsDataset.json // compressed dataset, containing under 10,000 reviews 
  originalDataset.json // full dataset containing over  (remote download link: http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_10.json.gz)
	README.md
#####

	
#####
# How To Execute
	
	-> Highlight everything and run. (should output SVM graph and accuracy, then Naive Bayes graph and accuracy, and then Wordcloud.  Final results printed at the end)
	
	**Please note** : if you get an error about setting the correct working directory, here is how you can fix it:
	In RStudio, in the menu bar all the way on top click Session -> Set Working Directory -> Choose Directory.  -> < select directory where this repo is saved locally >  
  This should clear the error.
#####
