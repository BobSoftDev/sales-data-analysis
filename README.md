# Black Friday Sales Analysis

This project is developed with Li Wang. 
We got Black Friday sales data from Kaggle and it made
537577 observations for customer behaviours on Black Friday in a retail store. We
developed different methods to predict customer age and the kinds of products that
customers will purchase. The performance of the selected models are relatively
good.

Here is the [Code for prediction](code) and the [Data Analytics Report](report.pdf).

* Data Visualization

We first tried to visualize data by compressing data using PCA. The PCA variance of the first two components
are 9.99997580e-01 and 1.72858695e-06. It turned out that the first variance influence the data too much. We also tried dimensional ISOMAP and KNN. But none of them is a great indicator of the data.
* Data prediction

According to related works and the characteristics of the data, we chose to predict customer age and the kinds of product that customers will purchase.
We tried several different models including decision tree, random forest, logistic regression, polynomial kernel, Gaussian kernel and softmax. 
According to the results, we chose decision tree for both predictions. 
