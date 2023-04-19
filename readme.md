# Steps required
## Step 1: Data Preparation

Download the iris dataset from the UCI Machine Learning Repository.

Load the dataset into your Python environment using a library like pandas.

Perform any necessary data preprocessing steps, such as handling missing values, encoding categorical variables, and normalizing numerical features.

Split the dataset into training and testing sets using a 70-30 ratio, and repeat this process 10 times to obtain 10 different samples .

## Step 2: SVM Parameter Optimization

For each of the 10 samples, train an SVM model with 1000 iterations using a library like scikit-learn.

Perform parameter optimization using techniques like Grid Search or Randomized Search to find the best hyperparameters for the SVM model. 

Consider optimizing parameters such as the kernel type (linear, polynomial, or radial basis function), regularization parameter (C), and kernel coefficient (gamma).

Record the best hyperparameters for each sample in a table, along with the corresponding accuracy achieved on the testing set.

## Step 3: Convergence Graph

Select the sample that achieved the highest accuracy on the testing set.

Plot a convergence graph for this sample, showing the change in SVM objective value (or loss) over the iterations of the optimization process. This can help visualize the convergence behavior of the SVM model and assess its performance.


