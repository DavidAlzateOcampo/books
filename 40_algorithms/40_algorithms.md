# 40 Algorithms Every Data Scientist Should Know - Master Edition

> [!IMPORTANT]
> **Disclaimer:** This study guide is a result of the 40 algorithms presented in the book *40 Algorithms Every Data Scientist Should Know* by Imran Ahmad. It is intended for educational purposes only.
---

## 1. Linear regression

**Category:** Supervised Learning

### 📐 Mathematical Foundation

$$y = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n + \epsilon$$
Linear regression models the relationship between a dependent variable and one or more independent variables by fitting a linear equation. It uses the Ordinary Least Squares (OLS) method to minimize the sum of squared residuals.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| *Simplicity*: Linear regression is relatively simple to understand and implement. It provides a straightforward approach for modeling the linear relationship between input features and the target variable. | *Linearity assumption*: Linear regression assumes a linear relationship between the input features and the target variable. If the relationship is non-linear, linear regression may result in poor performance and inaccurate predictions. |
| *Interpretability*: Linear regression allows easy interpretation of the modelʼs coefficients. The coefficients represent the impact of each input feature on the target variable, providing insights into the relationships and importance of different features. | *Limited complexity*: Linear regression is limited to capturing linear relationships between variables. It may not capture complex interactions or nonlinear patterns in the data. Polynomial regression can partially address this limitation but has its own constraints. |
| *Efficiency*: Linear regression algorithms are computationally efficient, making them suitable for large datasets with a high number of features. They have low training and prediction times compared to more complex models. | *Sensitivity to outliers*: Linear regression is sensitive to outliers, as they can significantly affect the estimated coefficients and the overall fit of the model. Outliers can distort the line of best fit and impact the accuracy of predictions. |
| *Feature importance*: Linear regression provides a measure of feature importance through the magnitude of the coefficients. Larger coefficients indicate a stronger impact of the corresponding feature on the target variable. | *Multicollinearity*: Linear regression assumes that the input features are not highly correlated with each other (multicollinearity). When multicollinearity exists, it can lead to unstable and unreliable coefficient estimates. |
| *Baseline model*: Linear regression is a baseline model for more advanced algorithms. It provides a simple benchmark against which the performance of more complex models can be evaluated. | *Limited applicability*: Linear regression is most suitable for problems where the relationship between the features and target variable is approximately linear. It may not perform well in cases where the relationship is highly nonlinear or when dealing with categorical variables that cannot be easily transformed. |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(n \cdot p^2 + p^3)`

### 🚀 Real-World Applications

- Predicting house prices
- sales forecasting
- trend analysis in finance.

### 💻 Python Source Code

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Prepare features (X) and target (y)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 3.8, 5.1, 6.2, 7.2])

# Initialize and train linear model
model = LinearRegression()
model.fit(X, y)

# Print coefficients
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Predict on new unseen inputs
X_new = np.array([[6]])
y_pred = model.predict(X_new)
print(f"Predicted target for X=6: {y_pred[0]:.2f}")
```

---

## 2. Logistic regression

**Category:** Supervised Learning

### 📐 Mathematical Foundation

$$P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \sum \beta_i x_i)}}$$
Used for binary classification. It models the probability that a given input belongs to a certain class using the sigmoid (logistic) function.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| *Leverages unlabeled data*: One of the major strengths of self-training is its ability to utilize large amounts of unlabeled data, which are typically more abundant and easier to obtain than labeled examples. | *Error reinforcement*: A major drawback of self-training is the risk of error amplification. If the model makes a mistake in the early stages and is confident about it, this error can be reinforced in subsequent iterations as the wrongly labeled instance is added to the training set. |
| *Cost-efficient*: By leveraging unlabeled data, self-training can reduce the need for expensive and time-consuming data annotation efforts. | *Dependence on initial model*: The quality of the initial model (trained only on labeled data) is crucial. If the initial model is of poor quality, the pseudo-labelling process might be ineffective or even detrimental. |
| *Flexibility*: Self-training can be applied with a variety of base classifiers, from traditional methods like decision trees to more complex models like deep neural networks. | *Overconfidence*: Some models might be overly confident about their predictions, even if they are incorrect. This can lead to the selection of many mislabeled instances, deteriorating the modelʼs performance. |
| *Improved performance*: In situations where labeled data is scarce, self-training can lead to significant performance boosts over supervised learning approaches that only utilize the initial labeled set. | *Diversity of unlabeled data*: If the unlabeled data is not diverse enough or does not represent the underlying distribution well, the gains from self-training might be limited. |
| *Domain adaptation*: Self-training can help in scenarios where the model needs to be adapted to a slightly different domain where labeled data might be limited. | *Complexity*: The iterative nature of self-training can increase the computational cost, especially when using complex models or large datasets. |
| *Regularization effect*: Incorporating predictions on unlabeled data can introduce a form of regularization, preventing the model from overfitting to a small set of labeled samples. | *Hard to debug and introspect*: If the performance drops or does not improve as expected, it can be challenging to determine whether the issue is with the pseudo-labelling process, the base model, or other factors. |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(n \cdot p)`

### 🚀 Real-World Applications

- Spam detection
- credit scoring
- medical diagnosis (e.g.
- disease presence).

### 💻 Python Source Code

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# Prepare training data
X = np.array([[1.5], [2.5], [0.5], [3.5], [1.0]])
y = np.array([0, 1, 0, 1, 0])

# Initialize and train logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Predict probabilities and classes
X_new = np.array([[2.0]])
prob = model.predict_proba(X_new)
pred = model.predict(X_new)
print(f"Probability of class 1: {prob[0][1]:.4f}, Predicted class: {pred[0]}")
```

---

## 3. Decision trees

**Category:** Supervised Learning

### 📐 Mathematical Foundation

$$Gini = 1 - \sum p_i^2 \quad \text{or} \quad Entropy = -\sum p_i \log_2 p_i$$
A non-parametric supervised learning method. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| High predictive accuracy: GBM algorithms, such as XGBoost, LightGBM, and CatBoost, are known for their high predictive accuracy. They can capture complex relationships and interactions in the data, making them suitable for a wide range of tasks, including regression, classification, and ranking problems. | Computational complexity and training time: GBM algorithms can be computationally expensive and require significant computational resources, especially for large datasets or complex models. Training time can be longer compared to simpler algorithms, particularly if the hyperparameter search space is large. |
| Handling complex data: GBM algorithms can effectively handle high-dimensional data with mixed feature types (continuous, categorical) and missing values. They can handle non-linear relationships, interactions, and outliers in the data, making them robust in various real-world scenarios. | Hyperparameter tuning: GBM algorithms have several hyperparameters that need to be tuned for optimal performance. Finding the right combination of hyperparameters can be challenging and time-consuming. Grid search or randomized search techniques are often employed to explore the hyperparameter space. |
| Feature importance: GBM algorithms provide insights into feature importance. By examining the contribution of each feature in the ensemble of models, analysts can understand which features have the most significant impact on the predictions. This information can be valuable for feature selection, dimensionality reduction, and interpreting the model. | Overfitting risk: GBM algorithms are prone to overfitting if the model complexity is too high or if the dataset is small. Regularization techniques, cross-validation, and early stopping can be used to mitigate overfitting. Careful monitoring of the modelʼs performance on validation or hold-out sets is necessary to ensure good generalization. |
| Flexibility and customization: GBM algorithms offer flexibility and customization options through various hyperparameters. Analysts can tune the hyperparameters to control model complexity, prevent overfitting, and optimize performance. Additionally, ensemble methods allow the combination of multiple GBM models or other types of models for even better performance. | Interpretability: GBM algorithms are generally considered less interpretable compared to simpler models like linear regression or decision trees. The ensemble nature and complexity of GBM models can make it challenging to interpret the underlying relationships and decision-making process. Feature importance can provide insights but does not provide a complete understanding. |
| Handling imbalanced and/or missing data: GBM algorithms can handle imbalanced and/or missing datasets by assigning appropriate class weights or using techniques like Synthetic Minority Over-sampling Technique (SMOTE). This makes them effective in tasks such as fraud detection, anomaly detection, and rare event prediction. | Data requirements: GBM algorithms typically require enough labeled data to achieve good performance. If labeled data is limited, GBM models may struggle to generalize well. Data collection and annotation efforts may be necessary to train robust GBM models, especially in domains with limited data availability. |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(n \cdot p \cdot \log n)`

### 🚀 Real-World Applications

- Customer segmentation
- risk assessment
- loan approval.

### 💻 Python Source Code

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
import numpy as np

# Generate classification data
X, y = make_classification(n_samples=100, n_features=4, random_state=42)

# Train decision tree classifier
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)

# Predict on a new sample
sample = np.array([[0.5, -0.2, 1.1, -0.5]])
prediction = clf.predict(sample)
print("Decision Tree Prediction:", prediction)
```

---

## 4. Random forests

**Category:** Supervised Learning

### 📐 Mathematical Foundation

$$\hat{f} = \frac{1}{B} \sum_{b=1}^B f_b(x)$$
An ensemble learning method that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Highly interpretable baseline model | Sensitive to hyperparameters |
| Computationally efficient to train | Struggles with non-linear correlations |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(M \cdot n \cdot p \cdot \log n)`

### 🚀 Real-World Applications

- Fraud detection
- stock market prediction
- image classification.

### 💻 Python Source Code

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Create synthetic data
X = np.random.rand(100, 4)
y = np.random.choice([0, 1], size=100)

# Initialize and train random forest classifier
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X, y)

# Predict on new samples
X_new = np.random.rand(5, 4)
y_pred = model.predict(X_new)
print("Random Forest Predictions:", y_pred)
```

---

## 5. Support Vector Machine

**Category:** Supervised Learning

### 📐 Mathematical Foundation

$$\min \frac{1}{2} ||w||^2 \text{ s.t. } y_i(w \cdot x_i + b) \geq 1$$
Finds the hyperplane that maximizes the margin between two classes. Kernels allow SVMs to perform non-linear classification by mapping inputs into high-dimensional feature spaces.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Highly interpretable baseline model | Sensitive to hyperparameters |
| Computationally efficient to train | Struggles with non-linear correlations |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(n^2 \cdot p) \text{ to } O(n^3 \cdot p)`

### 🚀 Real-World Applications

- Handwriting recognition
- protein fold and remote sensing
- text categorization.

### 💻 Python Source Code

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

# Create synthetic data
X = np.random.rand(100, 2)
y = np.random.choice([0, 1], size=100)

# Create a pipeline with feature scaling and SVM classifier with RBF kernel
model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, probability=True))
model.fit(X, y)

# Predict probabilities
X_new = np.random.rand(5, 2)
probs = model.predict_proba(X_new)
print("SVM Class Probabilities:\n", probs)
```

---

## 6. Naive Bayes

**Category:** Supervised Learning

### 📐 Mathematical Foundation

$$P(c|x) = \frac{P(x|c)P(c)}{P(x)}$$
Based on Bayes' Theorem with the "naive" assumption of conditional independence between every pair of features given the value of the class variable.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| *Simplicity and efficiency*: Naive Bayes algorithms are simple and computationally efficient. They are fast to train and make predictions, making them suitable for large datasets and real-time applications. | *Limited expressiveness*: Naive Bayes algorithms have limited expressiveness compared to more complex models. They may not capture subtle patterns and nuances in the data, leading to reduced predictive accuracy. |
| *Scalability*: Naive Bayes algorithms scale well with the number of features and training instances. They require a small amount of memory to store the model parameters, allowing them to handle high-dimensional data efficiently. | *Sensitivity to feature correlations*: Naive Bayes algorithms assume independence between features. When features are highly correlated, this assumption may result in biased predictions. Pre-processing techniques like feature engineering or dimensionality reduction can help alleviate this issue. |
| *Strong performance on small datasets*: Naive Bayes algorithms often perform well even with limited training data. They can make reasonable predictions when the training set is small, which is beneficial in scenarios where collecting a large amount of labeled data is challenging. | *Lack of calibration*: Naive Bayes algorithms can have issues with calibration, meaning the predicted probabilities may not align well with the actual probabilities. The predicted probabilities can be overconfident or underconfident, requiring calibration techniques for better estimation. |
| *Handles irrelevant features*: Naive Bayes algorithms are robust to irrelevant features. They can effectively filter out irrelevant features and focus on the informative ones, leading to good performance even when irrelevant features are present. | *Data sparsity*: Naive Bayes algorithms can struggle with rare or unseen feature-value combinations in the training data. Unseen data can lead to zero probabilities or inaccurate probability estimates, affecting the modelʼs performance. |
| *Handles categorical and numerical features*: Naive Bayes algorithms can handle both categorical and numerical features. They accommodate different types of data without the need for explicit feature scaling or encoding. | *Sensitivity to input quality*: Naive Bayes algorithms can be sensitive to the quality of the input features. Inaccurate or noisy input features can have a significant impact on the modelʼs performance. |
| *Interpretability*: Naive Bayes algorithms provide interpretable results. The classification decisions are based on conditional probabilities, allowing easy interpretation, and understanding of the modelʼs reasoning. | *Computational complexity*: The prediction step of k-NN algorithms can be computationally expensive, especially with large datasets. As the dataset grows, the time complexity increases as the algorithm needs to compute distances for each instance. |
| *Works well with independence assumption*: Naive Bayes algorithms assume feature independence given the class label. While this assumption may not hold, Naive Bayes can still provide good results, especially in cases where the features are conditionally independent. | *Sensitivity to feature scaling*: k-NN algorithms are sensitive to the scale of input features. Features with larger scales can dominate the distance calculation, leading to biased predictions. Feature scaling techniques such as normalization or standardization should be applied to ensure equal importance of all features. |
| *Simplicity*: k-NN algorithms are simple to understand and implement. They have a straightforward intuition, making them accessible to beginners in machine learning. | *Curse of dimensionality*: k-NN algorithms can suffer from the curse of dimensionality. As the number of features increases, the density of the data in the feature space decreases. This can lead to a degradation in performance and increased computational requirements. |
| *No training phase*: k-NN algorithms do not have an explicit training phase. The model stores the entire training dataset, making it easy to update the model with new data. | *Need for optimal k value*: The choice of the k value in k-NN algorithms is critical. A small value of k may result in overfitting, while a large value may lead to underfitting. Selecting an optimal k value requires experimentation or using techniques like cross-validation. |
| *Flexibility in classification and regression*: k-NN algorithms can be used for classification and regression tasks. For classification, the algorithm assigns the majority class among the k nearest neighbors. For regression, it predicts the average or median value of the k nearest neighbors. | *Imbalanced data impact*: k-NN algorithms are affected by imbalanced class distributions. If the classes are imbalanced, the majority class can dominate the predictions, leading to biased results. Techniques such as oversampling, under-sampling, or using different distance metrics can help address this issue. |
| *Non-parametric and lazy learning*: k-NN algorithms do not assume any underlying distribution or make assumptions about the data. They are non-parametric and make predictions based on the local characteristics of the data. This allows them to handle complex relationships between features and labels. | *Lack of model interpretability*: k-NN algorithms lack model interpretability. They do not provide insight into the underlying relationships between features and labels. The prediction is based on local neighbors without explicitly showing the decision boundaries or feature importance. |
| *Handles multiclass classification*: k-NN algorithms can handle multiclass classification problems by considering the class with the highest count among the k nearest neighbors. | *Real world applications* |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(n \cdot p)`

### 🚀 Real-World Applications

- Real-time prediction
- multi-class classification
- text classification/spam filtering.

### 💻 Python Source Code

```python
from sklearn.naive_bayes import GaussianNB
import numpy as np

# Create synthetic data
X = np.random.randn(100, 2)
y = np.random.choice([0, 1], size=100)

# Fit Gaussian Naive Bayes classifier
clf = GaussianNB()
clf.fit(X, y)

# Predict new data
X_new = np.array([[0.1, -0.5]])
pred = clf.predict(X_new)
print("Naive Bayes Class Prediction:", pred)
```

---

## 7. k-Nearest Neighbors

**Category:** Supervised Learning

### 📐 Mathematical Foundation

$$d(x, y) = \sqrt{\sum (x_i - y_i)^2}$$
A simple, instance-based learning algorithm. A new data point is classified by a majority vote of its neighbors, with the point being assigned to the class most common among its k nearest neighbors.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Highly interpretable baseline model | Sensitive to hyperparameters |
| Computationally efficient to train | Struggles with non-linear correlations |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(n \cdot p)`

### 🚀 Real-World Applications

- Recommendation systems
- visual pattern recognition
- gene expression analysis.

### 💻 Python Source Code

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
import numpy as np

# Create synthetic data
X = np.random.rand(100, 3)
y = np.random.choice([0, 1], size=100)

# Create a pipeline with MinMax scaling and KNN classifier
model = make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors=5, metric='euclidean'))
model.fit(X, y)

# Predict class labels
X_new = np.random.rand(5, 3)
y_pred = model.predict(X_new)
print("KNN Predictions:", y_pred)
```

---

## 8. Neural networks

**Category:** Supervised Learning

### 📐 Mathematical Foundation

$$a^{(l)} = \sigma(W^{(l)}a^{(l-1)} + b^{(l)})$$
A series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Highly interpretable baseline model | Sensitive to hyperparameters |
| Computationally efficient to train | Struggles with non-linear correlations |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(n \cdot p \cdot h_1 \cdot h_2 \dots)`

### 🚀 Real-World Applications

- Image and speech recognition
- natural language processing
- autonomous vehicles.

### 💻 Python Source Code

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple feedforward neural network (Multi-Layer Perceptron)
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)

# Instantiate the model, optimizer, and loss function
model = SimpleMLP(input_dim=10, hidden_dim=32, output_dim=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

---

## 9. Gradient Boosting Machines

**Category:** Supervised Learning

### 📐 Mathematical Foundation

$$F_m(x) = F_{m-1}(x) + \nu \sum \gamma_{im} I(x \in R_{im})$$
Produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It builds the model in a stage-wise fashion.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Highly interpretable baseline model | Sensitive to hyperparameters |
| Computationally efficient to train | Struggles with non-linear correlations |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(n \cdot p \cdot M)`

### 🚀 Real-World Applications

- Web search ranking
- ecology modeling
- credit risk analysis.

### 💻 Python Source Code

```python
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

# Generate classification data
X = np.random.rand(100, 5)
y = np.random.choice([0, 1], size=100)

# Fit GBM model
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X, y)

# Predict on new sample
X_new = np.random.rand(2, 5)
preds = model.predict(X_new)
print("GBM Predictions:", preds)
```

---

## 10. XGBoost

**Category:** Supervised Learning

### 📐 Mathematical Foundation

$$\mathcal{L}(\phi) = \sum_i l(y_i, \hat{y}_i) + \sum_k \Omega(f_k)$$
An optimized distributed gradient boosting library designed to be highly efficient, flexible and portable.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Highly interpretable baseline model | Sensitive to hyperparameters |
| Computationally efficient to train | Struggles with non-linear correlations |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(n \cdot p \cdot \log n)`

### 🚀 Real-World Applications

- Kaggle competitions
- high-energy physics
- sales forecasting.

### 💻 Python Source Code

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np

# Create synthetic binary classification dataset
X = np.random.rand(100, 5)
y = np.random.choice([0, 1], size=100)

# Convert to XGBoost DMatrix format
dtrain = xgb.DMatrix(X, label=y)

# Set hyperparameters
params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}

# Train the model
num_round = 50
bst = xgb.train(params, dtrain, num_round)

# Predict on new data
X_new = np.random.rand(5, 5)
dtest = xgb.DMatrix(X_new)
preds = bst.predict(dtest)
print("XGBoost Predictions:", preds)
```

---

## 11. K-means clustering

**Category:** Unsupervised Learning

### 📐 Mathematical Foundation

$$J = \sum_{j=1}^k \sum_{i=1}^n ||x_i^{(j)} - c_j||^2$$
Groups n observations into k clusters in which each observation belongs to the cluster with the nearest mean.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Highly interpretable baseline model | Sensitive to hyperparameters |
| Computationally efficient to train | Struggles with non-linear correlations |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(k \cdot n \cdot i \cdot p)`

### 🚀 Real-World Applications

- Market segmentation
- document clustering
- image compression.

### 💻 Python Source Code

```python
from sklearn.cluster import KMeans
import numpy as np

# Create synthetic 2D data with three distinct clusters
np.random.seed(42)
X = np.vstack([
    np.random.randn(50, 2) + [2, 2],
    np.random.randn(50, 2) + [-2, -2],
    np.random.randn(50, 2) + [2, -2]
])

# Initialize and fit K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
kmeans.fit(X)

# Retrieve cluster centers and labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_
print("Cluster Centers:\n", centers)
```

---

## 12. Hierarchical clustering

**Category:** Unsupervised Learning

### 📐 Mathematical Foundation

Agglomerative (bottom-up) or Divisive (top-down). It builds a hierarchy of clusters represented as a dendrogram.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Dimensionality reduction: PCA is effective in reducing the dimensionality of high-dimensional data while preserving most of the important information. This can lead to faster computation times, improved memory usage, and better model performance. | Flexibility in the number of clusters: Hierarchical clustering does not require specifying the number of clusters beforehand. Instead, it creates a hierarchy of nested clusters, and the number of clusters can be determined after analyzing the dendrogram. This flexibility is useful when the optimal number of clusters is unknown or can vary across different parts of the data. |
| Noise reduction: PCA can help filter out noise and minor variations in the data by focusing on the principal components that capture the most significant variance. This can lead to more robust and generalizable models. | Capturing complex structures: Hierarchical clustering can capture complex structures in the data, including clusters of varying sizes, shapes, and densities. It allows for a more nuanced view of the data, especially in situations where data points do not form well-separated clusters. |
| Visualization: PCA can be used for data visualization in lower-dimensional spaces (2D or 3D), making it easier to understand complex data structures and identify patterns. | Visual interpretability: The dendrogram produced by hierarchical clustering provides an intuitive visual representation of the clustering process. Analysts can visually explore the dendrogram and choose the appropriate level of clustering based on the problemʼs requirements. |
| Multicollinearity mitigation: PCA can address multicollinearity issues by creating orthogonal principal components, which can improve model stability and interpretability. | Connectivity constraints: Hierarchical clustering can incorporate connectivity constraints, allowing the algorithm to respect specific relationships or constraints between data points during the clustering process. |
| Pre-processing for Machine Learning: PCA is often used as a pre-processing step in machine learning pipelines to reduce the feature space before applying other algorithms. It can lead to improved model performance and faster training times. | Hierarchical ensemble clustering: Hierarchical clustering can be used in ensemble clustering, where multiple clustering algorithms are combined to improve the overall performance and robustness. |
| Feature extraction: PCA can be used for feature extraction, transforming a set of possibly correlated features into a new set of uncorrelated features that retain most of the dataʼs variability. | Nested clustering: The hierarchical nature of the output allows for nested clustering, where smaller clusters are naturally grouped into larger clusters. This nested structure can be beneficial for organizing and understanding hierarchical relationships in the data. |
| Nonlinear dimensionality reduction: t-SNE is well-suited for capturing complex and nonlinear structures in high-dimensional data. It can reveal intricate relationships among data points that may be difficult to detect using linear techniques like PCA. | Computational complexity: Agglomerative hierarchical clustering can be computationally expensive, especially for large datasets, as it requires O(n^3) operations in the worst-case scenario. |
| Data visualization: t-SNE is widely used for data visualization tasks. It can effectively project high-dimensional data into a lower-dimensional space (e.g., 2D or 3D), making it easier to visualize and interpret complex data patterns, clusters, and trends. | Scalability: Traditional hierarchical clustering can be less scalable compared to other clustering algorithms, particularly for large datasets. |
| Preservation of local structure: t-SNE focuses on preserving the local similarities between data points. This property makes it useful for visualizing fine-grained structures and clusters in the data. | Memory usage: The storage of the distance matrix can be memory-intensive for large datasets. |
| Useful for clustering: t-SNE can be applied as a pre-processing step to improve clustering performance. It helps separate clusters and facilitate the identification of natural groupings in the data. | Sensitivity to distance metric: The performance of hierarchical clustering heavily relies on the choice of distance metrics. Inappropriate distance metrics can lead to suboptimal clustering results. |
| Robust to outliers: t-SNE is relatively robust to the presence of outliers in the data. Outliers typically have a low influence on the t-SNE embedding, which helps maintain the integrity of the visualization. | Single linkage problem: Single linkage hierarchical clustering can suffer from the chaining effect, where clusters are connected by long chains of data points, leading to poor clustering results. |
| Visual appeal: t-SNE often produces visually appealing scatter plots, which are intuitive and easy to interpret, making it an attractive tool for data exploration and communication. | Not suitable for large datasets: Hierarchical clustering can become impractical for datasets with many data points due to its high computational and memory requirements. |
| Interpretable rules: ARM generates rules that are easily interpretable. These rules can be understood as if-then statements, which can be useful for understanding relationships and patterns in data. | Difficulty handling noisy data: Hierarchical clustering can be sensitive to noise in the data, leading to less reliable clustering results in the presence of outliers or noisy data points. |
| Versatility: ARM is not limited to market-basket analysis (its most popular application). It can be applied to any dataset where you want to uncover associations, such as in biology (finding gene patterns), web analytics (understanding user behavior), and so on. | Lack of global optimization: The hierarchical clustering process is a step-by-step merging or splitting of clusters, which may not lead to a globally optimal clustering solution. |
| Minimal domain knowledge required: You do not need extensive domain-specific knowledge to run and get meaningful outputs from ARM. It is an exploratory tool that reveals the underlying structure in the data. | Difficulty with irregularly shaped clusters: Traditional hierarchical clustering methods may struggle to handle irregularly shaped clusters or clusters with varying densities. |
| Modifiability: Parameters like support and confidence can be adjusted to change the strictness of the algorithm, which can be useful depending on the use-case. | Real world applications |
| Comprehensive: A priori explores all possible rule combinations and ensures nothing is overlooked, providing a thorough analysis. | Loss of interpretability: As PCA creates new features as linear combinations of the original features, the new dimensions may not have direct physical or intuitive meanings. This can reduce the modelʼs interpretability. |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(n^3)`

### 🚀 Real-World Applications

- Bioinformatics (evolutionary trees)
- social network analysis.

### 💻 Python Source Code

```python
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
X = np.random.rand(10, 2)

# Perform agglomerative hierarchical clustering using Ward's linkage
Z = linkage(X, method='ward')

# Plot the resulting dendrogram
plt.figure(figsize=(8, 5))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()
```

---

## 13. Principal Component Analysis

**Category:** Unsupervised Learning

### 📐 Mathematical Foundation

$$Z = XW$$
Transforms a large set of variables into a smaller one that still contains most of the information in the large set.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Highly interpretable baseline model | Sensitive to hyperparameters |
| Computationally efficient to train | Struggles with non-linear correlations |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(p^2 n + p^3)`

### 🚀 Real-World Applications

- Face recognition
- image compression
- gene data visualization.

### 💻 Python Source Code

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Create high-dimensional dataset
X = np.random.rand(100, 10)

# Standardize and project down to 2 principal components
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Reduced shape: {X_pca.shape}")
```

---

## 14. t-Distributed Stochastic Neighbor Embedding

**Category:** Unsupervised Learning

### 📐 Mathematical Foundation

$$P_{j|i} = \frac{\exp(-||x_i - x_j||^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-||x_i - x_k||^2 / 2\sigma_i^2)}$$
A non-linear dimensionality reduction technique well-suited for embedding high-dimensional data for visualization in a low-dimensional space of two or three dimensions.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Highly interpretable baseline model | Sensitive to hyperparameters |
| Computationally efficient to train | Struggles with non-linear correlations |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(n^2)`

### 🚀 Real-World Applications

- Genomic data visualization
- computer security (anomaly visualization).

### 💻 Python Source Code

```python
from sklearn.manifold import TSNE
import numpy as np

# Generate high-dimensional synthetic dataset
X = np.random.rand(100, 50)

# Project high-dimensional features down to 2D embeddings for visualization
tsne = TSNE(n_components=2, perplexity=30.0, random_state=42)
X_embedded = tsne.fit_transform(X)

print(f"Embedded Data Shape: {X_embedded.shape}")
```

---

## 15. Association Rule Mining

**Category:** Unsupervised Learning

### 📐 Mathematical Foundation

$$Support(A \to B) = P(A \cup B), \quad Confidence(A \to B) = P(B|A)$$
Finding frequent patterns, associations, correlations, or causal structures among sets of items in transaction databases.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Highly interpretable baseline model | Sensitive to hyperparameters |
| Computationally efficient to train | Struggles with non-linear correlations |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(2^p)`

### 🚀 Real-World Applications

- Market basket analysis
- inventory management
- user behavior analysis.

### 💻 Python Source Code

```python
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Transaction dataset representing market baskets
dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice Cream', 'Eggs']]

# Convert transaction list to one-hot encoded dataframe
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori algorithm to identify frequent itemsets
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print(rules[['antecedents', 'consequents', 'support', 'confidence']])
```

---

## 16. DBSCAN

**Category:** Unsupervised Learning

### 📐 Mathematical Foundation

Density-Based Spatial Clustering of Applications with Noise. It groups together points that are closely packed together, marking as outliers points that lie alone in low-density regions.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| No need to specify cluster count: Unlike many clustering algorithms, DBSCAN does not require the user to specify the number of clusters a priori. | A key disadvantage of DBSCAN is that to every point ends up in a cluster which is a challenge for certain applications of clustering. |
| Ability to detect arbitrarily shaped clusters: While methods like k-means assume that clusters are convex and isotropic, DBSCAN can find clusters of any shape, making it more versatile in real-world scenarios. | Parameter sensitivity: DBSCAN requires two parameters, eps and minPts, which can greatly influence the clustering results. Determining the best values for these parameters is not straightforward and often requires domain knowledge or multiple iterations. |
| Noise handling: DBSCAN is designed to handle noise. During the clustering process, it can identify and exclude outlier data points that do not belong to any cluster. | Varying density clusters: DBSCAN might struggle with datasets where clusters have varying densities. A single value of eps may not be suitable for all clusters. |
| Stability: Unlike k-means which can produce different clusters depending on initial conditions, DBSCAN always produces the same clusters when given the same input and parameters. | Border points: A point that is reachable from two clusters can be part of either cluster, which can sometimes lead to ambiguous results. |
| No assumption of cluster geometry: DBSCAN does not make strong assumptions about the structure and distribution of clusters, unlike some other algorithms. | Difficulty in high dimensions: The curse of dimensionality affects DBSCAN like many other clustering algorithms. In high-dimensional spaces, the concept of distance becomes less clear, which can impact the efficacy of DBSCAN. |
| Scalability: With appropriate data structures like KD-Tree or Ball-Tree, DBSCAN can be optimized to handle larger datasets efficiently. | Not well-suited for clusters of different sizes: Clusters of varying sizes and densities might not be effectively captured with a single parameter setting in DBSCAN. |
| However, it is essential to understand that the success of DBSCAN in these applications often depends on appropriate parameter settings (like distance threshold and minimum points) and domain-specific knowledge. | Scalability issues: While optimized structures can make DBSCAN efficient for moderately large datasets, it might still face challenges with extremely large datasets. |
| Soft clustering: GMM assigns a probability to each point for every cluster, indicating the likelihood of that point belonging to the cluster. This is more informative than hard clustering methods like k-means, which simply assign a point to the nearest cluster. | GMMs are a popular method in machine learning and statistics for clustering and density estimation. They work by representing a complex distribution as a combination of multiple Gaussian distributions. |
| Flexibility in shape: Unlike k-means, which assumes that clusters are spherical, GMM can model elliptical clusters since each Gaussian has its own covariance matrix. | Sensitivity to initialization: Like k-means, the results might vary with different initializations. It is common to run the algorithm multiple times to get a good solution. |
| Density estimation: Beyond clustering, GMMs can be used for estimating the density function of the data. | Risk of overfitting: Because of its flexibility, GMM can sometimes fit noise, leading to overfitting. Regularization techniques or choosing the right number of components can mitigate this. |
| Parameter estimation: The EM algorithm, which is used to estimate parameters in GMM, provides a way to get maximum likelihood estimates. | Complexity: The computational cost can be high, especially when the number of dimensions or components is large. Each step of the EM algorithm involves computing, inverting, and multiplying matrices, which are expensive operations. |
| Well-understood theory: Gaussian processes and the EM algorithm are well studied, which means a lot of resources and theoretical guarantees are available. | Local optima: The EM algorithm can get stuck in local optima. A common practice is to run the algorithm from multiple random starting points. |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(n \log n)`

### 🚀 Real-World Applications

- Geospatial data analysis
- satellite imagery
- anomaly detection.

### 💻 Python Source Code

```python
from sklearn.cluster import DBSCAN
import numpy as np

# Generate synthetic dataset of density-based clusters
X = np.vstack([
    np.random.randn(50, 2) + [2, 2],
    np.random.randn(50, 2) + [-2, -2],
    np.random.randn(20, 2) + [0, 8]
])

# Initialize and fit DBSCAN
db = DBSCAN(eps=1.0, min_samples=5)
labels = db.fit_predict(X)

print("Cluster Labels assigned (-1 represents noise):", set(labels))
```

---

## 17. Gaussian Mixture Models

**Category:** Unsupervised Learning

### 📐 Mathematical Foundation

$$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$$
A probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Highly interpretable baseline model | Sensitive to hyperparameters |
| Computationally efficient to train | Struggles with non-linear correlations |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(n \cdot K \cdot p^2)`

### 🚀 Real-World Applications

- Customer profiling
- speech recognition (acoustic modeling)
- image segmentation.

### 💻 Python Source Code

```python
from sklearn.mixture import GaussianMixture
import numpy as np

# Generate synthetic data with two overlapping clusters
np.random.seed(42)
X = np.vstack([
    np.random.randn(100, 2) + [3, 3],
    np.random.randn(100, 2) + [-3, -3]
])

# Fit GMM with 2 components
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm.fit(X)

# Predict cluster labels and estimate densities
labels = gmm.predict(X)
print("Component Means:\n", gmm.means_)
```

---

## 18. Autoencoders

**Category:** Unsupervised Learning

### 📐 Mathematical Foundation

$$L(x, g(f(x)))$$
Type of neural network used to learn efficient data codings in an unsupervised manner. The aim is to learn a representation (encoding) for a set of data, typically for dimensionality reduction.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Robust feature learning: Deep generative models are excellent at learning a rich set of features from the data. This is particularly useful when the feature space is complex and high-dimensional, such as in images or text. | Computational complexity: These models often require more computational resources compared to simpler methods, making them difficult to deploy on systems with limited computational capabilities. |
| Data augmentation: These models can generate new data points that are like the training data, which can be used to augment the training set and improve the performance of the model. | Difficulty in training: Training deep generative models can be challenging due to issues like mode collapse, vanishing/exploding gradients, or getting stuck in local minima. |
| Robustness to overfitting: Generative models provide a regularizing effect, making them more robust to overfitting, especially when the amount of labeled data is limited. | Hyperparameter sensitivity: The performance can be highly sensitive to the choice of hyperparameters and tuning them can be time-consuming. |
| Effective use of unlabeled data: VAT is particularly powerful in scenarios where labeled data is scarce. It makes good use of unlabeled data to improve model generalization. | Lack of interpretability: Deep generative models are often seen as black boxes, making them difficult to interpret or understand, which is a critical drawback in sensitive applications like healthcare or finance. |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(n \cdot p \cdot h)`

### 🚀 Real-World Applications

- Anomaly detection
- denoising
- feature learning.

### 💻 Python Source Code

```python
import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        # Encoder projects down to low-dimensional latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        # Decoder reconstructs high-dimensional input from latent space
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = Autoencoder(input_dim=784, encoding_dim=32)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

---

## 19. Anomaly detection

**Category:** Unsupervised Learning

### 📐 Mathematical Foundation

Identifying rare items, events or observations which raise suspicions by differing significantly from the majority of the data.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Highly interpretable baseline model | Sensitive to hyperparameters |
| Computationally efficient to train | Struggles with non-linear correlations |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(n \log n) \text{ (Isolation Forest)}`

### 🚀 Real-World Applications

- Fraud detection
- system health monitoring
- network security.

### 💻 Python Source Code

```python
from sklearn.ensemble import IsolationForest
import numpy as np

# Create training data with majority inliers
np.random.seed(42)
X = np.random.randn(100, 2)

# Fit Isolation Forest
clf = IsolationForest(contamination=0.1, random_state=42)
clf.fit(X)

# Predict anomaly score (-1 for anomalies, 1 for inliers)
X_test = np.array([[0.1, -0.2], [5.0, 5.0]])
predictions = clf.predict(X_test)
print("Isolation Forest Predictions:", predictions)
```

---

## 20. Latent Dirichlet Allocation

**Category:** Unsupervised Learning

### 📐 Mathematical Foundation

$$p(D|\alpha, \beta) = \prod_{d=1}^M \int p(\theta_d|\alpha) \prod_{n=1}^{N_d} \sum_{z_{dn}} p(z_{dn}|\theta_d) p(w_{dn}|z_{dn}, \beta) d\theta_d$$
A generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Excellent interpretability for large collections of unstructured text | Suffers from computational scaling issues on massive corpora |
| Works well as a generative document-topic clustering mechanism | Requires manual pre-specification of topic parameter k |
|  | Sensitive to document length and vocabulary preprocessing |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(n \cdot k)`

### 🚀 Real-World Applications

- Topic modeling
- document classification
- image labeling.

### 💻 Python Source Code

```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Dummy collection of text documents
documents = [
    "Machine learning neural networks algorithms and data science training",
    "Deep learning artificial neural networks model validation techniques",
    "Customer segmentation and market basket association rules classification",
    "Clustering and dimensional reduction techniques with PCA and K-Means"
]

# Extract feature frequencies
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# Fit Latent Dirichlet Allocation (LDA) for topic modeling
lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(X)

print("Topic word distribution shapes:", lda.components_.shape)
```

---

## 21. Q-Learning

**Category:** Reinforcement Learning

### 📐 Mathematical Foundation

$$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$
A model-free reinforcement learning algorithm to learn the quality of actions telling an agent what action to take under what circumstances.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Model-free approach requiring zero environment dynamics information | Suffers severely from the curse of dimensionality in large state spaces |
| Tabular settings offer robust mathematical convergence guarantees | Overestimation bias of Q-values due to the max operator |
|  | High variance during early exploration stages |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(|S| \cdot |A|)`

### 🚀 Real-World Applications

- Pathfinding
- game AI
- resource management.

### 💻 Python Source Code

```python
import numpy as np

class QLearningAgent:
    def __init__(self, states_count, actions_count, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros((states_count, actions_count))
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.q_table.shape[1])  # Explore
        else:
            return np.argmax(self.q_table[state])           # Exploit

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * (target - predict)
```

---

## 22. Deep Q-Networks

**Category:** Reinforcement Learning

### 📐 Mathematical Foundation

$$L(\theta) = \mathbb{E}[(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2]$$
Combines Q-Learning with deep neural networks. It uses experience replay and target networks to stabilize learning.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Extends Q-learning to high-dimensional state spaces (e.g. raw pixels) | High sample inefficiency |
| Experience replay buffer breaks temporal correlations, stabilizing training | Struggles with continuous action spaces |
|  | Training can be unstable and highly sensitive to hyperparameter tuning |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(\text{Network parameters})`

### 🚀 Real-World Applications

- Atari games
- robotics control
- autonomous driving.

### 💻 Python Source Code

```python
import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, state):
        return self.network(state)
```

---

## 23. Policy gradient methods

**Category:** Reinforcement Learning

### 📐 Mathematical Foundation

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi [\nabla_\theta \log \pi_\theta(a|s) G_t]$$
A class of reinforcement learning algorithms that optimize the policy directly using gradient descent.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Directly optimizes the policy objective | Very high variance in gradient estimates |
| Naturally handles continuous and stochastic action spaces | Prone to getting trapped in suboptimal local optima |
| Smooth convergence properties compared to value-based methods | Extremely poor sample efficiency |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(\text{Policy parameters})`

### 🚀 Real-World Applications

- Robotics
- financial trading
- continuous control tasks.

### 💻 Python Source Code

```python
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.fc(x)
```

---

## 24. Advantage Actor-Critic

**Category:** Reinforcement Learning

### 📐 Mathematical Foundation

$$A(s, a) = Q(s, a) - V(s)$$
A synchronous reinforcement learning method where multiple agents interact with their environments simultaneously.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Synchronous updates allow parallelized, highly efficient training across multiple environments | Relies on highly synchronized steps, causing bottlenecks if environment steps take variable time |
| Critic baseline reduces policy gradient variance | Complex model architecture to balance and tune |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(\text{Actor + Critic parameters})`

### 🚀 Real-World Applications

- Real-time strategy games
- robot locomotion.

### 💻 Python Source Code

```python
import torch
import torch.nn as nn
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Shared feature extractor
        self.common = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, action_dim)      # Policy head
        self.critic = nn.Linear(128, 1)              # Value head

    def forward(self, state):
        x = self.common(state)
        policy_dist = torch.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return policy_dist, state_value
```

---

## 25. Trust Region Policy Optimization

**Category:** Reinforcement Learning

### 📐 Mathematical Foundation

$$\max_{\theta} \mathbb{E} [\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A_{\theta_{old}}(s, a)] \text{ s.t. } KL(\pi_{\theta_{old}} || \pi_\theta) \leq \delta$$
Optimizes policies with a guarantee of monotonic improvement by constraining the policy update.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Guarantees monotonic policy improvement | Computationally expensive due to calculating the Hessian of the KL divergence |
| Mathematical constraints prevent destructive large updates, ensuring high sample safety | Extremely complex implementation relying on Conjugate Gradient methods |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(\text{High due to Hessian calculation})`

### 🚀 Real-World Applications

- Complex robotic tasks
- simulated locomotion.

### 💻 Python Source Code

```python
import torch
import torch.nn as nn

class TRPOObjective(nn.Module):
    def __init__(self):
        super(TRPOObjective, self).__init__()

    def forward(self, new_probs, old_probs, advantages):
        # Surrogate advantage loss
        ratio = new_probs / (old_probs + 1e-8)
        surrogate_loss = ratio * advantages
        return -torch.mean(surrogate_loss)
```

---

## 26. Asynchronous Advantage Actor-Critic

**Category:** Reinforcement Learning

### 📐 Mathematical Foundation

Asynchronous version of A2C where multiple worker threads update a global model asynchronously.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Asynchronous updates break temporal correlations without an experience replay buffer | Thread-unsynchronized parameter updates can lead to instability |
| Faster training times by utilizing multiple CPU/GPU worker threads | Complex multiprocessing architecture |
|  | Hard to debug thread race conditions |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(\text{Actor + Critic parameters})`

### 🚀 Real-World Applications

- Complex 3D simulations
- multi-agent systems.

### 💻 Python Source Code

```python
import torch
import torch.nn as nn
import torch.multiprocessing as mp

class GlobalNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(GlobalNetwork, self).__init__()
        self.fc = nn.Linear(state_dim, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        h = torch.relu(self.fc(x))
        return torch.softmax(self.actor(h), dim=-1), self.critic(h)

# Shared optimizer across worker processes
class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3):
        super(SharedAdam, self).__init__(params, lr=lr)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
```

---

## 27. Proximal Policy Optimization

**Category:** Reinforcement Learning

### 📐 Mathematical Foundation

$$L^{CLIP}(\theta) = \mathbb{E} [\min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t)]$$
A simpler-to-implement and more stable alternative to TRPO using a clipped objective function.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Provides stability and monotonic guarantees of TRPO but with much simpler, computationally cheaper first-order gradient updates | High sample complexity |
| Widely considered the industry benchmark for general reinforcement learning | Sensitive to clipping hyperparameter epsilon and value loss scaling |
|  | Policy can occasionally become overly conservative in exploration |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(\text{Scales with data})`

### 🚀 Real-World Applications

- OpenAI Five (Dota 2)
- robotics
- general RL benchmark.

### 💻 Python Source Code

```python
import torch
import torch.nn as nn

class PPOLoss(nn.Module):
    def __init__(self, clip_eps=0.2):
        super(PPOLoss, self).__init__()
        self.clip_eps = clip_eps

    def forward(self, new_probs, old_probs, advantages):
        # Probability ratio r_t(theta)
        ratio = new_probs / (old_probs + 1e-8)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        # Clipped surrogate objective
        return -torch.mean(torch.min(surr1, surr2))
```

---

## 28. Deep Deterministic Policy Gradient

**Category:** Reinforcement Learning

### 📐 Mathematical Foundation

Actor-critic, model-free algorithm for continuous action spaces. It uses the deterministic policy gradient theorem.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Sample-efficient off-policy method designed specifically for high-dimensional, continuous action spaces | High instability during training due to overestimation bias |
|  | Very sensitive to hyperparameter tuning and noise exploration parameters |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(\text{Actor + Critic parameters})`

### 🚀 Real-World Applications

- Mechanical control
- industrial process optimization.

### 💻 Python Source Code

```python
import torch
import torch.nn as nn

class DeterministicActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(DeterministicActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.fc2(torch.relu(self.fc1(x)))
```

---

## 29. Twin Delayed Deep Deterministic Policy Gradient

**Category:** Reinforcement Learning

### 📐 Mathematical Foundation

Addresses overestimation bias in DDPG using twin Q-networks and delayed policy updates.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Mitigates the overestimation bias of DDPG using twin critic networks and delayed policy updates | High hyperparameter sensitivity |
| Significantly more stable and robust than baseline DDPG | Computationally slower than DDPG per update step |
|  | Still suffers from general off-policy RL sample inefficiencies |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(\text{Higher than DDPG})`

### 🚀 Real-World Applications

- High-precision robotics
- fluid dynamics control.

### 💻 Python Source Code

```python
import torch
import torch.nn as nn

class TD3TwinCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(TD3TwinCritic, self).__init__()
        # Critic network 1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # Critic network 2 (Twin)
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=-1)
        return self.q1(xu), self.q2(xu)
```

---

## 30. Soft Actor-Critic

**Category:** Reinforcement Learning

### 📐 Mathematical Foundation

$$J(\pi) = \sum_t \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} [r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))]$$
Maximum entropy RL algorithm that provides sample-efficient learning and robustness.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Maximum entropy framework encourages aggressive exploration, resulting in exceptional sample efficiency and robust policies that generalize well | Highly sensitive to the temperature parameter alpha (requires auto-tuning) |
|  | Challenging to balance actor and twin critic networks |
|  | High computational overhead per gradient update |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(\text{Actor + Critic parameters})`

### 🚀 Real-World Applications

- Real-world robot manipulation
- complex motor tasks.

### 💻 Python Source Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-20, log_std_max=2):
        super(GaussianPolicy, self).__init__()
        self.fc = nn.Linear(state_dim, 128)
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        x = F.relu(self.fc(state))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)
        return action
```

---

## 31. Self-training

**Category:** Semi-supervised Learning

### 📐 Mathematical Foundation

Iterative process where a model labels unlabeled data and retrains on its own predictions.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Simple, conceptual framework that can leverage any standard base classifier | Danger of error reinforcement where early incorrect predictions are pseudo-labeled and degrade model performance over iterations |
| Highly cost-effective by utilizing unlabeled datasets without external annotations | Highly dependent on a good initial labeled set |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(I \cdot n \cdot p) \text{ where } I \text{ is iterations}`

### 🚀 Real-World Applications

- NLP (with small labeled data)
- medical image classification.

### 💻 Python Source Code

```python
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
import numpy as np

# Prepare semi-supervised dataset (unlabeled marked as -1)
X = np.random.rand(150, 4)
y = np.random.choice([0, 1], size=150)
y[100:] = -1  # Mask 50 labels as unlabeled

# Train a Self-Training model using SVM classifier with probability estimates
svc = SVC(probability=True, kernel='linear')
self_training_model = SelfTrainingClassifier(svc, threshold=0.75)
self_training_model.fit(X, y)
```

---

## 32. Co-training

**Category:** Semi-supervised Learning

### 📐 Mathematical Foundation

Two classifiers are trained on different views of the data and iteratively provide each other with labeled examples.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| High sample efficiency by using independent disjoint feature views to cross-label samples | Requires the strict assumption of view independence and sufficiency (which is extremely rare and difficult to satisfy in real-world datasets) |
| Highly robust to early pseudo-label noise compared to self-training |  |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(I \cdot n \cdot p)`

### 🚀 Real-World Applications

- Web page classification
- multi-modal biometrics.

### 💻 Python Source Code

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB

class CoTrainingClassifier:
    def __init__(self, clf1=GaussianNB(), clf2=GaussianNB(), p=1, n=1):
        self.clf1 = clf1  # Classifier for view 1
        self.clf2 = clf2  # Classifier for view 2
        self.p = p        # Positive samples to add per iteration
        self.n = n        # Negative samples to add per iteration

    def fit(self, X1, X2, y):
        # Simple co-training algorithm framework
        # X1 is feature view 1, X2 is feature view 2
        # Train base classifiers on initial labeled set
        labeled_idx = np.where(y != -1)[0]
        self.clf1.fit(X1[labeled_idx], y[labeled_idx])
        self.clf2.fit(X2[labeled_idx], y[labeled_idx])
```

---

## 33. Multi-view learning

**Category:** Semi-supervised Learning

### 📐 Mathematical Foundation

Learning from multiple distinct feature sets (views) to improve performance.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Integrates heterogeneous data representations (e.g., audio and video) into a unified projection | High computational cost of aligning views |
| Enhances generalization by utilizing complementary features | Sensitive to missing views or noise in a single channel |
|  | Requires specialized co-regularization algorithms |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(\text{Depends on base models})`

### 🚀 Real-World Applications

- Video analysis (audio + visual)
- multi-sensor fusion.

### 💻 Python Source Code

```python
import numpy as np
from sklearn.decomposition import CCA

class MultiViewRepresentation:
    def __init__(self, n_components=2):
        self.cca = CCA(n_components=n_components)

    def fit_transform(self, X1, X2):
        # Canonical Correlation Analysis to find a shared latent space
        self.cca.fit(X1, X2)
        X1_c, X2_c = self.cca.transform(X1, X2)
        # Fuse projections by concatenation
        return np.hstack([X1_c, X2_c])
```

---

## 34. Expectation-Maximization

**Category:** Semi-supervised Learning

### 📐 Mathematical Foundation

Iterative method to find maximum likelihood estimates of parameters in models with latent variables.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Elegant, mathematically rigorous framework for fitting latent variable models | Prone to getting trapped in local maxima |
| Guarantees monotonic convergence of log-likelihood | Highly sensitive to initial parameter seeding |
|  | Computationally slow for large datasets |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(I \cdot n \cdot p)`

### 🚀 Real-World Applications

- Clustering (GMM)
- missing data imputation
- HMM training.

### 💻 Python Source Code

```python
from sklearn.mixture import GaussianMixture
import numpy as np

# EM clustering via GMM model
X = np.random.rand(100, 2)
gmm = GaussianMixture(n_components=3, covariance_type='full')
gmm.fit(X)
labels = gmm.predict(X)
```

---

## 35. Graph-based methods

**Category:** Semi-supervised Learning

### 📐 Mathematical Foundation

Model data as a graph and propagate labels through edges based on similarity.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Captures complex, non-linear manifold structures of data | Highly sensitive to the quality of graph construction (edge weights) |
| Extremely effective at propagating labels when very few labeled nodes exist | Does not scale well computationally to large datasets due to graph Laplacian inversion costs |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(n^2) \text{ or } O(n^3)`

### 🚀 Real-World Applications

- Social network classification
- protein-protein interaction prediction.

### 💻 Python Source Code

```python
from sklearn.semi_supervised import LabelSpreading
import numpy as np

X = np.random.rand(100, 2)
y = np.random.choice([0, 1], size=100)
y[75:] = -1  # Unlabeled nodes

model = LabelSpreading(kernel='knn', alpha=0.2)
model.fit(X, y)
```

---

## 36. Transductive Support Vector Machines

**Category:** Semi-supervised Learning

### 📐 Mathematical Foundation

Extends SVM to semi-supervised learning by maximizing the margin on both labeled and unlabeled data.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Maximizes the margin over both labeled and unlabeled data, providing strong generalization bounds on small datasets | Non-convex optimization problem that is computationally NP-hard to solve exactly |
|  | Approximate methods are sensitive to initialization |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(2^{n_{unlabeled}} \cdot n^3) \text{ (approximate algorithms exist)}`

### 🚀 Real-World Applications

- Text classification where unlabeled data is available at training time.

### 💻 Python Source Code

```python
import numpy as np
from sklearn.svm import SVC

class SimpleTSVM:
    def __init__(self, C=1.0, cl_unlabeled=0.1):
        self.C = C
        self.cl_unlabeled = cl_unlabeled
        self.clf = SVC(kernel='linear', C=C)

    def fit(self, X_labeled, y_labeled, X_unlabeled):
        # Initial supervised training on labeled set
        self.clf.fit(X_labeled, y_labeled)
        # Predict pseudo-labels for unlabeled points
        pseudo_labels = self.clf.predict(X_unlabeled)
        # Iteratively train on combined labeled & pseudo-labeled points
        X_combined = np.vstack([X_labeled, X_unlabeled])
        y_combined = np.hstack([y_labeled, pseudo_labels])
        self.clf.fit(X_combined, y_combined)
```

---

## 37. Co-regularization

**Category:** Semi-supervised Learning

### 📐 Mathematical Foundation

Minimizes a loss function that includes a penalty for disagreement between classifiers trained on different views.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Enforces agreement across different feature representations, providing robust regularization constraints that limit overfitting | Requires multi-view feature setups |
|  | High hyperparameter sensitivity |
|  | Complex joint loss optimization |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(\text{Depends on base models})`

### 🚀 Real-World Applications

- Multi-view clustering
- collaborative filtering.

### 💻 Python Source Code

```python
import numpy as np
from sklearn.linear_model import Ridge

class CoRegularizedRegression:
    def __init__(self, alpha=1.0, mu=0.5):
        self.clf1 = Ridge(alpha=alpha)
        self.clf2 = Ridge(alpha=alpha)
        self.mu = mu  # Agreement regularization weight

    def fit(self, X1_labeled, X2_labeled, y_labeled, X1_unlabeled, X2_unlabeled, iterations=5):
        # Iterative update minimizing self-loss and view disagreement
        for _ in range(iterations):
            self.clf1.fit(X1_labeled, y_labeled)
            self.clf2.fit(X2_labeled, y_labeled)
            # Regularize towards agreement on unlabeled data
            pred1 = self.clf1.predict(X1_unlabeled)
            pred2 = self.clf2.predict(X2_unlabeled)
            # Adjust targets using multi-view consensus
            y_unlabeled = 0.5 * (pred1 + pred2)
```

---

## 38. Deep generative models

**Category:** Semi-supervised Learning

### 📐 Mathematical Foundation

Models like VAEs or GANs that learn to generate new data samples from the same distribution as the training data.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Outstanding capacity to learn complex, high-dimensional target distributions | High training instability (e.g., mode collapse in GANs) |
| Creates high-fidelity synthetic samples for data augmentation | Challenging to evaluate generation quality |
|  | High computational and memory footprint |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(\text{Network parameters})`

### 🚀 Real-World Applications

- Image synthesis
- data augmentation
- anomaly detection.

### 💻 Python Source Code

```python
import torch
import torch.nn as nn

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        # Encoder projects inputs to latent distribution parameters
        self.fc_enc = nn.Linear(input_dim, 64)
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        # Decoder generates samples from latent vector
        self.fc_dec1 = nn.Linear(latent_dim, 64)
        self.fc_dec2 = nn.Linear(64, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # Latent sampling trick

    def forward(self, x):
        h = torch.relu(self.fc_enc(x))
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        reconstruction = torch.sigmoid(self.fc_dec2(torch.relu(self.fc_dec1(z))))
        return reconstruction, mu, logvar
```

---

## 39. Virtual Adversarial Training

**Category:** Semi-supervised Learning

### 📐 Mathematical Foundation

Regularizes the model by making it robust to virtual adversarial perturbations in the input space.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Local distribution robustness regularizer that does not require labeled data | High computational overhead due to multiple forward/backward passes to estimate the adversarial perturbation direction |
| Yields outstanding classification boundaries by pushing model to be robust against adversarial perturbations | Sensitive to perturbation epsilon |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(\text{Standard training + perturbation cost})`

### 🚀 Real-World Applications

- Semi-supervised image classification
- NLP robustness.

### 💻 Python Source Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VATLoss(nn.Module):
    def __init__(self, xi=1e-6, eps=1.0, ip=1):
        super(VATLoss, self).__init__()
        self.xi = xi      # Perturbation scale for gradient approximation
        self.eps = eps    # Adversarial epsilon bound
        self.ip = ip      # Power iterations count

    def forward(self, model, x):
        # Generate virtual adversarial perturbation to maximize KL divergence
        with torch.no_grad():
            pred = F.softmax(model(x), dim=-1)
        d = torch.randn_like(x)
        d = F.normalize(d, p=2, dim=-1)
        for _ in range(self.ip):
            d.requires_grad_()
            pred_hat = F.softmax(model(x + self.xi * d), dim=-1)
            kl = F.kl_div(pred_hat.log(), pred, reduction='batchmean')
            kl.backward()
            d = F.normalize(d.grad, p=2, dim=-1)
        r_vadv = self.eps * d
        return r_vadv
```

---

## 40. Tri-training

**Category:** Semi-supervised Learning

### 📐 Mathematical Foundation

Uses three classifiers; an unlabeled sample is labeled if two of the three classifiers agree on the label.

### ⚙️ Pros & Cons

| **Advantages (Pros)** | **Disadvantages (Cons)** |
| :--- | :--- |
| Eliminates the view independence requirement of Co-training by utilizing bootstrap sampling to construct three diverse classifiers | High computational overhead (requires training and querying three separate classifiers) |
| Highly robust to pseudo-labeling noise | Consensus voting can occasionally collapse if two classifiers synchronize incorrect predictions |

### ⏱️ Computational & Space Complexity

- **Big O Complexities:** `O(3 \cdot I \cdot n \cdot p)`

### 🚀 Real-World Applications

- Spam detection
- fraud detection
- NLU.

### 💻 Python Source Code

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class TriTrainingClassifier:
    def __init__(self, clf1=DecisionTreeClassifier(), clf2=DecisionTreeClassifier(), clf3=DecisionTreeClassifier()):
        self.clfs = [clf1, clf2, clf3]

    def fit(self, X, y):
        # X contains both labeled and unlabeled data, unlabeled y marked as -1
        labeled_idx = np.where(y != -1)[0]
        unlabeled_idx = np.where(y == -1)[0]
        # Train three independent classifiers using bootstrap sampling
        for clf in self.clfs:
            bootstrap = np.random.choice(labeled_idx, size=len(labeled_idx), replace=True)
            clf.fit(X[bootstrap], y[bootstrap])
```

---
