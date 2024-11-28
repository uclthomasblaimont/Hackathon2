# %% [markdown]
# # **[LEPL1109] - STATISTICS AND DATA SCIENCES**
# ## **Hackaton 02 - Classification: Diabetes Health indicators**
# \
# Prof. D. Hainaut\
# Prod. L. Jacques\
# \
# \
# Adrien Banse (adrien.banse@uclouvain.be)\
# Jana Jovcheva (jana.jovcheva@uclouvain.be)\
# François Lessage (francois.lessage@uclouvain.be)\
# Sofiane Tanji (sofiane.tanji@uclouvain.be)

# %% [markdown]
# ![alt text](figures/diab_illustration.jpg)

# %% [markdown]
# <div class="alert alert-danger">
# <b>[IMPORTANT] Read all the documentation</b>  <br>
#     Make sure that you read the whole notebook, <b>and</b> the <code>README.md</code> file in the folder.
# </div>

# %% [markdown]
# # **Guidelines and Deliverables**
# 
# *   This hackaton is due on the **29 November 2024 at 23h59**
# *   Copying code or answers from other groups (or from the internet) is strictly forbidden. <b>Each source of inspiration (stack overflow, git, other groups, ChatGPT...) must be clearly indicated!</b>
# *  This notebook (with the "ipynb" extension) file, the Python source file (".py"), the report (PDF format) and all other files that are necessary to run your code must be delivered on <b>Moodle</b>.
# * Only the PDF report and the python source file will be graded, both on their content and the quality of the text / figures.
#   * 4/10 for the code.
#   * 4/10 for the Latex report.
#   * 2/10 for the vizualisation. <br><br>
# 
# <div class="alert alert-info">
# <b>[DELIVERABLE] Summary</b>  <br>
# After the reading of this document (and playing with the code!), we expect you to provide us with:
# <ol>
#    <li> a PDF file (written in LaTeX, see example on Moodle) that answers all the questions below. The report should contain high quality figures with named axes (we recommend saving plots with the <samp>.pdf</samp> extension);
#    <li> a Python file with your classifier implementation. Please follow the template that is provided and ensure it passes the so-called <i>sanity</i> tests;
#    <li> this Jupyter Notebook (it will not be read, just checked for plagiarism);
#    <li> and all other files (not the datasets!) we would need to run your code.
# </ol>
# </div>
# 
# As mentioned above, plagiarism is forbidden. However, we cannot forbid you to use artificial intelligence BUT we remind you that the aim of this project is to learn classification on your own and with the help of the course material. Finally, we remind you that for the same question, artificial intelligence presents similar solutions, which could be perceived as a form of plagiarism.

# %% [markdown]
# # **Context & Objective**
# Diabetes is among the most prevalent chronic diseases in the United States, impacting millions of Americans each year and exerting a significant financial burden on the economy. Diabetes is a serious chronic disease in which individuals lose the ability to effectively regulate levels of glucose in the blood, and can lead to reduced quality of life and life expectancy. After different foods are broken down into sugars during digestion, the sugars are then released into the bloodstream. This signals the pancreas to release insulin. Insulin helps enable cells within the body to use those sugars in the bloodstream for energy. Diabetes is generally characterized by either the body not making enough insulin or being unable to use the insulin that is made as effectively as needed.\
# Complications like heart disease, vision loss, lower-limb amputation, and kidney disease are associated with chronically high levels of sugar remaining in the bloodstream for those with diabetes. While there is no cure for diabetes, strategies like losing weight, eating healthily, being active, and receiving medical treatments can mitigate the harms of this disease in many patients. Early diagnosis can lead to lifestyle changes and more effective treatment, making predictive models for diabetes risk important tools for public and public health officials.
# 
# You work in the diabetology department at **Saint Luc University Hospital**. The head of the department has asked you to find a solution for classifying and predicting **whether patients are at high risk of developing diabetes**. This will enable them to schedule an appointment with these patients to set up prevention tools. To do this, you have a database of patients who have passed through the department in recent years. In addition, the head of the department feels that the poll is too long, and would like to **reduce the number of questions while maintaining the reliability and quality of the results**.\
# Your aim is to determine which characteristics are relevant and enable reliable patient classification. Be careful, don’t let a potential diabetic patient slip through the cracks. The rest of this document will guide you in this process.
# 
# ## **Dataset description**
# 
#  
# The data set is a real-world data set based on a survey (BRFSS) conducted by the Centers for Disease Control and Prevention in the USA some ten years ago.\
# The Behavioral Risk Factor Surveillance System (BRFSS) is an annual telephone health survey conducted by the Centers for Disease Control and Prevention. Each year, the survey collects responses from over 400,000 Americans on health-related risk behaviors, chronic diseases and use of preventive services. The survey has been conducted annually since 1984. It contains 22 headings and around 70,000 entries.
# 
# 
# <img src="figures/Features_table.png" alt="drawing" width="800"/>
# 
# ## **Notebook structure**
# 
# * PART 1 - Preliminaries
#    - 1.1 - Importing the packages
#    - 1.2 - Importing the dataset
#    - 1.3 - Is the dataset balanced?
#    - 1.4 - Scale the dataset
#     <br><br>
# * PART 2 - Correlation
#    - 2.1 - Correlation matrix 
#    - 2.2 -Analyze the correlation with diabetes
#    - 2.3 - Model selection and parameters tuning
#    - 2.4 - Precision-Recall curve and thresholding
#    <br><br>
# * PART 3 - Classifiers
#    - 3.1 - Linear regressor
#    - 3.2 - Logisitic regressor
#    - 3.3 - KNN regressor
#    <br><br>
# * PART 4 - Validation metrics
#    - 4.1 - Precision score
#    - 4.2 - Recall score
#    - 4.3 - F1 score
#    <br><br>
# * PART 5 - Reduce the questionnaire size
#    - 5.1 - K-Fold preparation
#    - 5.2 - Find the right combination length/regressor
#    - 5.3 - Visualize the scores
#    <br><br>   
# * PART 6 - Visualization
#    - 6.1 - Visualize your results
# 
# We filled this notebook with preliminary (trivial) code. This practice makes possible to run each cell, even the last ones, without throwing warnings. <b>Take advantage of this aspect to divide the work between all team members!</b> <br><br>
# Remember that many libraries exist in Python, so many functions have already been developed. Read the documentation and don't reinvent the wheel! You can import whatever you want.
# 

# %% [markdown]
# <br><font size=7 color=#009999> <b>PART I - Preliminaries</b> </font> <br><br>

# %% [markdown]
# In this part of the hackathon, we will import the necessary packages, then we will import the dataset, scale it and analyze its distribution.

# %%
"""
CELL N°1.1 : IMPORTING ALL THE NECESSARY PACKAGES

@pre:  /
@post: The necessary packages should be loaded.
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
# Import all the necessary packages here...

# %%
"""
CELL N°1.2 : IMPORTING THE DATASET

@pre:  /
@post: The object `df` should contain a Pandas DataFrame corresponding to the file `diabetes_dataset.csv`
"""

trh = 0.5
n_n = 10

df = pd.read_csv('diabetes_dataset.csv',sep =',') # To modify

df.info()
df.describe()

# %% [markdown]
# ***Is the dataset balanced?***
# 
# It's good practice to check this to better understand the contents of our dataset. The balance between the different classes has an impact on the binarization threshold (which is initialized here at 0.5). Other things can also have an impact on the choice of threshold.

# %%
"""
CELL N°1.3 : IS THE DATASET BALANCED?

@pre:  `df` contains the dataset
@post: Plot the diabetic/non-diabetic distribution in a pie chart
"""

diabetes_counts = df['Diabetes'].value_counts()
print(diabetes_counts)
plt.figure(figsize=(6,6))
labels = ['Non-Diabetic', 'Diabetic']
plt.pie(diabetes_counts, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Diabetic and Non-Diabetic Patients')
plt.show()




# Here! 

# %% [markdown]
# ***Standardize*** is important when you work with data because it allows data to be compared with one another. 
# 
# $z$ is the standard score of a population $x$. It can be computed as follows:
# $$z = \frac{x-\mu}{\sigma}$$
# with $\mu$ the mean of the population and $\sigma$ the standard deviation of the poplutation.
# 
# Please consult, [Wikipedia](https://en.wikipedia.org/wiki/Standard_score) for further information about the standardization.\
# Be careful to use the same formula as us, check in `scikit-learn`
# 

# %%
def scale_dataset(df): 
  
    df_scaled = df.copy()
    
    
    features = df.columns.drop('Diabetes')
    
   
    scaler = StandardScaler()
    
    df_scaled[features] = scaler.fit_transform(df[features])
    
    return df_scaled


df = scale_dataset(df)

#
df.info()
df.describe()


# %% [markdown]
# <br><font size=7 color=#009999> <b>PART II - Correlation</b> </font> <br><br>

# %% [markdown]
# ***In order to keep*** the important features for our classification, we can compute and plot (see e.g. `seaborn.heatmap`) the correlation matrix. With these correlation coefficient, we can establish a feature selection strategy.\
# Be sure to use the `pearson` correlation.
# 

# %%
"""
CELL N°2.1 : CORRELATION MATRIX

@pre:  `df` contains the diabetes dataset
@post: `corr_matrix` is a Pandas DataFrame that contains the correlation matrix of the full dataset
"""


corr_matrix = df.corr(method='pearson')

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix of Diabetes Dataset')
plt.show()

# %% [markdown]
# After this visualization, it is time to sort the coefficients of correlation to keep them with the best correlation with `Diabetes`. **Be careful** with the sign.

# %%
"""
CELL N°2.2 : ANALYZE THE CORRELATION WITH DIABETE

@pre:  `corr_matrix` is a Pandas DataFrame that contains the correlation matrix of the training set
@post: `sorted_features` contains a list of features (columns of `df`) 
       sorted according to their correlation with `Diabetes` 
"""



corr_matrix = df.corr(method='pearson')

def sort_features(corr_matrix):
    """
    Retourne une liste des features triées par corrélation absolue avec 'Diabetes'.
    """
    corr_with_diabetes = corr_matrix['Diabetes'].drop('Diabetes')  # Exclut la colonne 'Diabetes'
    sorted_features = corr_with_diabetes.abs().sort_values(ascending=False).index.tolist()
    return sorted_features




sorted_features = sort_features(corr_matrix)

print("Features sorted by their absolute correlation with Diabetes (including sign):")
print(type(sorted_features))  # Doit afficher "<class 'list'>"


# Optional: Visualize the correlations
#plt.figure(figsize=(10, 6))
#corr_with_diabetes = corr_matrix['Diabetes'].drop('Diabetes')
#corr_with_diabetes_sorted = corr_with_diabetes.loc[sorted_features]

#corr_with_diabetes_sorted.plot(kind='bar', color=['green' if val >= 0 else 'red' for val in corr_with_diabetes_sorted])
#plt.title('Correlation of Features with Diabetes')
#plt.ylabel('Correlation Coefficient')
#plt.xlabel('Features')
#plt.xticks(rotation=45)
#plt.tight_layout()
#plt.show()

"""Features sorted by their absolute correlation with Diabetes (including sign):
                     Feature  Correlation
GenHlth              GenHlth     0.407612
HighBP                HighBP     0.381516
BMI                      BMI     0.293373
HighChol            HighChol     0.289213
Age                      Age     0.278738
DiffWalk            DiffWalk     0.272646
Income                Income    -0.224449
PhysHlth            PhysHlth     0.213081
HeartDisease    HeartDisease     0.211523
Education          Education    -0.170481
PhysActivity    PhysActivity    -0.158666
Stroke                Stroke     0.125427
CholCheck          CholCheck     0.115382
Alcohol              Alcohol    -0.094853
MentHlth            MentHlth     0.087029
Smoker                Smoker     0.085999
Veggies              Veggies    -0.079293
Fruits                Fruits    -0.054077
Sex                      Sex     0.044413
NoDocbcCost      NoDocbcCost     0.040977
AnyHealthcare  AnyHealthcare     0.023191
"""




# %% [markdown]
# <br><font size=7 color=#009999> <b>PART III - Classifiers</b> </font> <br><br>

# %% [markdown]
# In this third part, you need to write functions that return a lamba function with a classifier for the test set. **Be careful** to keep the same form as the one suggested to pass the sanity checks.

# %% [markdown]
# **Implement** the *linear_regressor*. Please follow the specifications in the provided template.
# 
# **Reminder:** Linear regressor is a model that predicts a continuous value by fitting a line (or hyperplane) to the data, minimizing the difference between observed and predicted values.
# 
# Please consult, [Wikipedia](https://en.wikipedia.org/wiki/Linear_regression) for further information about the classifier.

# %%
"""
CELL N°3.1 : LINEAR REGRESSOR

@pre:  `X_train` and `y_train` contain the training set of `df` and a threshold that is a numerical value (float) by default 0.5.
@post: Lambda function that takes `X_test` and returns a 1D array of binary predictions (0 or 1) according to the given threshold.
"""


def linear_regressor(X_train, y_train, threshold=0.5):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return lambda X_test: (model.predict(X_test) >= threshold).astype(int)



# %% [markdown]
# **Implement** the *logistic_regressor*. Please follow the specifications in the provided template.
# 
# **Reminder:** Logisitic regressor is a classification model that estimates the probability of an observation belonging to a class using a logistic function; suitable for binary (and multiclass) problems.
# 
# Please consult, [Wikipedia](https://en.wikipedia.org/wiki/Linear_regression) for further information about the classifier.

# %%
"""
CELL N°3.2 : LOGISTIC REGRESSOR

@pre:  `X_train` and `y_train` contain the training set of `df` and a threshold that is a numerical value (float) by default 0.5.
@post:  Lambda function that takes `X_test` and returns a 1D array of binary predictions (0 or 1) according to the given threshold.
"""



def logistic_regressor(X_train, y_train, threshold=0.5):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return lambda X_test: (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)


# %% [markdown]
# **Implement** the *knn_regressor*. Please follow the specifications in the provided template.  <br>
# 
# **Reminder:** Knn regressor is a non-parametric classification algorithm that classifies an observation according to the classes of its k nearest neighbors in feature space.
# 
# Please consult, [Wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) for further information about the classifier.\
# Attention, you must implement it with **Euclidian distance** and **10** neighbors.

# %%
"""
CELL N°3.3 : KNN REGRESSOR

@pre:  `X_train` and `y_train` contain the training set of `df` and a threshold that is a numerical value (float) by default 0.5.
@post: Lambda function that takes `X_test` and returns a 1D array of binary predictions (0 or 1) according to the given threshold.
"""



def knn_regressor(X_train, y_train, threshold=0.5, n_neighbors=10):
    model = KNeighborsRegressor(n_neighbors=n_neighbors, metric='euclidean')
    model.fit(X_train, y_train)
    return lambda X_test: (model.predict(X_test) >= threshold).astype(int)


# %% [markdown]
# <br><font size=7 color=#009999> <b>PART IV - Validation metrics</b> </font> <br><br>

# %% [markdown]
# In this part, we will implement tools that will help us to **validate** the prediction models implemented in Part III. In particular, we will use the _precision, recall_ and _F1 score_ metrics. 
# 
# **Implement** the _precision, recall_ and _F1 score_. Please follow the specifications in the provided template.  <br>

# %% [markdown]
# **Reminder**
# 
# $F_1$ is a performance metric allowing to obtain some trade-off between the precision and recall criterions. It can be computed as follows:
# $$F_1 = 2~\frac{\mathrm{precision} \cdot \mathrm{recall}}{\mathrm{precision} + \mathrm{recall}}.$$
# 
# Please consult, [Wikipedia](https://en.wikipedia.org/wiki/F-score) for further information about the three metrics.

# %%
"""
CELL N°4.1 : PRECISION SCORE

@pre:  /
@post: `precision(y_test, y_pred)` returns the prediction metric based on the predicted labels `y_pred`
       and the true labels `y_test`. 
    
       
       Precision = TP / (TP + FP)
"""

def precision(y_test, y_pred):
    import numpy as np
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    true_positive = np.sum((y_pred == 1) & (y_test == 1))
    false_positive = np.sum((y_pred == 1) & (y_test == 0))
    if true_positive + false_positive == 0:
        return 0.0
    precision_score = true_positive / (true_positive + false_positive)
    return precision_score

# %%
"""
CELL N°4.2 : RECALL SCORE

@pre:  /
@post: `recall(y_test, y_pred)` returns the recall metric based on the predicted labels `y_pred`
       and the true labels `y_test`. 

        Recall = TP / (TP + FN)


"""

def recall(y_test, y_pred):
    import numpy as np
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    true_positive = np.sum((y_pred == 1) & (y_test == 1))
    false_negative = np.sum((y_pred == 0) & (y_test == 1))
    if true_positive + false_negative == 0:
        return 0.0
    recall_score = true_positive / (true_positive + false_negative)
    return recall_score

# %%
"""
CELL N°4.3 : F1 SCORE

@pre:  /
@post: `f1_score(y_test, y_pred)` returns the F1 score metric based on the predicted labels `y_pred`
       and the true labels `y_test`. 

        F1_score = 2 * ((precison*recall)/(precision+recall) 
       
"""

def f1_score(y_test, y_pred):
    p = precision(y_test, y_pred)
    r = recall(y_test, y_pred)
    if p + r == 0:
        return 0.0
    f1 = 2 * p * r / (p + r)
    return f1

# %% [markdown]
# <br><font size=7 color=#009999> <b>PART V - Reduce the questionnaire size</b> </font> <br><br>

# %% [markdown]
# In this part, find a model that satisfies the following specifications: 
# - A recall of at least 95%
# - A F1 score of at least 75%
# 
# For that, we will use **k-fold** cross validation (see [the Wikipedia page](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation) for a reminder), and then test the three models above with 
# - Different number of features
# - Different thresholds

# %% [markdown]
# In order to use k-fold cross validation, use the class `sklearn.model_selection.KFold` from the `scikit-learn` library (see [the documentation](https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.KFold.html)) with `n_splits = 3`.
# 
# <div class="alert alert-danger">
# <b>[IMPORTANT] Grading</b>  <br>
# In order for us to be able to automatically grade your submission, put <code>shuffle=True</code>, and <code>random_state=1109</code> when you initialize <code>KFold</code>.
# </div>

# %%

#comme dans le tp2 


kf = KFold(n_splits=3, shuffle=True, random_state=1109)

X = df.drop('Diabetes', axis=1)
y = df['Diabetes']

best_model_name = ''
best_features = []
best_threshold = 0.5
best_recall = 0
best_f1 = 0


thresholds = [i / 10 for i in range(1, 10)]  # De 0.1 à 0.9
num_features_list = range(1, len(sorted_features) + 1)

models = {
    'linear': linear_regressor,
    'logistic': logistic_regressor,
    'knn': knn_regressor
}

# Run the thresholds
for threshold in thresholds:
    print(f"\nTesting with threshold: {threshold}")
    for num_features in num_features_list:
        selected_features = sorted_features[:num_features]
        X_selected = X[selected_features]
        
        reg_validation = {
            'linear': [],
            'logistic': [],
            'knn': []
        }
        
        # CV : Cross Validation 
        for train_index, test_index in kf.split(X_selected):
            X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Standardisation des features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            y_train_array = y_train.values.ravel()
            y_test_array = y_test.values.ravel()
            
            # Linear Regressor
            linear_predictor = linear_regressor(X_train_scaled, y_train_array, threshold=threshold)
            y_pred_linear = linear_predictor(X_test_scaled)
            linear_metrics = (recall(y_test_array, y_pred_linear), precision(y_test_array, y_pred_linear), f1_score(y_test_array, y_pred_linear))
            reg_validation['linear'].append(linear_metrics)
            
            # Logistic Regressor
            logistic_predictor = logistic_regressor(X_train_scaled, y_train_array, threshold=threshold)
            y_pred_logistic = logistic_predictor(X_test_scaled)
            logistic_metrics = (recall(y_test_array, y_pred_logistic), precision(y_test_array, y_pred_logistic), f1_score(y_test_array, y_pred_logistic))
            reg_validation['logistic'].append(logistic_metrics)
            
            # KNN Regressor
            knn_predictor = knn_regressor(X_train_scaled, y_train_array, threshold=threshold, n_neighbors=10)
            y_pred_knn = knn_predictor(X_test_scaled)
            knn_metrics = (recall(y_test_array, y_pred_knn), precision(y_test_array, y_pred_knn), f1_score(y_test_array, y_pred_knn))
            reg_validation['knn'].append(knn_metrics)
        
        
        for reg in ['linear', 'logistic', 'knn']:
            metrics_array = np.array(reg_validation[reg])
            avg_metrics = tuple(metrics_array.mean(axis=0))
            avg_recall, avg_precision, avg_f1 = avg_metrics
            if avg_recall >= 0.95 and avg_f1 >= 0.75:
                print(f"\nModel: {reg.capitalize()} Regressor")
                print(f"Number of features: {num_features}")
                print(f"Threshold: {threshold}")
                print(f"Selected features: {selected_features}")
                print(f"Average Recall: {avg_recall:.4f}")
                print(f"Average Precision: {avg_precision:.4f}")
                print(f"Average F1 Score: {avg_f1:.4f}")
                # Update the best model
                if avg_f1 > best_f1:
                    best_model_name = reg
                    best_features = selected_features
                    best_threshold = threshold
                    best_recall = avg_recall
                    best_f1 = avg_f1

# Display the best model
if best_f1 > 0:
    print(f"\nBest model found:")
    print(f"Model: {best_model_name.capitalize()} Regressor")
    print(f"Number of features: {len(best_features)}")
    print(f"Threshold: {best_threshold}")
    print(f"Selected features: {best_features}\n")
    print(f"Average Recall: {best_recall*100:.4f}%")
    print(f"Average F1 Score: {best_f1*100:.4f}%")
else:
    print("No model found that satisfies the specifications.")


# %% [markdown]
# In order to find our model, proceed as follows: 
# - Fix a threshold in $(0, 1)$
# - Define a dictionary `result` of the form 
# 
# <code>result = {
#     "linear": {}, 
#     "logistic": {}, 
#     "knn": {}
# }
# </code>
# 
# - For $ i \in \{1, \dots, \texttt{N\_features}\} $: 
#     - Select the $i$ **most correlated features** (use `sorted_features` defined above)
#     - For all the pairs $((X_{\text{train}}, y_{\text{train}}), (X_{\text{test}}, y_{\text{test}}))$ given by k-fold
#         - Compute the linear, logistic and KNN regressors with the fixed threshold on $X_{\text{train}}$
#         - Compute the 3 different 3-tuple `validation(regressor, X_test, y_test)`
#     - In `result[reg][i]`, save the **average** of all the validation tuples you computed for `reg`

# %%
"""
CELL N°5.2 : FIND THE RIGHT COMBINATION LENGTH/REGRESSOR

@pre:  `kf`, `X`, `y`, `sorted_features`, et `models` sont définis comme dans la CELL N°5.1.
@post: `result` est tel que `result[reg][i]` contient la moyenne des validations pour le régressieur `reg`, 
       en conservant les `i` features les plus corrélées.
"""

def validation(regressor, X_test, y_test):
    y_pred = regressor(X_test)
    return (recall(y_test, y_pred), precision(y_test, y_pred), f1_score(y_test, y_pred))


threshold = 0.5 # we can adapt the value

result = {
    "linear": {},
    "logistic": {},
    "knn": {}
}


for i in range(1, len(sorted_features) + 1):
 
    selected_features = sorted_features[:i]
    X_selected = X[selected_features]
    
    #set to get the measures for the validation
    reg_validation = {
        "linear": [],
        "logistic": [],
        "knn": []
    }
    
    #CV = Cross Validation 
    # cell 5.1
    for train_index, test_index in kf.split(X_selected):
      
        X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
    
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
    
        y_train_array = y_train.values.ravel()
        y_test_array = y_test.values.ravel()
    
        for reg_name, reg_function in models.items():
            
            if reg_name == 'knn':
                predictor = reg_function(X_train_scaled, y_train_array, threshold=threshold, n_neighbors=10)
            else:
                predictor = reg_function(X_train_scaled, y_train_array, threshold=threshold)
            
            metrics = validation(predictor, X_test_scaled, y_test_array)
            reg_validation[reg_name].append(metrics)
    

    for reg in ["linear", "logistic", "knn"]:
     
        metrics_array = np.array(reg_validation[reg])
        avg_metrics = tuple(metrics_array.mean(axis=0))

        result[reg][i] = avg_metrics



# Display the result
# we display the correct regressor 
for num_features, metrics in result['logistic'].items():
    avg_recall, avg_precision, avg_f1 = metrics
    print(f"Nombre de features : {num_features}")
    print(f"Rappel moyen : {avg_recall:.4f}")
    print(f"Précision moyenne : {avg_precision:.4f}")
    print(f"Score F1 moyen : {avg_f1:.4f}\n")



# %% [markdown]
# The following cell allows you to test if the threshold that you chose satisfies the specifications, that are 
# - A recall of at least 95%
# - A F1 score of at least 75%
# 
# Plot these graphs for different threshold, and select the model **with the smallest number of questions** that satisfy the conditions above.

# %%
"""
CELL N°5.3 : VISUALIZE THE SCORES

@pre:  `result` contains the average of the validations for regressor `reg`, when keeping the `i` most correlated features
@post: plot of the scores for each condition
"""

# Nothing to do here, just run me! 

from helper import plot_result
plot_result(result, threshold, to_show = "recall")
plot_result(result, threshold, to_show = "f1_score")

# %% [markdown]
# <br><font size=7 color=#009999> <b>PART VI - Visualization</b> </font> <br><br>
# 

# %% [markdown]
# In this part, you are asked to produce a **clear and clean figure** expressing a result
# or giving an overall vision of your work for this hackaton. **Please feel free to do as you
# wish. Be original!** 
# 
# The **clarity**, **content** and **description** (in the report) of your figure will be evaluated.

# %%
"""
CELL N°6.1 : VISUALIZE YOUR RESULTS

@pre:  /
@post: A figure showing the Recall and F1 Score of the Logistic Regressor at different thresholds.
"""




thresholds = [i / 10 for i in range(1, 10)]  # Thresholds from 0.1 to 0.9
recalls = []
f1_scores = []


for threshold in thresholds:
    reg_validation = []
    selected_features = sorted_features  # Using all selected features
    X_selected = X[selected_features]
    for train_index, test_index in kf.split(X_selected):
        X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        y_train_array = y_train.values.ravel()
        y_test_array = y_test.values.ravel()
        logistic_predictor = logistic_regressor(X_train_scaled, y_train_array, threshold=threshold)
        y_pred_logistic = logistic_predictor(X_test_scaled)
        logistic_metrics = (
            recall(y_test_array, y_pred_logistic),
            precision(y_test_array, y_pred_logistic),
            f1_score(y_test_array, y_pred_logistic)
        )
        reg_validation.append(logistic_metrics)
    # Compute the average metrics
    metrics_array = np.array(reg_validation)
    avg_metrics = tuple(metrics_array.mean(axis=0))
    avg_recall, avg_precision, avg_f1 = avg_metrics

    recalls.append(avg_recall)
    f1_scores.append(avg_f1)

# Plotting the Recall and F1 Score vs Threshold
plt.figure(figsize=(10,6))
plt.plot(thresholds, recalls, marker='o', label='Recall')
plt.plot(thresholds, f1_scores, marker='s', label='F1 Score')
plt.axhline(y=0.95, color='r', linestyle='--', label='Recall = 0.95')
plt.axhline(y=0.75, color='g', linestyle='--', label='F1 Score = 0.75')
plt.title('Recall and F1 Score vs Threshold for Logistic Regressor')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.xticks(thresholds)
plt.legend()
plt.grid(True)
plt.show()



