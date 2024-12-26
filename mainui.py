from numpy.core.fromnumeric import var
import streamlit
import streamlit as st
import sys
from streamlit.web import cli as stcli
from PIL import Image
from functions import *
import streamlit.components.v1 as components
import pandas as pd
from st_clickable_images import clickable_images
import numpy as np
import statsmodels.api as sm
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import seaborn as sns
from io import BytesIO
from statsmodels.formula.api import ols
# from streamlit.state.session_state import SessionState
import tkinter
import matplotlib
# matplotlib.use('TkAgg')
# matplotlib.use('Agg')
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.tree import DecisionTreeRegressor, plot_tree
import sklearn
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
import time
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import dtale
from dtale.views import startup
from dtale.app import get_instance
import webbrowser
import dtale.global_state as global_state
import dtale.app as dtale_app

def abiui():

    # if 'model' not in st.session_state:
    #     st.session_state.model = "None"

    if 'key' not in st.session_state:
        st.session_state['pcasession'] = 'False'

    header1 = st.container()
    header2 = st.container()
    guidance = st.container()
    dataset = st.container()
    infosection = st.container()
    model_training = st.container()

    with header1:
            clm1, clm2, clm3, clm4, clm5 = st.columns(5)
            with clm1:
                pass
            with clm2:
                pass
            with clm3:
                pass
            with clm4:
                pass
            with clm5:
                pass

    with header2:

        col1,col2,col3 = st.columns(3)
        with col1:
            pass
        with col2:
            st.markdown("<h1 style='text-align: center; color: red;'>"'How to use the UI'"</h1>", unsafe_allow_html=True)
        with col3:
            pass

    with guidance:

        col1,col2,col3 = st.columns(3)
        with col1:
            pass
        with col2:
            st.markdown("<h3 style='text-align: center; color: green;'>"'Press the button below to automate the experience'"</h3>", unsafe_allow_html=True)
        with col3:
            pass

        clicked = clickable_images(
        [
            "https://i.postimg.cc/158HHcRn/Automate.png", # Link for links to picture: https://postimg.cc/7fy0Pz2w/b5680871
        ],
        titles=[f"Image #{str(i)}" for i in range(1)], # Change range if more pictures are needed
        div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
        img_style={"margin": "5px", "height": "200px"},
    )

        # st.markdown(f"Image #{clicked} clicked" if clicked > -1 else "")

        if clicked == 0:
            st.success("We will automate the experience for you")

        st.button("AutoML - choose the best fit model for your analysis using FLAML")

    with dataset:
        # components.html('''<script src="//ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js"></script>
        # <script src="jquery.zoomooz.min.js"></script>''')
        # st.markdown('''<div class="zoomTarget" data-targetsize="0.45" data-duration="600">This element zooms when clicked on.</div>''', unsafe_allow_html=True)

        ### From Jupyter - 0. Prepare the data

        # Read data
        df = pd.read_csv('allrecordsohe.csv', low_memory=False)
        df2 = pd.read_csv('allrecords.csv', low_memory=False)

        # Check for empty data
        df.isnull().sum()
        df2.isnull().sum()

        # Remove NaN
        nr_samples_before = df.shape[0]
        df = df.fillna(0)
        print('Removed %s samples' % (nr_samples_before - df.shape[0]))
        nr_samples_before = df2.shape[0]
        df2 = df2.fillna(0)
        print('Removed %s samples' % (nr_samples_before - df2.shape[0]))

        # Drop irrelevant variables
        df.drop(['TD_ID', 'KRUX_ID', 'TAP_IT_ID', 'GOOGLE_CLIENT_ID'], axis=1, inplace=True)
        df2.drop(['TD_ID', 'KRUX_ID', 'TAP_IT_ID', 'GOOGLE_CLIENT_ID'], axis=1, inplace=True)

        ### End

        vars = list(df2.columns.values.tolist())
        variables = st.button("List of available variables for analysis")
        if variables == True:
            st.write(vars)

    with infosection:
        st.header("Main analytical playground")
        st.write("Set modelling parameters here")
        # st.header("How to view and edit data")
        # viewedit = st.expander("More information on data tool", expanded=False)
        # with viewedit:
        #     st.write("Use PGUI")

    with model_training:
        st.subheader("Choose and train model to your preferences")
        modelchoice = st.selectbox("Please choose preferred model", ["Please select model", "Exploratory data analysis", "Hypothesis testing: correlation", "Linear regression", "Logistic regression", "Cluster analysis", "ANOVA", "Principal component analysis", "Conjoint analysis", "Neural networks", "Decision trees", "Ensemble methods - random forest"], key="modelchoice")

        if modelchoice == "Exploratory data analysis":

            # Pandas profiling, Autoviz, Dtale

            if get_instance("1") is None:
                startup(data_id="1", data=df)

            d=get_instance("1")

            checkbtn = st.button("Validate data")

            if checkbtn != True:
            
               webbrowser.open_new_tab('http://localhost:8501/dtale/main/1')

            if checkbtn == True:
                df_amended = get_instance(data_id="1").data # The amended dataframe ready for upsert
                st.write("Sample of amended data:")
                st.write("")
                st.write(df_amended.head(5))

            clearchanges = st.button("Clear changes made to data")
            if clearchanges == True:
                global_state.cleanup()

        if modelchoice == "Hypothesis testing: correlation":

            with st.spinner('Please wait while we conduct the correlation analysis'):

                my_bar = st.progress(0)

                time.sleep(10)

                for percent_complete in range(100):
                    time.sleep(0.1)
                    my_bar.progress(percent_complete + 1)

                start_time = time.time()

                x=0
                y=0

                x = st.selectbox("Please choose first variable:", vars, key="correlx", index=x)
                y = st.selectbox("Please choose second variable:", vars, key="correly", index=y)

                # var1 = df[x].iloc[1:101]
                # var2 = df[y].iloc[1:101]

                var1 = df[x].sample(10000)
                var2 = df[y].sample(10000)

                st.subheader("Correlation between chosen 2 variables:" + " " + x + " and " + y)

                st.write("Correlation coefficient: ", var1.corr(var2))
                st.write("Pearson correlation coefficient: ", pearsonr(var1, var2))
                st.write("Spearman correlation coefficient: ", spearmanr(var1, var2))

                ### From Jupyter: 1. Hypothesis testing: correlation

                st.subheader("Correlation matrix, Pearson")

                st.write(df.corr(method = 'pearson'))

                ### End

                st.write("")

                st.write("Hypothesis testing through correlation analysis took ", time.time() - start_time, "seconds to run")

        if modelchoice == "Linear regression":

            with st.spinner('Please wait while we conduct the linear regression analysis'):

                my_bar = st.progress(0)

                time.sleep(5)

                for percent_complete in range(100):
                    time.sleep(0.2)
                    my_bar.progress(percent_complete + 1)

                start_time = time.time()

                ### From Jupyter - 2. Linear regression

                # Choose predicted variable - this will become dynamic in the app
                y = df['BRAND']

                # Define predictor variables
                x = df.iloc[:, 1:-1]

                x, y = np.array(x), np.array(y)

                x = sm.add_constant(x)

                model = sm.OLS(y, x)

                results = model.fit()

                st.write(results.summary())

                st.write("")

                st.write('\nPredicted response:', results.fittedvalues, sep='\n') # Or print('predicted response:', results.predict(x), sep='\n')

                ### End

                st.write("")

                st.write("Linear regression took ", time.time() - start_time, "seconds to run")

        if modelchoice == "Logistic regression":

            with st.spinner('Please wait while we conduct the multinomial logistic regression analysis'):

                my_bar = st.progress(0)

                time.sleep(10)

                for percent_complete in range(100):
                    time.sleep(0.1)
                    my_bar.progress(percent_complete + 1)

                start_time = time.time()

                ### From Jupyter - 3. Multinomial logistic regression

                # Evaluate multinomial logistic regression model

                # Define dataset

                # y = df['BRAND'].iloc[0:10000]
                # X = df.iloc[0:10000, 1:-1] #subsamping for efficiency and speed
                y = df['BRAND'].sample(10000)
                X = df.sample(10000) #subsamping for efficiency and speed
                varnames = df.columns.values.tolist()
                X, y = np.array(X), np.array(y)

                # Define the multinomial logistic regression model
                model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

                # Define the model evaluation procedure
                cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=3, random_state=1)

                # Evaluate the model and collect the scores
                n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
                # report the model performance
                st.write('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

                # make a prediction with a multinomial logistic regression model

                # define dataset
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

                # define the multinomial logistic regression model
                model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

                # fit the model on the training dataset
                model.fit(X_train, y_train)

                # define a single row of test data
                row = X_test[0,0:]

                # predict the class label
                yhat = model.predict([row])
                # summarize the predicted class
                st.write('Predicted Class: %d' % yhat[0])

                # predict a multinomial probability distribution
                yhat = model.predict_proba([row])
                # summarize the predicted probabilities
                # st.write('Predicted Probabilities: %s' % yhat[0])
                st.write('Predicted probabilities:')
                # data = {
                # "Variable": varnames,
                # "Pred Prob": yhat[0],
                # }
                # st.write(pd.DataFrame(data))
                #st.write(varnames)
                st.write(yhat[0].tolist())
                # plt.figure(figsize=(7, 3))
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.plot(yhat[0].tolist())
                buf = BytesIO()
                fig.savefig(buf, format="png")
                # st.pyplot(fig)
                st.image(buf)

                # Tune penalty hyperparameter (next cell)

                ### End

                st.write("")

                st.write("Logistic regression took ", time.time() - start_time, "seconds to run")

                st.write("")

            tunelogreg = st.button("Tune penalty hyperparameter")

            if tunelogreg == True:

                with st.spinner('Please wait while we tune the hyperparameters for the logistic regression analysis'):

                    my_bar = st.progress(0)

                    time.sleep(10)

                    for percent_complete in range(100):
                        time.sleep(0.01)
                        my_bar.progress(percent_complete + 1)

                    start_time = time.time()

                    ### From Jupyter - tune multinomial logistic regression hyperparameters

                    # define the multinomial logistic regression model with a default penalty
                    LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', C=1.0)
                    # get a list of models to evaluate
                    def get_models():
                        models = dict()
                        for p in [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]:
                            # create name for model
                            key = '%.4f' % p
                            # turn off penalty in some cases
                            if p == 0.0:
                                # no penalty in this case
                                models[key] = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='none')
                            else:
                                models[key] = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', C=p)
                        return models
                    # evaluate a give model using cross-validation
                    def evaluate_model(model, X, y):
                        # define the evaluation procedure
                        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
                        # evaluate the model
                        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
                        return scores
                    # get the models to evaluate
                    models = get_models()
                    # evaluate the models and store results
                    results, names = list(), list()
                    for name, model in models.items():
                        # evaluate the model and collect the scores
                        scores = evaluate_model(model, X, y)
                        # store the results
                        results.append(scores)
                        names.append(name)
                        # summarize progress along the way
                        st.write('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
                    # plot model performance for comparison
                    fig, ax = plt.subplots(figsize=(10, 6))
                    pyplot.boxplot(results, labels=names, showmeans=True)
                    buf = BytesIO()
                    fig.savefig(buf, format="png")
                    # st.pyplot(fig)
                    st.image(buf)

                    st.write("Smaller C = larger penalty; our results show we need a larger penalty for better model performance")

                    ### End

                    st.write("")

                    st.write("Tuning hyperparameters for the logistic regression took ", time.time() - start_time, "seconds to run")

        if modelchoice == "Cluster analysis":

            with st.spinner('Please wait while we conduct the cluster analysis using the K-means algorithm'):

                my_bar = st.progress(0)

                time.sleep(10)

                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1)

                start_time = time.time()

                ### From Jupyter - 4. Clustering - K-means

                # Perform K-means clustering for consumer segmentation

                # Size of the data set after removing NaN
                print(df2.shape)
                # Explore type of data and feature names
                #print(df2.sample(5))

                # Select continuous variables for clustering - this is for age
                # X2 = df2.iloc[0:10000, 14:15] #subsamping for efficiency and speed
                X2 = df.iloc[:, 10:11].sample(10000) #subsamping for efficiency and speed

                # Find optimal number of clusters

                # 1. Elbow method
                # Calculate distortions
                distortions = []

                for i in range(1, 16):
                    km = KMeans(n_clusters=i, init='k-means++', n_init=10, 
                                max_iter=300,tol=1e-04, random_state=0)
                    km.fit(X2)
                    distortions.append(sum(np.min(cdist(X2, km.cluster_centers_, 
                                    'euclidean'),axis=1)) / X2.shape[0])

                # Plot distortions
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.plot(range(1, 16), distortions, marker='o')
                plt.xlabel('Number of clusters')
                plt.ylabel('Distortion')
                buf = BytesIO()
                fig.savefig(buf, format="png")
                # st.pyplot(fig)
                st.image(buf)

                # 2. Silhouette method
                sil = []
                kmax = 10

                for k in range(2, kmax+1):
                    kmeans = KMeans(n_clusters = k).fit(X2)
                    labels = kmeans.labels_
                    sil.append(silhouette_score(X2, labels, metric = 'euclidean'))

                # Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.plot(range(2, kmax+1), sil, marker='o')
                plt.xlabel('Number of clusters')
                plt.ylabel('Silhouette score')
                buf = BytesIO()
                fig.savefig(buf, format="png")
                # st.pyplot(fig)
                st.image(buf)
                # Use the output from the elbow or silhouette method to decide how many clusters to use.
                # Cluster the data

                km = KMeans(n_clusters=3, init='k-means++', 
                            n_init=10, max_iter=300, 
                            tol=1e-04, random_state=0)
                km.fit(X2)

                # Re-cluster with 4 - this will become dynamic in the app

                X2new = X2.copy()
                kmnew = KMeans(n_clusters=4, init='k-means++', 
                            n_init=10, max_iter=300, 
                            tol=1e-04, random_state=0)
                kmnew.fit(X2new) 
                # Check how many observations are in each cluster

                # print("Cluster 0 size: %s \nCluster 1 size: %s"
                #       % (len(km.labels_)- km.labels_.sum(), km.labels_.sum()))
                # Check cluster size once re-clustered

                print(kmnew.labels_)

                Xnew2 = X2.copy()
                Xnew2["CLUSTERS"] = kmnew.labels_
                Xnew2.sample(8, random_state=0)

                countclusters = Xnew2.apply(lambda x: True if x['CLUSTERS'] == 0 else False , axis=1)
                # Count number of True in series
                numOfRows = len(countclusters[countclusters == True].index)
                st.write('Number of Rows in dataframe in which cluster = 0 : ', numOfRows)

                countclusters = Xnew2.apply(lambda x: True if x['CLUSTERS'] == 1 else False , axis=1)
                # Count number of True in series
                numOfRows = len(countclusters[countclusters == True].index)
                st.write('Number of Rows in dataframe in which cluster = 1 : ', numOfRows)

                countclusters = Xnew2.apply(lambda x: True if x['CLUSTERS'] == 2 else False , axis=1)
                # Count number of True in series
                numOfRows = len(countclusters[countclusters == True].index)
                st.write('Number of Rows in dataframe in which cluster = 2 : ', numOfRows)
                # Set up a dataframe with cluster allocations

                countclusters = Xnew2.apply(lambda x: True if x['CLUSTERS'] == 3 else False , axis=1)
                # Count number of True in series
                numOfRows = len(countclusters[countclusters == True].index)
                st.write('Number of Rows in dataframe in which cluster = 3 : ', numOfRows)
                # Set up a dataframe with cluster allocations

                Xnew = X2.copy()
                Xnew["CLUSTERS"] = km.labels_
                Xnew.sample(8, random_state=0)

                # Plot the following variables and their clusters
                var = ['AGE']

                # Plot using seaborn

                # fig, ax = plt.subplots(figsize=(7, 3))
                fig = sns.pairplot(Xnew2, vars=var, hue="CLUSTERS", palette=sns.color_palette("hls", 4), height=5)
                buf = BytesIO()
                fig.savefig(buf, format="png")
                #st.pyplot(fig)
                st.image(buf)
                #df.head()
                #df = df.dropna()

                # Optional - log transformations

                ### End

                st.write("")

                st.write("Running the cluster analysis using K-means took ", time.time() - start_time, "seconds to run")

        if modelchoice == "ANOVA":

            with st.spinner('Please wait while we run the ANOVA analysis'):

                my_bar = st.progress(0)

                time.sleep(10)

                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1)

                start_time = time.time()

                ### From Jupyter - 5. ANOVA

                data = df2

                #Create a boxplot
                fig = data.boxplot('AGE', by='BRAND', figsize=(18, 18)) # Dynamic to variables of choice
                fig.set_xticklabels(fig.get_xticklabels(),rotation=90)

                ctrl = data['AGE'][data.BRAND == 'Castle Lager'] # Dynamic to brand and variables of choice

                grps = pd.unique(data.BRAND.values)
                d_data = {grp:data['AGE'][data.BRAND == grp] for grp in grps}

                k = len(pd.unique(data.BRAND))  # number of conditions
                N = len(data.values)  # conditions times participants
                n = data.groupby('BRAND').size()[0] #Participants in each condition

                buf = BytesIO()
                fig.figure.savefig(buf, format="png")
                st.image(buf)

                # ANOVA using statsmodels

                mod = ols('AGE ~ BRAND',
                                data=data).fit()
                                
                aov_table = sm.stats.anova_lm(mod, typ=2)
                st.table(aov_table)

                # Pairwise comparisons

                pair_t = mod.t_test_pairwise('BRAND') # can add optional method = "sidak" or "bonferroni" here
                st.write(pair_t.result_frame)

                ### End

                st.write("")

                st.write("ANOVA analysis took ", time.time() - start_time, "seconds to run")

        if modelchoice == "Principal component analysis":

            with st.spinner('Please wait while we conduct principal component analysis'):

                my_bar = st.progress(0)

                time.sleep(10)

                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1)

                start_time = time.time()

                ### From Jupyter - Principal component analysis

                # Initially, visualize the important data features

                # Scale the features
                # Separating out the features
                x = df.iloc[:, 1:-1].sample(10000).values #subsampling for efficiency and speed
                # Separating out the target
                y = df.iloc[:,0].sample(10000).values #subsampling for efficiency and speed
                # Standardizing the features
                x = StandardScaler().fit_transform(x)

                # Dimensionality reduction
                from sklearn.decomposition import PCA
                pca = PCA(n_components=10)
                principalComponents = pca.fit_transform(x)
                principalDf = pd.DataFrame(data = principalComponents
                            , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5', 'principal component 6', 'principal component 7', 'principal component 8', 'principal component 9', 'principal component 10'])
                # Concatenate DF across axis 1
                finalDf = pd.concat([principalDf, df['BRAND']], axis = 1)
                st.write("Table of top 10 principal components")
                st.write(finalDf)

                # Plot 2D data
                fig = plt.figure(figsize = (8,8))
                ax = fig.add_subplot(1,1,1) 
                ax.set_xlabel('Principal Component 1', fontsize = 15)
                ax.set_ylabel('Principal Component 2', fontsize = 15)
                ax.set_title('PCA showing top 2 components', fontsize = 20)
                targets = ['BRAND']
                colors = ['r', 'g', 'b']
                for target, color in zip(targets,colors):
                    indicesToKeep = finalDf['BRAND'] == target
                    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                            , finalDf.loc[indicesToKeep, 'principal component 2']
                            , c = color
                            , s = 50)
                    # ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
                    # ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
                    ax.set_xticks([-100, -10, -0.1, 0, 0.1, 1, 10, 100])
                    ax.set_yticks([-100, -10, -0.1, 0, 0.1, 1, 10, 100])
                ax.legend(targets)
                ax.grid()
                buf = BytesIO()
                fig.savefig(buf, format="png")
                st.image(buf)

                # Explain the variance
                st.write("Explained variance from top 10 components:")
                st.write(pca.explained_variance_ratio_)

                ### End

                st.text("") # Spacer

                st.write("")

                st.write("Principal component analysis took ", time.time() - start_time, "seconds to run")

            pca = st.button("Click to see how PCA can speed up machine learning and to run a new regression model")

            if pca == True:

                st.session_state.pcasession = 'True'

                with st.spinner('Please wait while we conduct a new linear regression using the principal components'):

                    my_bar = st.progress(0)

                    time.sleep(10)

                    for percent_complete in range(100):
                        time.sleep(0.01)
                        my_bar.progress(percent_complete + 1)

                    start_time = time.time()

                    ### From Jupter - Principal component analysis continued

                    # Now use PCA to speed up machine learning

                    #from sklearn.model_selection import train_test_split
                    # test_size: what proportion of original data is used for test set
                    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=1/4.0, random_state=0)
                    # Scale the data
                    scaler = StandardScaler()
                    # Fit on training set only
                    scaler.fit(train_X)
                    # Apply transform to both the training set and the test set.
                    train_X = scaler.transform(train_X)
                    test_X = scaler.transform(test_X)
                    # Choose minimum number of principal components such that 95% of the variance is retained
                    from sklearn.decomposition import PCA
                    # Make an instance of the model
                    pca = PCA(.95)
                    # Fit on training set
                    pca.fit(train_X)
                    # Apply the mapping (transformation) to both the training set and the test set
                    train_X = pca.transform(train_X)
                    test_X = pca.transform(test_X)
                    # Apply model of choice, e.g. logistic regression - this will become dynamic in the app; choose model here
                    # Determine number of components
                    st.write("Number of useful components:")
                    st.write(pca.n_components_)
                    # Determine components
                    st.write("Component contributions:")
                    st.write(pca.components_)
                    df3 = pd.DataFrame(pca.components_)
                    # st.table(df3)

                    ### End

                    # tunedreg = st.button("Click to run a regression model with these components") # For brand

                    # if tunedreg == True and st.session_state.pcasession == True:

                    ### From Jupyter - Linear regression

                    # Choose predicted variable - this will become dynamic in the app
                    y = df['BRAND'].sample(7500)

                    print(y.shape)
                    print(train_X.shape)

                    # Define predictor variables
                    x = train_X

                    x, y = np.array(x), np.array(y)

                    x = sm.add_constant(x)

                    model = sm.OLS(y, x)

                    results = model.fit()

                    st.subheader("PCA regression results:")

                    st.write(results.summary())

                    st.write("")

                    st.write('\nPredicted response:', results.fittedvalues, sep='\n') # Or print('predicted response:', results.predict(x), sep='\n')

                    st.write("")

                    st.write("Conducting a new linear regression with principal components took ", time.time() - start_time, "seconds to run")

        if modelchoice == "Conjoint analysis":

            st.warning("This model takes some time to run. Please be patient.")

            with st.spinner('Please wait while we conduct the conjoint analysis'):

                my_bar = st.progress(0)

                time.sleep(10)

                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1)

                start_time = time.time()

                ### From Jupyter - conjoint analysis

                y = df2.BRAND.apply(lambda x: '1' if 'Castle' in x else 0).head(10000) # Dynamic to brand, subsampling for speed and efficiency
                x = df2[[x for x in df2.columns if x != 'BRAND']].head(10000) # and x !="FIRST_INTERACTION" and x !="LAST_INTERACTION" and x != "DAYS_LEFT_TO_ENGAGE" and x != "FIRST_NAME" and x != "BIRTH_DATE" and x != "PURCHASE_DESCRIPTION"]].head(10000) # Subsampling for speed and efficiency

                xdum = pd.get_dummies(x, columns=[c for c in x.columns if c != 'BRAND'])
                xdum.head()

                # st.write(y.head(100))

                plt.style.use('bmh')

                res = sm.OLS(y.astype(float), xdum.astype(float), family=sm.families.Binomial()).fit()
                # st.subheader("OLS regression results showing importance of each type of factor")
                # st.write(res.summary())

                st.subheader("Factor importances:")

                df_res = pd.DataFrame({
                'param_name': res.params.keys()
                , 'param_w': res.params.values
                , 'pval': res.pvalues
                })
                # adding field for absolute of parameters
                df_res['abs_param_w'] = np.abs(df_res['param_w'])
                # marking field is significant under 95% confidence interval
                df_res['is_sig_95'] = (df_res['pval'] < 0.05)
                # constructing color naming for each param
                df_res['c'] = ['blue' if x else 'red' for x in df_res['is_sig_95']]

                # make it sorted by abs of parameter value
                df_res = df_res.sort_values(by='abs_param_w', ascending=True)

                fig, ax = plt.subplots(figsize=(14, 8))
                plt.title('Part Worth')
                pwu = df_res['param_w']
                xbar = np.arange(len(pwu))
                plt.barh(xbar, pwu, color=df_res['c'])
                plt.yticks(xbar, labels=df_res['param_name'])
                # plt.show()

                buf = BytesIO()
                fig.savefig(buf, format="png")
                st.image(buf)

                st.subheader("Absolute and relative/normalized importances:")

                # need to assemble per attribute for every level of that attribute in dicionary
                range_per_feature = dict()
                for key, coeff in res.params.items():
                    sk =  key.split('_')
                    feature = sk[0]
                    if len(sk) == 1:
                        feature = key
                    if feature not in range_per_feature:
                        range_per_feature[feature] = list()
                        
                    range_per_feature[feature].append(coeff)
                # importance per feature is range of coef in a feature
                # while range is simply max(x) - min(x)
                importance_per_feature = {
                    k: max(v) - min(v) for k, v in range_per_feature.items()
                }

                # compute relative importance per feature
                # or normalized feature importance by dividing 
                # sum of importance for all features
                total_feature_importance = sum(importance_per_feature.values())
                relative_importance_per_feature = {
                    k: 100 * round(v/total_feature_importance, 3) for k, v in importance_per_feature.items()
                }

                alt_data = pd.DataFrame(
                    list(importance_per_feature.items()), 
                    columns=['attr', 'importance']
                ).sort_values(by='importance', ascending=False)

                fig, ax = plt.subplots(figsize=(12, 8))
                xbar = np.arange(len(alt_data['attr']))
                plt.title('Importance')
                plt.barh(xbar, alt_data['importance'])
                for i, v in enumerate(alt_data['importance']):
                    ax.text(v , i + .25, '{:.2f}'.format(v))
                plt.ylabel('attributes')
                plt.xlabel('% importance')
                plt.yticks(xbar, alt_data['attr'])
                plt.show()

                buf = BytesIO()
                fig.savefig(buf, format="png")
                st.image(buf)

                alt_data = pd.DataFrame(
                    list(relative_importance_per_feature.items()), 
                    columns=['attr', 'relative_importance (pct)']
                ).sort_values(by='relative_importance (pct)', ascending=False)

                fig, ax = plt.subplots(figsize=(12, 8))
                xbar = np.arange(len(alt_data['attr']))
                plt.title('Relative importance / Normalized importance')
                plt.barh(xbar, alt_data['relative_importance (pct)'])
                for i, v in enumerate(alt_data['relative_importance (pct)']):
                    ax.text(v , i + .25, '{:.2f}%'.format(v))
                plt.ylabel('attributes')
                plt.xlabel('% relative importance')
                plt.yticks(xbar, alt_data['attr'])
                plt.show()

                buf = BytesIO()
                fig.savefig(buf, format="png")
                st.image(buf)

                st.write("")

                st.write("Tuning hyperparameters for the logistic regression took ", time.time() - start_time, "seconds to run")

        if modelchoice == "Neural networks":

            st.warning("This model can take a long time to run. Please be patient.")

            with st.spinner('Please wait while we conduct the neural networks analysis'):

                my_bar = st.progress(0)

                time.sleep(10)

                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1)

                start_time = time.time()

                # print(df2['BRAND'].unique()) # Can do this in a different part of the app (EDA)

                # Split data into features (X) and response (y)
                # Clean up data - can eliminate columns now (features) - not a pruning activity
                X = df.loc[0:10000,["GENDER", "AGE", "CITY"]] # Subsampling for speed and efficiency, dynamic to certain relevant explanatory variables (could combine with dimensionality reduction or with conjoint analysis above)
                y = df["BRAND"].head(10001) # Subsampling for speed and efficiency
                # y = df2.BRAND.apply(lambda x: '1' if 'Castle' in x else 0).head(10001) # Dynamic to brand, subsampling for speed and efficiency

                # Change the array shape of the output from a dataframe single column vector to a contiguous flattened array to avoid technical issues
                # Could technically just go ahead with regressor but would get lots of warning messages
                y = np.ravel(y) # Could use "flatten" function

                # Split the data into the training set and testing (accuracy scores) set
                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

                # Scale the data - instance of scaler
                scaler = StandardScaler()  

                # Fit using only the training data
                scaler.fit(X_train)  
                X_train = scaler.transform(X_train)  

                # Apply the same transformation to test data
                X_test = scaler.transform(X_test)

                # Create instance of class we imported

                reg = MLPClassifier(max_iter=2000, hidden_layer_sizes=(5,5), random_state=1)
                reg.fit(X_train, y_train)

                # Predict
                y_pred = reg.predict(X_test)
                    
                # Accuracy before model parameter optimisation
                st.write("Accuracy score before model parameter optimization: %0.3f" % accuracy_score(y_pred,y_test))

                # Fit and check accuracy for various numbers of nodes on both layers
                # Note this will take some time
                # Hidden layer size is a tuple i:j 3 to 6
                validation_scores = {}
                st.write("Nodes | Validation score")
                # st.write("      | score")

                for hidden_layer_size in [(i,j) for i in range(3,7) for j in range(3,7)]:

                    reg = MLPClassifier(max_iter=2000, hidden_layer_sizes=hidden_layer_size, random_state=1)

                    score = cross_val_score(estimator=reg, X=X_train, y=y_train, cv=2)
                    validation_scores[hidden_layer_size] = score.mean() # Mean of scores on cross validations using 2 CVs (in halves; already computationally intensive)
                    print(hidden_layer_size, ": %0.5f" % validation_scores[hidden_layer_size])

                # Vizualise these using a 3D surface plot

                fig = plt.figure()
                ax = fig.gca(projection='3d')

                # Prepare the data, x, y, and z as 2D arrays, i.e. unflatten the list (list comprehension, 2 lists in 1)
                px, py = np.meshgrid(np.arange(3,7), np.arange(3,7))
                pz = np.array([[validation_scores[(i,j)] for i in range(3,7)] for j in range(3,7)])

                # Customize the z-axis
                ax.set_zlim(0.2, .3)

                # Plot the surface
                surf = ax.plot_surface(px, py, pz)
                plt.show()

                buf = BytesIO()
                fig.savefig(buf, format="png")
                st.image(buf)

                # Check scores, from array
                st.write("The highest validation score is: %0.4f" % max(validation_scores.values()))  
                optimal_hidden_layer_size = [name for name, score in validation_scores.items() 
                                            if score==max(validation_scores.values())][0]
                st.write("This corresponds to nodes", optimal_hidden_layer_size )

                # Fit data with best parameter
                clf = MLPClassifier(max_iter=2000, 
                                    hidden_layer_sizes=optimal_hidden_layer_size, 
                                    random_state=1) # Fit instance to CLFclassifier from class MLPClassifier
                clf.fit(X_train, y_train)
                # Does not converge fully without changing max_iter

                # Predict
                y_pred = clf.predict(X_test)

                # Accuracy 
                st.write("Accuracy score after model parameter optimization: %0.3f" % accuracy_score(y_pred,y_test))

                # Draw a response function to observe response vs desired feature

                # Consider converting features to mean

                # Copy dataframe so as to not change original, and obtain medians
                X_design = X.copy()
                X_design_vec = pd.DataFrame(X_design.median()).transpose()

                # View X_design_vec
                X_design_vec.head()

                # Find the min and max of the desired feature and set up a sequence
                min_feature = min(X.loc[:,"AGE"]) # Dynamic to desired feature
                max_feature = max(X.loc[:,"AGE"]) # Dynamic to desired feature
                seq = np.linspace(start=min_feature,stop=max_feature,num=50)

                # Set up a list of moving features
                to_predict = []
                for result in seq:
                    X_design_vec.loc[0,"AGE"] = result # Dynamic to desired feature
                    to_predict.append(X_design_vec.copy())

                # Convert back to dataframe
                to_predict = pd.concat(to_predict)

                # Scale and predict
                to_predict = scaler.transform(to_predict)
                predictions = clf.predict(to_predict)

                # Plot 
                plt.plot(seq,predictions)
                plt.xlabel("Age") # Dynamic to desired feature
                plt.ylabel("Brand") # Dynamic to desired feature
                plt.title("Response vs Age") # Dynamic to desired feature
                plt.show()

                buf2 = BytesIO()
                fig.savefig(buf2, format="png")
                st.image(buf2)

                st.write("")

                st.write("Running the neural networks model took ", time.time() - start_time, "seconds to run")

        if modelchoice == "Decision trees":

            with st.spinner('Please wait while we conduct the decision tree analysis'):

                my_bar = st.progress(0)

                time.sleep(10)

                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1)

                start_time = time.time()

                # Split data into features (X) and response (y)
                X = df.loc[0:10000,["GENDER", "AGE", "CITY"]] # Subsampling for speed and efficiency, dynamic to certain relevant explanatory variables (could combine with dimensionality reduction or with conjoint analysis above)
                y = df2.BRAND.apply(lambda x: '1' if 'Castle' in x else 0).head(10001) # Dynamic to brand, subsampling for speed and efficiency

                # Fit data to tree-based regression model
                regressor = DecisionTreeRegressor(random_state=0)
                regressor=regressor.fit(X,y)

                # Visualising the decision tree regression results
                plt.figure(figsize=(6,6), dpi=150)
                plot_tree(regressor,max_depth=3,feature_names=X.columns, impurity=False)
                plt.show()

                buf = BytesIO()
                plt.savefig(buf, format="png")
                st.image(buf)

                # A scatter plot of latitude vs longitude
                df.plot.scatter(x='AGE',y='GENDER',c='DarkBlue',s=1.5) # Dynamic to chosen variables

                plt.figure(figsize=[6,4], dpi=120)
                cutx, cuty = -116, 42
                plt.ylim(33,cuty)       
                plt.xlim(-125,cutx)
                plt.xlabel('AGE')
                plt.ylabel('GENDER')
                plt.scatter(x=X['AGE'],y=X['GENDER'],c=df['BRAND'].head(10001)) # Dynamic to chosen variables

                buf2 = BytesIO()
                plt.savefig(buf2, format="png")
                st.image(buf2)

                splits = regressor.tree_.threshold[:2]
                print(splits, cutx, cuty)
                plt.plot([splits[1],splits[1]], [0,cuty]) 
                plt.plot([splits[1],cutx], [splits[0],splits[0]])
                plt.colorbar()
                plt.show()

                buf3 = BytesIO()
                plt.savefig(buf3, format="png")
                st.image(buf3)

                st.write("")

                st.write("Conducting the decision tree analysis took ", time.time() - start_time, "seconds to run")

        if modelchoice == "Ensemble methods - random forest":

            with st.spinner('Please wait while we conduct the ensemble methods analysis using the random forest algorithm'):

                my_bar = st.progress(0)

                time.sleep(10)

                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1)

                start_time = time.time()

                # test classification dataset

                st.write("Test classification dataset:")

                # define dataset
                X = df.loc[0:10000,["GENDER", "AGE", "CITY"]] # Subsampling for speed and efficiency, dynamic to certain relevant explanatory variables (could combine with dimensionality reduction or with conjoint analysis above)
                y = df2.BRAND.head(10001) #.apply(lambda x: '1' if '*astle' in x else 0).head(10001) # Dynamic to brand, subsampling for speed and efficiency
                # summarize the dataset
                st.write("The shape of your subsampled data, in the form of explanatory variables (gender, age, city), and predicted variable (brand):")
                st.write(X.shape, y.shape)

                # evaluate random forest algorithm for classification

                st.write("")

                st.write("Evaluate random forest algorithm for classification:")

                # define the model
                model = RandomForestClassifier()
                # evaluate the model
                cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
                n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
                # report performance
                st.write('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

                # random forest for classification
                # define 

                st.write("")
                
                st.write("Model outputs:")

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
                # define the model
                model = RandomForestClassifier()
                # fit the model on the training dataset
                model.fit(X_train, y_train)
                # make a single prediction
                print(X)
                row = X.head(1)
                yhat = model.predict(row)
                st.write('Predicted Class: ' + yhat[0])

                st.write("")

                # test regression dataset
                st.write("Test regression dataset:")
                # define dataset
                # Choose predicted variable - this will become dynamic in the app
                y = df['BRAND'].head(10000)
                # Define predictor variables
                X = df.iloc[:, 1:-1].head(10000)
                # X, y = np.array(x), np.array(y)
                # summarize the dataset
                st.write("The shape of your subsampled data, in the form of explanatory variables (gender, age, city), and predicted variable (brand):")
                st.write(X.shape, y.shape)

                st.write("")

                # evaluate random forest ensemble for regression
                st.write("Evaluate random forest ensemble for regression:")
                # define the model
                model = RandomForestRegressor()
                # evaluate the model
                cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
                n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
                # report performance
                st.write('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

                st.write("")

                # random forest for making predictions for regression
                st.write("Making predictions for regression:")
                # define dataset
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
                # define the model
                model = RandomForestRegressor()
                # fit the model on the whole dataset
                model.fit(X_train, y_train)
                # make a single prediction
                row = X.head(1)
                yhat = model.predict(row)
                st.write('Prediction: %d' % yhat[0])

                st.write("")

                st.write("")

                st.write("Conducting the random forest model took ", time.time() - start_time, "seconds to run")

                st.write("")

            tunerfparams = st.button("Click here to tune the hyperparameters for the random forest model")

            if tunerfparams == True:

                with st.spinner('Please wait while we tune the hyperparameters for the random forest model'):

                    my_bar = st.progress(0)

                    time.sleep(10)

                    for percent_complete in range(100):
                        time.sleep(0.01)
                        my_bar.progress(percent_complete + 1)

                    start_time = time.time()

                    # Tuning hyperparameters for random forest

                    # 1

                    # Bootstrap sizes of 10-100% on random forest algorithm
                    # Explore random forest bootstrap sample size on performance
                    # Bootstrap sample size that is equal to the size of the training dataset generally achieves the best results and is the default

                    st.write("Explore bootstrap sizes of '10-100%' on random forest algorithm")
                    st.write("Bootstrap sample size that is equal to the size of the training dataset generally achieves the best results and is the default")

                    # get the dataset
                    def get_dataset():
                        X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
                        return X, y

                    # get a list of models to evaluate
                    def get_models():
                        models = dict()
                        # explore ratios from 10% to 100% in 10% increments
                        for i in np.arange(0.1, 1.1, 0.1):
                            key = '%.1f' % i
                            # set max_samples=None to use 100%
                            if i == 1.0:
                                i = None
                            models[key] = RandomForestClassifier(max_samples=i)
                        return models

                    # evaluate a given model using cross-validation
                    def evaluate_model(model, X, y):
                        # define the evaluation procedure
                        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
                        # evaluate the model and collect the results
                        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
                        return scores

                    # define dataset
                    X, y = get_dataset()
                    # get the models to evaluate
                    models = get_models()
                    # evaluate the models and store results
                    results, names = list(), list()
                    for name, model in models.items():
                        # evaluate the model
                        scores = evaluate_model(model, X, y)
                        # store the results
                        results.append(scores)
                        names.append(name)
                        # summarize the performance along the way
                        st.write('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
                    # plot model performance for comparison
                    pyplot.boxplot(results, labels=names, showmeans=True)
                    pyplot.show()

                    buf1 = BytesIO()
                    plt.savefig(buf1, format="png")
                    st.image(buf1)

                    # 2

                    # Explore random forest number of features effect on performance

                    st.write("")

                    st.write("Explore random forest number of features effect on performance")

                    # get the dataset
                    def get_dataset():
                        X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
                        return X, y

                    # get a list of models to evaluate
                    def get_models():
                        models = dict()
                        # explore number of features from 1 to 21
                        for i in range(1,22):
                            models[str(i)] = RandomForestClassifier(max_features=i)
                        return models

                    # evaluate a given model using cross-validation
                    def evaluate_model(model, X, y):
                        # define the evaluation procedure
                        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
                        # evaluate the model and collect the results
                        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
                        return scores

                    # define dataset
                    X, y = get_dataset()
                    # get the models to evaluate
                    models = get_models()
                    # evaluate the models and store results
                    results, names = list(), list()
                    for name, model in models.items():
                        # evaluate the model
                        scores = evaluate_model(model, X, y)
                        # store the results
                        results.append(scores)
                        names.append(name)
                        # summarize the performance along the way
                        st.write('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
                    # plot model performance for comparison
                    pyplot.boxplot(results, labels=names, showmeans=True)
                    pyplot.show()

                    buf2 = BytesIO()
                    plt.savefig(buf2, format="png")
                    st.image(buf2)

                    # 3

                    # Explore random forest number of trees effect on performance

                    st.write("")

                    st.write("Explore random forest number of trees effect on performance")

                    # get the dataset
                    def get_dataset():
                        X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
                        return X, y

                    # get a list of models to evaluate
                    def get_models():
                        models = dict()
                        # define number of trees to consider
                        n_trees = [10, 50, 100, 500, 1000]
                        for n in n_trees:
                            models[str(n)] = RandomForestClassifier(n_estimators=n)
                        return models

                    # evaluate a given model using cross-validation
                    def evaluate_model(model, X, y):
                        # define the evaluation procedure
                        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
                        # evaluate the model and collect the results
                        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
                        return scores

                    # define dataset
                    X, y = get_dataset()
                    # get the models to evaluate
                    models = get_models()
                    # evaluate the models and store results
                    results, names = list(), list()
                    for name, model in models.items():
                        # evaluate the model
                        scores = evaluate_model(model, X, y)
                        # store the results
                        results.append(scores)
                        names.append(name)
                        # summarize the performance along the way
                        st.write('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
                    # plot model performance for comparison
                    pyplot.boxplot(results, labels=names, showmeans=True)
                    pyplot.show()

                    buf3 = BytesIO()
                    plt.savefig(buf3, format="png")
                    st.image(buf3)

                    # 4

                    # Explore random forest tree depth effect on performance

                    st.write("")

                    st.write("Explore random forest tree depth effect on performance")

                    # get the dataset
                    def get_dataset():
                        X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
                        return X, y

                    # get a list of models to evaluate
                    def get_models():
                        models = dict()
                        # consider tree depths from 1 to 7 and None=full
                        depths = [i for i in range(1,8)] + [None]
                        for n in depths:
                            models[str(n)] = RandomForestClassifier(max_depth=n)
                        return models

                    # evaluate a given model using cross-validation
                    def evaluate_model(model, X, y):
                        # define the evaluation procedure
                        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
                        # evaluate the model and collect the results
                        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
                        return scores

                    # define dataset
                    X, y = get_dataset()
                    # get the models to evaluate
                    models = get_models()
                    # evaluate the models and store results
                    results, names = list(), list()
                    for name, model in models.items():
                        # evaluate the model
                        scores = evaluate_model(model, X, y)
                        # store the results
                        results.append(scores)
                        names.append(name)
                        # summarize the performance along the way
                        st.write('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
                    # plot model performance for comparison
                    pyplot.boxplot(results, labels=names, showmeans=True)
                    pyplot.show()

                    buf4 = BytesIO()
                    plt.savefig(buf4, format="png")
                    st.image(buf4)

                    st.write("")

                    st.write("Tuning hyperparameters for the random forest model took ", time.time() - start_time, "seconds to run")

        if modelchoice == "This will come later":

            max_depth = st.slider("What should be the max_depth of the model?", min_value=10, max_value=100, value=20, step=10)
            n_estimators = st.selectbox("How many trees?", options=[100, 200, 300, 'No limit'], index = 0)
            st.text("Here is a list of features from the database:")
            input_feature = st.text_input("What is the input feature?", 'uuid')

if __name__ == '__main__':
    if streamlit._is_running_with_streamlit:
        abiui.run()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())