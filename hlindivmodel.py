import hydralit as hy
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
from matplotlib.pyplot import hist
from scipy import stats as stats
from bioinfokit.analys import stat
import statsmodels.api as sm
from statsmodels.graphics.factorplots import interaction_plot
from hlautoanalytics import autoanalytics
from st_click_detector import click_detector
from streamlit_option_menu import option_menu
from streamlit_quill import st_quill
import dataingestion

#add an import to Hydralit
from hydralit import HydraHeadApp
from hydralit import HydraApp

#create a wrapper class
class indivmodels(HydraHeadApp):

#wrap all your code in this method and you should be done

    def run(self):

        ### UNTOUCHED ORIGINAL CODE
        
        # if 'model' not in st.session_state:
        #     st.session_state.model = "None"

        # Virtual assistant #1

        # with st.sidebar:

            # import torch
            # import transformers
            # from transformers import AutoModelForCausalLM, AutoTokenizer

            # @st.cache(hash_funcs={transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast: hash}, suppress_st_warning=True)
            # def load_data():    
            #     tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
            #     model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
            #     return tokenizer, model
            # tokenizer, model = load_data()

            # st.write("Welcome to the Chatbot. I am still learning, please be patient")
            # input = st.text_input('User:')
            # if 'count' not in st.session_state or st.session_state.count == 6:
            #     st.session_state.count = 0 
            #     st.session_state.chat_history_ids = None
            #     st.session_state.old_response = ''
            # else:
            #     st.session_state.count += 1

            # new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')

            # bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_user_input_ids], dim=-1) if st.session_state.count > 1 else new_user_input_ids

            # st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=5000, pad_token_id=tokenizer.eos_token_id)

            # response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

            # if st.session_state.old_response == response:
            #     bot_input_ids = new_user_input_ids
 
            #     st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=5000, pad_token_id=tokenizer.eos_token_id)
            #     response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

            # st.write(f"Chatbot: {response}")

            # st.session_state.old_response = response

            # Virtual assistant #2

            # with st.sidebar:

            #     #importing libraries
            #     from chatterbot import ChatBot
            #     from chatterbot.trainers import ListTrainer
            #     from chatterbot.trainers import ChatterBotCorpusTrainer 
            #     import json
            #     #get_text is a simple function to get user input from text_input
            #     def get_text():
            #         input_text = st.text_input("You: ","So, what's in your mind")
            #         return input_text
            #     #data input
            #     data = json.loads(open(r'C:\Users\Jojo\Desktop\projects\chatbot\chatbot\chatbot\data_tolokers.json','r').read())#change path accordingly
            #     data2 = json.loads(open(r'C:\Users\Jojo\Desktop\projects\chatbot\chatbot\chatbot\sw.json','r').read())#change path accordingly
            #     tra = []
            #     for k, row in enumerate(data):
            #         #print(k)
            #         tra.append(row['dialog'][0]['text'])
            #     for k, row in enumerate(data2):
            #         #print(k)
            #         tra.append(row['dialog'][0]['text'])
            #     st.sidebar.title("NLP Bot")
            #     st.title("""
            #     NLP Bot  
            #     NLP Bot is an NLP conversational chatterbot. Initialize the bot by clicking the "Initialize bot" button. 
            #     """)
            #     bot = ChatBot(name = 'PyBot', read_only = False,preprocessors=['chatterbot.preprocessors.clean_whitespace','chatterbot.preprocessors.convert_to_ascii','chatterbot.preprocessors.unescape_html'], logic_adapters = ['chatterbot.logic.MathematicalEvaluation','chatterbot.logic.BestMatch'])
            #     ind = 1
            #     if st.sidebar.button('Initialize bot'):
            #         trainer2 = ListTrainer(bot) 
            #         trainer2.train(tra)
            #         st.title("Your bot is ready to talk to you")
            #         ind = ind +1
                        
            #     user_input = get_text()
            #     if True:
            #         st.text_area("Bot:", value=bot.get_response(user_input), height=200, max_chars=None, key=None)
            #     else:
            #         st.text_area("Bot:", value="Please start the bot by clicking sidebar button", height=200, max_chars=None, key=None)

            # Virtual assistant #3

            # from streamlit_chat import message as st_message
            # from transformers import BlenderbotTokenizer
            # from transformers import BlenderbotForConditionalGeneration

            # with st.sidebar:

            #     @st.experimental_singleton
            #     def get_models():
            #         # it may be necessary for other frameworks to cache the model
            #         # seems pytorch keeps an internal state of the conversation
            #         model_name = "facebook/blenderbot-400M-distill"
            #         tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
            #         model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
            #         return tokenizer, model


            #     if "history" not in st.session_state:
            #         st.session_state.history = []

            #     st.title("Hello Chatbot")


            #     def generate_answer():
            #         tokenizer, model = get_models()
            #         user_message = st.session_state.input_text
            #         inputs = tokenizer(st.session_state.input_text, return_tensors="pt")
            #         result = model.generate(**inputs)
            #         message_bot = tokenizer.decode(
            #             result[0], skip_special_tokens=True
            #         )  # .replace("<s>", "").replace("</s>", "")

            #         st.session_state.history.append({"message": user_message, "is_user": True})
            #         st.session_state.history.append({"message": message_bot, "is_user": False})


            #     st.text_input("Talk to the bot", key="input_text", on_change=generate_answer)

            #     for chat in st.session_state.history:
            #         st_message(**chat)  # unpacking

        st.session_state['pagechoice'] = 'analytics'

        with st.sidebar:
            choose = option_menu("ABI Analytics", ["Keep in mind", "Show me the steps", "Give me tips", "Give feedback"],
                                icons=['key', 'bezier2', 'joystick', 'keyboard'],
                                menu_icon="app-indicator", default_index=0,
                                styles={
                "container": {"padding": "5!important", "background-color": "#fafafa"},
                "icon": {"color": "black", "font-size": "25px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#ab0202"},
            }
            )

            print(st.session_state.pagechoice)
            
            if choose == "Keep in mind":
                st.write("Remember to ask good questions. That is the basis of making good decisions.")

            if choose == "Show me the steps" and st.session_state.pagechoice=="test":
                st.write("Steps you should follow:")

            if choose == "Give me tips":
                st.write("Here are some tips:")

            if choose == "Give feedback":
                st.write("Give feedback")
                with st.sidebar.form(key='columns_in_form',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
                    rating=st.slider("Please rate the app", min_value=1, max_value=5, value=3,help='Drag the slider to rate the app. This is a 1-5 rating scale where 5 is the highest rating')
                    text=st.text_input(label='Please leave your feedback here')
                    submitted = st.form_submit_button('Submit')
                    if submitted:
                        st.write('Thanks for your feedback!')
                        st.markdown('Your Rating:')
                        st.markdown(rating)
                        st.markdown('Your Feedback:')
                        st.markdown(text)

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
                st.markdown("<h1 style='text-align: center; color: red;'>"'Run a specific model'"</h1>", unsafe_allow_html=True)
            with col3:
                pass

        # Automate button

        # with guidance:

        #     col1,col2,col3 = st.columns(3)
        #     with col1:
        #         pass
        #     with col2:
        #         st.markdown("<h3 style='text-align: center; color: green;'>"'Press the button below to automate the experience'"</h3>", unsafe_allow_html=True)
        #         content = """
        #         <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
        #         <style>
        #         .zoom {
        #         padding: 10px;
        #         # width: 210px;
        #         # height: 210px;
        #         transition: transform .21s; /* Animation */
        #         margin: 0 auto;
        #         }
        #         .zoom:hover {
        #         transform: scale(1.1); /* (110% zoom - Note: if the zoom is too large, it will go outside of the viewport) */
        #         }
        #         </style>
        #         <div class="w3-container">
        #         <a href='#' id='Image 1'><img width='100%' src='https://i.postimg.cc/158HHcRn/Automate.png' class='zoom w3-hover-opacity w3-image'></a>
        #         </div>
        #         """
        #         clicked = click_detector(content)

        #         # st.markdown(f"**{clicked} clicked**" if clicked != "" else "")
                
        #         if clicked == 'Image 1':
        #             # self.do_redirect('Automated analytics flow')
        #             self.do_redirect(redirect_target_app='Automated analytics flow')
    
        #     with col3:
        #         pass

        # OLDER BUT WORKING CLICKABLE IMAGE

        #     clicked = clickable_images(
        #     [
        #         "https://i.postimg.cc/158HHcRn/Automate.png", # Link for links to picture: https://postimg.cc/7fy0Pz2w/b5680871
        #     ],
        #     titles=[f"Image #{str(i)}" for i in range(1)], # Change range if more pictures are needed
        #     div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
        #     img_style={"margin": "5px", "height": "200px"},
        # )

        #     # st.markdown(f"Image #{clicked} clicked" if clicked > -1 else "")

        #     if clicked == 0:
        #         # st.success("We will automate the experience for you")
        #         autoanalytics.run()

        # END OLDER CLICKABLE IMAGE

            st.button("AutoML - choose the best fit model for your analysis using FLAML")

        with dataset:
            # components.html('''<script src="//ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js"></script>
            # <script src="jquery.zoomooz.min.js"></script>''')
            # st.markdown('''<div class="zoomTarget" data-targetsize="0.45" data-duration="600">This element zooms when clicked on.</div>''', unsafe_allow_html=True)

            df, df2, branddf = dataingestion.readdata()
            print(df.head())

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
            modelchoice = st.selectbox("Please choose preferred model", ["Please select model", "Exploratory data analysis", "Correlation analysis", "Hypothesis testing", "Linear regression", "Logistic regression", "Cluster analysis", "ANOVA", "Principal component analysis", "Conjoint analysis", "Neural networks", "Decision trees", "Ensemble methods - random forest"], key="modelchoice")

            if modelchoice == "Exploratory data analysis":

                st.subheader("Exploratory data analysis")

                # Pandas profiling, Autoviz, Dtale

                datadescrip = st.expander("Description of data")
                
                with datadescrip:

                    st.write(df.describe(include=np.number))
                    # pass

                if get_instance("1") is None:
                    startup(data_id="1", data=df)

                d=get_instance("1")

                checkbtn = st.button("Validate data")

                if checkbtn != True:
                
                    # webbrowser.open_new_tab('http://localhost:8501/dtale/main/1')
                    html = f"""<iframe src="/dtale/main/1" height="1000" width="1400"></iframe>""" # Iframe
                    
                    st.markdown(html, unsafe_allow_html=True)

                if checkbtn == True:
                    df_amended = get_instance(data_id="1").data # The amended dataframe ready for upsert
                    st.write("Sample of amended data:")
                    st.write("")
                    st.write(df_amended.head(5))

                clearchanges = st.button("Clear changes made to data")
                if clearchanges == True:
                    global_state.cleanup()

                # Spawn a new Quill 
                st.subheader("Notes on exploratory data analysis")
                edacontent = st_quill(placeholder="Write your notes here")

            if modelchoice == "Correlation analysis":

                st.subheader("Correlation analysis")

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

                    with st.expander("Interpretation guide"):
                        st.write("Variables within a dataset can be related for many reasons.\n\n"
                        "For example:\n"
                        "One variable could cause or depend on the values of another variable.\n"
                        "One variable could be lightly associated with another variable.\n"
                        "Two variables could depend on a third unknown variable.\n\n"
                        "It can be useful in data analysis and modeling to better understand the relationships between variables. The statistical relationship between two variables is referred to as their correlation.\n"
                        "A correlation could be positive, meaning both variables move in the same direction, or negative, meaning that when one variable's value increases, the other variables' values decrease. Correlation can also be neutral or zero, meaning that the variables are unrelated.")
                        st.info("Positive Correlation: both variables change in the same direction.\n\n"
                        "Neutral Correlation: No relationship in the change of the variables.\n\n"
                        "Negative Correlation: variables change in opposite directions.\n\n"
                        "A correlation coefficient closer to 1 or to -1 indicates stronger relationships between the variables, while a correlation coefficient closer to 0 indicates a lesser relationship between the variables.")
                        st.write("The performance of some algorithms can deteriorate if two or more variables are tightly related, called multicollinearity. An example is linear regression, where one of the offending correlated variables should be removed in order to improve the skill of the model.\n\n"
                        "We may also be interested in the correlation between input variables with the output variable in order provide insight into which variables may or may not be relevant as input for developing a model.\n"
                        "The structure of the relationship may be known, e.g. it may be linear, or we may have no idea whether a relationship exists between two variables or what structure it may take. Depending what is known about the relationship and the distribution of the variables, different correlation scores can be calculated.")

                    ### End

                    st.write("")

                    # Spawn a new Quill editor
                    st.subheader("Notes on correlation analysis")
                    correlcontent = st_quill(placeholder="Write your notes here")
                    st.session_state.correlnotes = correlcontent

                    st.write("Correlation analysis took ", time.time() - start_time, "seconds to run")

            if modelchoice == "Hypothesis testing":

                st.subheader("Hypothesis testing on characteristics with optional disaggregation by brand or brand groups")

                with st.spinner('Please wait while we conduct hypothesis testing'):

                    my_bar = st.progress(0)

                    for percent_complete in range(100):
                        time.sleep(0.1)
                        my_bar.progress(percent_complete + 1)

                    start_time = time.time()

                    st.write("**Background to the model**")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        pass
                    with col2:
                        st.image("ttest.png")
                    with col3:
                        pass

                    with st.expander("Assumptions of the one-sample t-test"):
                        st.success("One-sample t-test")
                        st.write("Student's t-test or t-test is a parametric inferential statistical method used for comparing the means between two different groups (two-sample t-test) or with the specific value (one-sample t-test).")
                        st.write("This test works well for hypothesis testing, particularly amongst data with low sample sizes (in contrast to the z-test).")
                        st.write("One Sample t-test (single sample t-test) is used to compare the sample mean (a random sample from a population) with the specific value (hypothesized or known mean of the population).")
                        st.info("Dependent variable should have an approximately normal distribution (Shapiro-Wilks Test).")
                        st.info("Observations are independent of each other.")
                        st.info("Note: One sample t-test is relatively robust to the assumption of normality when the sample size is large (n ≥ 30).")
                        st.warning("Null hypothesis: Sample mean is equal to the hypothesized or known population mean.")
                        st.warning("Alternative hypothesis: Sample mean is not equal to the hypothesized or known population mean (two-tailed or two-sided).")
                        st.warning("Alternative hypothesis: Sample mean is either greater or lesser to the hypothesized or known population mean (one-tailed or one-sided).")

                    with st.expander("Assumptions of the two-sample t-test"):
                        st.success("One-sample t-test")
                        st.write("The two-sample (unpaired or independent) t-test compares the means of two independent groups, determining whether they are equal or significantly different. In two sample t-test, usually, we compute the sample means from two groups and derives the conclusion for the population’s means (unknown means) from which two groups are drawn.")
                        st.info("Observations in two groups have an approximately normal distribution (Shapiro-Wilks Test).")
                        st.info("Homogeneity of variances (variances are equal between treatment groups) (Levene or Bartlett Test).")
                        st.info("The two groups are sampled independently from each other from the same population.")
                        st.info("Note: Two sample t-test is relatively robust to the assumption of normality and homogeneity of variances when sample size is large (n ≥ 30) and there are equal number of samples (n1 = n2) in both groups.")
                        st.warning("Null hypothesis: Two group means are equal.")
                        st.warning("Alternative hypothesis: Two group means are different (two-tailed or two-sided).")
                        st.warning("Alternative hypothesis: Mean of one group either greater or lesser than another group (one-tailed or one-sided).")

                    with st.expander("Sample size considerations for t-test"):
                        st.write("The t-test can be applied for the extremely small sample size (n = 2 to 5) provided the effect size is large and data follows the t-test assumptions. Remember, a larger sample size is preferred over small sample sizes.")
                        st.write("t-test is relatively robust to the assumption of normality and homogeneity of variances when the sample size is large (n ≥ 30).")

                    brands = pd.Series(df2['BRAND'].drop_duplicates()).sort_values().tolist() # Faster than .unique()
                    brandsmod = ["None"] + brands
                    
                    st.markdown("**Please choose brands. If only one brand is chosen, a one sample t-test will be performed. If two brands are chosen, a two sample t-test will be performed.**")

                    brand1 = st.selectbox("Please select first brand of interest.", brandsmod, key="brand1")
                    brand2 = st.selectbox("Please select second brand of interest.", brandsmod, key="brand2")

                    ttestvars = list(df.drop(['BRAND', 'BRANDNAME'], axis=1).columns)
                    # ttestvars2 = ["NONE"] + ttestvars
                    ttestvars.insert(0, "NONE")

                    var1 = st.selectbox("Please select variable for analysis", ttestvars)
                    # var2 = st.selectbox("Please select second variable for analysis", ttestvars2)

                    if var1 != "NONE":

                        mask1 = df.BRANDNAME==brand1
                        dfa = df[mask1]

                        mask2 = df.BRANDNAME==brand2
                        dfb = df[mask2]

                        result1s, result2s = [0,0]

                        from scipy import stats

                        if brand1 != "None":

                            if brand2 == "None":

                                # 1-sample t-test using scipy
                                a =  dfa[var1].to_numpy()
                                print(a)
                                popmean = df[var1].mean()
                                # use parameter "alternative" for two-sided or one-sided test
                                st.write("**Results of one sample t-test:**")
                                result1s = stats.ttest_1samp(a=a, popmean=popmean)
                                st.write(result1s)

                                res = stat()
                                res.ttest(df=dfa, test_type=1, res=var1, mu=5)
                                st.text(res.summary)

                            else:

                                # 2-sample t-test

                                options = [brand1, brand2]

                                # dfc = df.loc[(df["BRANDNAME"] == brand1) & (df["BRANDNAME"] == brand2)]

                                dfc = df[df['BRANDNAME'].isin(options)]

                                # dfc = df.query("BRAND==" + brand1 + " & BRAND==" + brand2)

                                a = dfa[var1].to_numpy()
                                b = dfb[var1].to_numpy()

                                result2s = stats.ttest_ind(a=a, b=b, equal_var=False)

                                st.write(result2s)

                                dfc.loc[dfc["BRANDNAME"] == brand1, "BRAND"] = 1
                                dfc.loc[dfc["BRANDNAME"] == brand2, "BRAND"] = 2

                                res2=stat()
                                res2.ttest(df=dfc, xfac="BRANDNAME", res=var1, evar=False, test_type=2) # evar=False for unequal variance t-test (Welch)
                                st.text(res2.summary)

                                st.write("**Description of chosen variable for each brand**")

                                # st.write(df[var1].loc[df["BRANDNAME" == brand1]].describe())
                                st.write(dfc[[var1, "BRAND"]].groupby("BRAND").describe())

                                print(brand1, brand2)

                                # cond = {'a': brand1, 'b': brand2}
                                # hist = df.loc[(df.BRANDNAME == cond['a']) & (df.BRANDNAME == cond['b']), ['x','y']].plot(title='a: {a}, b: {b} distribution histogram'.format(**cond))

                                # fig, ax = df.plot.hist(bins=12, alpha=0.5)

                                subheader = ("**Histogram of {} for each brand chosen**").format(var1)

                                st.write(subheader)

                                hist = dfc.pivot(columns='BRANDNAME').AGE.plot(kind = 'hist', stacked=True, alpha=0.3, title="Frequency distribution histogram for {}".format(var1)).figure

                                buf = BytesIO()
                                hist.savefig(buf, format="png")
                                # st.pyplot(fig)
                                st.image(buf)

                            with st.expander("Interpretation guide"):
                                st.info("The higher the t-score, and the lower the p-value, the more able you are to reject the null hypothesis in favour of the alternative hypothesis.")
                                st.info("Couple this insight with a view of the underlying distributions, to see how the hypotheses function.")
                                # print(result1s, result2s, result2s[1])
                                if brand2 == "None":
                                    res1 = result1s.pvalue
                                    if res1 <= 0.1:
                                        st.success(f"Our p-value is less than 0.1, so we can reject the null hypothesis (10% level of significance) and conclude that our consumers for {brand1} are different to the total population of consumers across different brands with respect to {var1}. Consult the histogram to see how.")
                                    if res1 > 0.1:
                                        st.error(f"Our p-value is more than 0.1, so we can reject the null hypothesis (10% level of significance) and conclude that our consumers for {brand1} are similar to the total population of consumers across different brands with respect to {var1}.")
                                else:
                                    res2 = result2s.pvalue
                                    if res2 <= 0.1:
                                        st.success(f"Our p-value is less than 0.1, so we can reject the null hypothesis (10% level of significance) and conclude that our consumers for each chosen brand are different with respect to {var1}. Consult the histogram to see how.")
                                    if res2 > 0.1:
                                        st.error(f"Our p-value is more than 0.1, so we can reject the null hypothesis (10% level of significance) and conclude that our consumers for each of the chosen brands are similar with respect to {var1}.")

                    else:
                        pass

                # Spawn a new Quill editor
                st.subheader("Notes on hypothesis analysis")
                hyptestingcontent = st_quill(placeholder="Write your notes here")       

                st.write("Hypothesis testing took ", time.time() - start_time, "seconds to run")

            if modelchoice == "Linear regression":

                st.subheader("Linear regression analysis")

                with st.spinner('Please wait while we conduct the linear regression analysis'):

                    my_bar = st.progress(0)

                    time.sleep(5)

                    for percent_complete in range(100):
                        time.sleep(0.2)
                        my_bar.progress(percent_complete + 1)

                    start_time = time.time()

                    ### From Jupyter - 2. Linear regression

                    # Choose predicted variable - this will become dynamic in the app
                    regvars = list(df.columns)
                    # regvars.insert(0, "NONE")
                    # y = df['BRAND'] # old predicted variable
                    chosenvar = st.selectbox("Please select variable for analysis", ["Choose dependent variable"] + regvars)

                    if chosenvar != "Choose dependent variable":

                        y = df[chosenvar]
                        df3 = y.copy()

                        # Define predictor variables
                        df2 = df.drop([chosenvar], axis=1)
                        indepvars = st.multiselect("Please select independent (input) variables for analysis", regvars)
                        # df3 = df2.copy()
                        # df3 = y.copy()

                        if indepvars != []:

                            for indepvar in indepvars:
                                dfi = df[indepvar]
                                df3 = pd.concat((df3, dfi), axis=1)
                            x = df3.drop([chosenvar], axis=1)
                            # print(df3)
                            # x = df3.iloc[:, :]

                        if chosenvar != "NONE" and indepvars != []: 

                            x, y = np.array(x), np.array(y)

                            x = sm.add_constant(x)

                            model = sm.OLS(y, x)

                            results = model.fit()

                            st.write(results.summary())

                            st.write("")

                            st.write('Predicted response:', results.fittedvalues, sep='\n') # Or print('predicted response:', results.predict(x), sep='\n')

                            ### End

                            st.write("")

                            with st.expander("Interpretation guide"):
                                st.write("Regression searches for relationships (mathematical dependencies) among variables. This differs from correlation because regression analysis seeks to unpack **causal** relationships, where a change in one variable causes another variable to change. We find a function that maps some features or variables to others sufficiently well.\n\n"
                                "The dependent features are called the dependent variables, outputs, or responses. The independent features are called the independent variables, inputs, or predictors.\n\n"
                                "Regression problems usually have one continuous and unbounded dependent variable. The inputs, however, can be continuous, discrete, or even categorical data such as gender, nationality, brand, and so on. It is a common practice to denote the outputs with y and inputs with x.\n\n"
                                "Regression is also useful when you want to forecast a response using a new set of predictors. The regression coefficients can be applied as a multiplier factor to understand how changes in an predictor feature would cause the predicted variable to change.")
                                st.info("Regression coefficients indicate the predicted change in the output variable caused by a change in the input variable.")
                                st.info("The estimated or predicted response for each observation should be as close as possible to the corresponding actual response in the underlying data (in real life). The differences for all observations from the actual responses are called the residuals. Regression is about determining the best predicted weights, that is the weights corresponding to the smallest residuals. To get the best weights, you usually minimize the sum of squared residuals (SSR) for all observations. This approach is called the method of ordinary least squares.")
                                st.info("The variation of actual responses occurs partly due to the dependence on the predictors. However, there is also an additional, systematic, inherent variance of the output. The coefficient of determination, denoted as R-squared, tells you which amount of variation in 𝑦 can be explained by the dependence on 𝐱 using the particular regression model. Larger 𝑅² indicates a better fit and means that the model can better explain the variation of the output with different inputs.")
                                st.error("You should, however, be aware of two problems that might follow the choice of the degree: underfitting and overfitting.\n"
                                "Underfitting occurs when a model can't accurately capture the dependencies among data, usually as a consequence of its own simplicity. It often yields a low 𝑅² with known data and bad generalization capabilities when applied with new data. Overfitting happens when a model learns both dependencies among data and random fluctuations. In other words, a model learns the existing data too well. Complex models, which have many features or terms, are often prone to overfitting. When applied to known data, such models usually yield high 𝑅². However, they often don't generalize well and have significantly lower 𝑅² when used with new data.")
                                st.success("The coefficients in our model indicate the extent to which we can expect the predicted variable chosen to change for a 1 unit change in each respective input variable. The p-values indicate the level of statistical significance (lower means more statistically significant). R-squared is a measure of how well the model explains variance in the data.")

                    # Spawn a new Quill editor
                    st.subheader("Notes on linear regression analysis")
                    regcontent = st_quill(placeholder="Write your notes here")

                    st.write("Linear regression took ", time.time() - start_time, "seconds to run")

            if modelchoice == "Logistic regression":

                st.subheader("Logistic regression")

                with st.spinner('Please wait while we conduct the multinomial logistic regression analysis'):

                    my_bar = st.progress(0)

                    time.sleep(10)

                    for percent_complete in range(100):
                        time.sleep(0.1)
                        my_bar.progress(percent_complete + 1)

                    start_time = time.time()

                    st.write("**This model looks at the probability of consumers preferring particular brands (i.e. brand is the dependent categorical variable). With future ingestion of new data the scope of choosing additional categorical dependent variables may increase.**")

                    ### From Jupyter - 3. Multinomial logistic regression

                    # Evaluate multinomial logistic regression model

                    with st.expander("Logistic regression assumptions"):

                        st.info("Basic assumptions that must be met for logistic regression include independence of errors, linearity in the logit for continuous variables, absence of multicollinearity, and lack of strongly influential outliers.")
                        st.info("There are some fundamental differences between logistic, and linear, regression, in terms of necessary conditions on the data. First, logistic regression does not require a linear relationship between the dependent and independent variables.  Second, the error terms (residuals) do not need to be normally distributed.  Third, homoscedasticity is not required.  Finally, the dependent variable in logistic regression is not measured on an interval or ratio scale.")
                        st.info("We should still have a large sample size (for normality), the dependent variable should be ordinal (or binary), and observations should be independent - i.e. repeated measurements or matched data is not used. Happily, these conditions are satisfied with our data.")
                        st.info("This will enable us to draw conclusions about the type of consumer that prefers each brand, based on experimentation.")
                        link = f'[{"See more"}]({"https://www.statology.org/assumptions-of-logistic-regression/"})'
                        st.markdown(link, unsafe_allow_html=True)

                    st.info("We can interpret the coefficients of the logistic regression as the probability of an observation fallinng into a particular class of dependent variable given changes in a set of independent, or input, variables. This means we can predict the potential outcome class for a relevant variable, if we know how other variables in the dataset change. This is what we do below, by predicting which brand a hypothetical consumer would prefer.")
                    st.write("")
                    st.write("Please set up a hypothetical consumer to determine their likely purchase habits")

                    age = st.number_input("Please enter consumer age")
                    gender = st.radio("Please select consumer gender", options=["Male", "Female"])
                    firstint = st.number_input("How many days ago was the consumer's first interaction?")
                    lastint = st.number_input("How many days ago was the consumer's last interaction?")
                    province = st.radio("Please select consumer province", ["Eastern Cape", "Free State", "Gauteng", "KwaZulu-Natal", "Limpopo", "Mpumalanga", "Northern Cape", "North West", "Western Cape"])
                    email = st.radio("Has the consumer opted in for email notifications?", options=["Yes", "No"])
                    sms = st.radio("Has the consumer opted in for sms notifications?", options=["Yes", "No"])
                    push = st.radio("Has the consumer opted in for push notifications?", options=["Yes", "No"])

                    if gender == "Male":
                        gender = 1
                    else:
                        gender=2

                    # Check if these correspond with the encoding
                    
                    if province == "Eastern Cape":
                        province=1
                    if province == "Free State":
                        province=2
                    if province == "Gauteng":
                        province=3
                    if province == "KwaZulu-Natal":
                        province=4
                    if province == "Limpopo":
                        province=5
                    if province == "Mpumalanga":
                        province=6
                    if province == "Northern Cape":
                        province=7
                    if province == "North West":
                        province=8
                    if province == "Western Cape":
                        province=9

                    if email == "Yes":
                        email = 1
                    else:
                        email = 2
                    if sms == "Yes":
                        sms = 1
                    else:
                        sms = 2
                    if push == "Yes":
                        push = 1
                    else:
                        push = 2

                    # Define dataset

                    # y = df['BRAND'].iloc[0:10000]
                    # X = df.iloc[0:10000, 1:-1] #subsamping for efficiency and speed
                    # dflogreg = df.drop(columns=['BRANDNAME'])
                    dflogreg = df
                    # print(dflogreg.head(5))
                    ylogreg = dflogreg['BRAND'].sample(10000)
                    Xlogreg = dflogreg.drop(columns=["BRANDNAME"]).sample(10000) #subsamping for efficiency and speed
                    varnames = dflogreg.columns.values.tolist()
                    Xlogreg, ylogreg = np.array(Xlogreg), np.array(ylogreg)

                    # Define the multinomial logistic regression model
                    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

                    # Define the model evaluation procedure
                    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=3, random_state=1)

                    # Evaluate the model and collect the scores
                    n_scores = cross_val_score(model, Xlogreg, ylogreg, scoring='accuracy', cv=cv, n_jobs=-1)
                    # report the model performance
                    st.write('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

                    # make a prediction with a multinomial logistic regression model

                    # define dataset
                    X_trainlogreg, X_testlogreg, y_trainlogreg, y_testlogreg = train_test_split(Xlogreg, ylogreg, test_size=0.25, random_state=42)

                    # define the multinomial logistic regression model
                    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

                    # fit the model on the training dataset
                    model.fit(X_trainlogreg, y_trainlogreg)

                    # define a single row of test data
                    # row = X_testlogreg[0,0:] # Previous approach
                    row = [0, 0, firstint, lastint, 0, 0, 0, gender, 0, 0, 0, 0, province, 0, email, sms, push, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                    # predict the class label
                    yhat = model.predict([row])
                    # summarize the predicted class
                    st.success('Predicted Class: %d' % yhat[0])
                    brandpreference = branddf.loc[branddf.BRAND == yhat[0],'BRAND NAME'].values[0]
                    st.success("Predicted brand consumer prefers: {}".format(brandpreference))
                    st.write(branddf) # Put this back in once the data has been sorted out

                    # brands = pd.Series(df['BRANDNAME'].drop_duplicates()).sort_values().tolist() # Faster than .unique()

                    brands = df['BRANDNAME'].unique()

                    brandselection = st.selectbox("Please select a brand to profile", brands, key="brandprofileselection")

                    # Average consumer profile per brand
                    # Pseudocode

                    # for brand in df[BRANDNAME.unique()]:
                    #     for col in df.columns():
                    #         brand.col.name = df[col].average
                    #         brandprofiledf.append(brand.col.name)

                    # df.groupby('group').agg({'a':['sum', 'max'], 
                    #                          'b':'mean', 
                    #                          'c':'sum', 
                    #                          'd': max_min})

                    st.subheader("Profile of average consumer for {}".format(brandselection))

                    # st.write(brandprofiledf.loc[brandprofiledf.BRANDNAME == brandselection])

                    consumerprofile = df.groupby('BRANDNAME').agg({'DAYS_LEFT_TO_ENGAGE':np.mean,
                                                                'GENDER':['mean'],
                                                                'AGE':['mean'],
                                                                'EMAIL_CONSENT':['mean'],
                                                                'SMS_CONSENT':['mean'],
                                                                'PUSH_CONSENT':['mean'],
                                                                }).round()

                    # filteredconsumerprofile = consumerprofile[brandselection]

                    st.write(consumerprofile.loc[brandselection])

                    st.write("**Predict a multinomial probability distribution for brands, based on whole dataset:**")
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

                    # Spawn a new Quill editor
                    st.subheader("Notes on logistic regression analysis")
                    logregcontent = st_quill(placeholder="Write your notes here")

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
                        def evaluate_model(model, Xlogreg, ylogreg):
                            # define the evaluation procedure
                            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
                            # evaluate the model
                            scores = cross_val_score(model, Xlogreg, ylogreg, scoring='accuracy', cv=cv, n_jobs=-1)
                            return scores
                        # get the models to evaluate
                        models = get_models()
                        # evaluate the models and store results
                        results, names = list(), list()
                        for name, model in models.items():
                            # evaluate the model and collect the scores
                            scores = evaluate_model(model, Xlogreg, ylogreg)
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

                with st.expander("Assumptions and operation of the model"):

                    st.write("The technique of cluster analysis aims to group similar observations in a dataset, such that observations in the same group are as similar to each other as possible, and similarly, observations in different groups are as different to each other as possible.")
                    st.warning("This allows us to get a sense of how many different groups with different characteristics and observable behaviour we have in our data, with respect to a chosen variable.")
                    st.info("Cluster analysis does this by minimizing the distance between observations (or the error involved in grouping them), by considering the mean of each potential group. Each group, by construct, has specific characteristics as it relates to the chosen variable.")
                    st.write("Cluster analysis in this case assumes each explanatory variable has the same within-group variance, with spherical variance, and that clusters are roughly similarly sized. Our data is sourced in a way that produces usable variance, and cluster size as reported can be used to further refine results by choosing different cluster sizes as a tuned hyperparameter.")

                ### From Jupyter - 4. Clustering - K-means

                # Perform K-means clustering for consumer segmentation

                # Size of the data set after removing NaN
                print(df2.shape)
                # Explore type of data and feature names
                #print(df2.sample(5))

                # Select explanatory variable

                # Choose brand

                brands = pd.Series(df2['BRAND'].drop_duplicates()).sort_values().tolist() # Faster than .unique()
                brandsmod = ["All"] + brands

                st.markdown("**Please choose brands. If option 'All' is selected, or no option is selected, the analysis will be run for all brands.**")
                st.info("Tip: it is instructive to compare analysis between brands for different variables.")
                # brandssel = []
                brandssel = st.multiselect("Please select brands of interest.", brandsmod, key="brands")

                if "All" not in brandssel:
                    df = df[df['BRANDNAME'].isin(brands)]

                # if brandssel == []:
                #     pass

                # print(df)

                # Select continuous variables for clustering - this is for age
                # X2 = df2.iloc[0:10000, 14:15] #subsamping for efficiency and speed
                clustervars = list(df.columns)
                chosenvar = st.selectbox("Please select variable for cluster analysis", clustervars)
                if chosenvar == "BRANDNAME":
                    chosenvar = "BRAND"
                chosenvarindex = df.columns.get_loc(chosenvar)

                # X2 = df.iloc[:, 10:11].sample(10000) #subsamping for efficiency and speed
                X2 = df.iloc[:, chosenvarindex:chosenvarindex+1].sample(10000, replace=True) # Subsampling for efficiency and speed

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

                st.info("You can use the elbow method to refine the analysis, by informing the number of clusters to use below. You should choose the number of clusters based on where the graph sensibly starts to taper/level off, meaning that additional clusters add little additional explanatory power to the model.")

                # 2. Silhouette method
                sil = []
                kmax = 10
                nclusters = []

                for k in range(2, kmax+1):
                    kmeans = KMeans(n_clusters = k).fit(X2)
                    labels = kmeans.labels_
                    silscore = silhouette_score(X2, labels, metric = 'euclidean')
                    sil.append(silscore)
                    if silscore >= max(sil):
                        nclusters = k
                        print(nclusters)

                # Calculate best silhouette score
                print(max(sil))

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

                nclustersused = 0

                silrun = st.button("Run cluster analysis with maximum silhouette score")

                if silrun == True:

                    nclustersused = nclusters

                    # Re-cluster with max silhouette score - this will become dynamic in the app - used to be 4 here

                    X2new = X2.copy()
                    kmnew = KMeans(n_clusters=nclusters, init='k-means++', 
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

                    for i in range(0, nclusters):
                        countclusters = Xnew2.apply(lambda x: True if x['CLUSTERS'] == i else False , axis=1)
                        # Count number of True in series
                        numOfRows = len(countclusters[countclusters == True].index)
                        st.write('Number of Rows in dataframe in which cluster = {} : '.format(i), numOfRows)

                    Xnew = X2.copy()
                    Xnew["CLUSTERS"] = km.labels_
                    Xnew.sample(8, random_state=0)

                    # Xnew2.drop_duplicates(subset=['brand'])
                    Xnew2 = Xnew2[~Xnew2.index.duplicated(keep='first')]
                    Xnew2.to_csv('Xnew2.csv')

                    # Plot the following variables and their clusters
                    var = [chosenvar]

                    # Plot using seaborn

                    # fig, ax = plt.subplots(figsize=(7, 3))
                    fig = sns.pairplot(Xnew2, vars=var, hue="CLUSTERS", palette=sns.color_palette("hls", nclusters), height=5)
                    buf = BytesIO()
                    fig.savefig(buf, format="png")
                    #st.pyplot(fig)
                    st.image(buf)
                    #df.head()
                    #df = df.dropna()

                    if nclustersused != 0:

                        st.success("Our cluster analysis indicates that there are " + str(nclustersused) + " clusters in the data for " + chosenvar + " for the brands selected. Each group has specific characteristics and behaviour that be intuited (you can assign meanings to each group based on hypotheses), and further investigation using data exploration, correlation analysis, and linear/logistic regression analysis can provide further insight for " + chosenvar + " clusters in these brands, as compared to other brands or the whole dataset.")

                st.write("Or choose the number of clusters based on the elbow method, enter this in the slider below, and click the 'Conduct analysis' button")
                numclus = st.slider("Please choose number of clusters", min_value=1, max_value=nclusters)
                ownclus = st.button("Conduct analysis")
                if ownclus == True:

                    nclustersused = numclus
                    
                    # Re-cluster with chosen number of clusters

                    X2new = X2.copy()
                    kmnew = KMeans(n_clusters=numclus, init='k-means++', 
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

                    for i in range(0, numclus):
                        countclusters = Xnew2.apply(lambda x: True if x['CLUSTERS'] == i else False , axis=1)
                        # Count number of True in series
                        numOfRows = len(countclusters[countclusters == True].index)
                        st.write('Number of Rows in dataframe in which cluster = {} : '.format(i), numOfRows)

                    Xnew = X2.copy()
                    Xnew["CLUSTERS"] = km.labels_
                    Xnew.sample(8, random_state=0)

                    Xnew2 = Xnew2[~Xnew2.index.duplicated(keep='first')]
                    Xnew2.to_csv('Xnew2.csv')

                    # Plot the following variables and their clusters
                    var = [chosenvar]

                    # Plot using seaborn

                    # fig, ax = plt.subplots(figsize=(7, 3))
                    fig = sns.pairplot(Xnew2, vars=var, hue="CLUSTERS", palette=sns.color_palette("hls", numclus), height=5)
                    buf = BytesIO()
                    fig.savefig(buf, format="png")
                    #st.pyplot(fig)
                    st.image(buf)
                    #df.head()
                    #df = df.dropna()

                    if nclustersused != 0:

                        st.success("You chose " + str(nclustersused) + " clusters in the data for " + chosenvar + " for the brands selected. Each group has specific characteristics and behaviour that be intuited (you can assign meanings to each group based on hypotheses), and further investigation using data exploration, correlation analysis, and linear/logistic regression analysis can provide further insight for " + chosenvar + " clusters in these brands, as compared to other brands or the whole dataset.")

                # Optional - log transformations

                ### End

                st.write("")

                # Spawn a new Quill editor
                st.subheader("Notes on cluster analysis")
                clustercontent = st_quill(placeholder="Write your notes here")

                st.write("Running the cluster analysis using K-means took ", time.time() - start_time, "seconds to run")

            if modelchoice == "ANOVA":

                st.warning("This model takes a long time to run. Please be patient.")

                with st.spinner('Please wait while we run the ANOVA analysis'):

                    my_bar = st.progress(0)

                    time.sleep(10)

                    for percent_complete in range(100):
                        time.sleep(0.01)
                        my_bar.progress(percent_complete + 1)

                    start_time = time.time()

                    ### From Jupyter - 5. ANOVA

                    with st.expander("Explanation of ANOVA analysis"):
                        st.write("Analysis of Variance (ANOVA) is used to test how different sample groups within a dataset compare to each other. In our case, it will be used to test either (a) how one (or a group) of brands compare to the overall dataset, or (b) how two different groups of brands compare to each other, with respect to a chosen variable.")
                        st.write("Groups mean differences inferred by analyzing variances. ANOVA uses variance-based F test to check the group mean equality.")
                        st.info("The null hypothesis that group means are equal is tested with an F-test for all groups, followed by post-hoc tests to see individual group differences.")
                    with st.expander("Assumptions of ANOVA"):
                        st.warning("1. Residuals (experimental error) are approximately normally distributed."
                        "\n\n2. Homoscedasticity or Homogeneity of variances (variances are equal between treatment groups)."
                        "\n\n3. Observations are sampled independently from each other."
                        "\n\n4. The dependent variable should be continuous. If the dependent variable is ordinal or rank (e.g. Likert item data), it is more likely to violate the assumptions of normality and homogeneity of variances.")

                    data = df.head(10000)

                    brands = pd.Series(data['BRANDNAME'].astype(str).drop_duplicates()).sort_values().tolist() # Faster than .unique()
                    brandsmod = ["All"] + brands

                    st.markdown("**Please choose brands of interest. If option 'All' is selected, or no option is selected, the analysis will be run for all brands.**")
                    st.info("Tip: it is instructive to compare analysis between brands for different variables.")
                    # brandssel = []
                    brandssel = st.multiselect("Please select brands of interest.", brandsmod, key="brands")

                    st.markdown("**Please choose second brands of interest.**")
                    st.info("Tip: it is instructive to compare analysis between brands for different variables.")
                    # brandssel = []
                    brandssel2 = st.multiselect("Please select second (comparison) brands of interest.", brands, key="brands2") # No "All" option for second one

                    st.info("NB: If one brand is selected, ANOVA is run for the variable of choice below for that brand against all brands. If all brands are selected, then ANOVA is run for the chosen variable against brand choice for all brands. If two brands or brand groups are selected, the groups are compared to each other.")
                    st.warning("If no interaction variable is chosen, ANOVA is run purely with the relationship between the chosen variable and brand choice. If an interaction variable is chosen, ANOVA is run incorporating both the direct effects of the variables on brand choice, but also the interaction between the two explanatory variables.")

                    # dfanv = df.drop(['BRAND', 'BRANDNAME'], axis = 1, inplace = False)
                    anovavars = list(df.columns)
                    chosenvar = st.selectbox("Please select variable for ANOVA analysis", anovavars)
                    if chosenvar == "BRANDNAME":
                        chosenvar = "BRAND"
                    chosenvarindex = df.columns.get_loc(chosenvar)

                    # groupvar = st.selectbox("Please choose a variable by which to group", anovavars)
                    # if groupvar == "BRANDNAME":
                    #     groupvar = "BRAND"
                    # groupvarindex = df.columns.get_loc(groupvar)

                    interacvar = st.selectbox("Please select interaction variable for ANOVA analysis", anovavars)
                    if interacvar == "BRANDNAME":
                        interacvar = "BRAND"
                    interacvarindex = df.columns.get_loc(interacvar)

                    print(brandssel)
                    print(brandssel2)

                    if brandssel != [] and brandssel2 == []:

                        if "All" not in brandssel:
                            data = data[data['BRANDNAME'].isin(brandssel)]

                        if interacvar == []:

                            #Create a boxplot
                            fig = data.boxplot(chosenvar, by='BRAND', figsize=(18, 18)) # Dynamic to variables of choice
                            fig.set_xticklabels(fig.get_xticklabels(),rotation=270)

                            ctrl = data[chosenvar][data.BRAND == 'Castle Lager'] # Dynamic to brand and variables of choice

                            grps = pd.unique(data.BRAND.values)
                            d_data = {grp:data[chosenvar][data.BRAND == grp] for grp in grps}

                            k = len(pd.unique(data.BRAND))  # number of conditions
                            N = len(data.values)  # conditions times participants
                            n = data.groupby('BRAND').size() #Participants in each condition

                            buf = BytesIO()
                            fig.figure.savefig(buf, format="png")
                            st.image(buf)

                            # Alternate boxplot

                            ax = sns.boxplot(x=chosenvar, y='BRAND', data=data, color='#99c2a2')
                            ax = sns.swarmplot(x=chosenvar, y="BRAND", data=data, color='#7d0013')

                            buf = BytesIO()
                            ax.figure.savefig(buf, format="png")
                            st.image(buf)

                            # ANOVA using statsmodels

                            print(chosenvar)

                            # Start of block comment

                            # mod = ols(chosenvar + '~ BRAND',
                            #                 data=data).fit()
                                            
                            # aov_table = sm.stats.anova_lm(mod, typ=2)

                            # # aovrm = AnovaRM(df, depvar='BRAND', subject=chosenvar, within=[chosenvar], aggregate_func=mean)

                            # # res = aovrm.fit()

                            # aov_table = sm.stats.anova_lm(mod, typ=2)

                            # st.write("ANOVA table of " + chosenvar + " appears below, grouped by brand. This shows the statistical significance of " + chosenvar + " on the choice of brands.")
                            # st.table(aov_table)

                            # # Pairwise comparisons

                            # print(data.head(5))

                            # pair_t = mod.t_test_pairwise('BRAND') # can add optional method = "sidak" or "bonferroni" here
                            # st.write("The impact of " + chosenvar + " on the different brand choices appears below:")
                            # st.write(pair_t.result_frame)

                            # End of block comment

                            res = stat()
                            res.anova_stat(df=data, res_var='BRAND', anova_model='BRAND ~ C(' + chosenvar + ')')
                            st.write(res.anova_summary)

                            st.success("**If the p-value (PR(>F)) is smaller than 0.05, we can reject the null hypothesis and conclude that the chosen group is significantly differently impacted by the chosen variables, compared to all brands.**")

                            with st.expander("Explaining the model pairwise comparison results"):
                                st.info("If we conclude that there are significant differences between groups, we do not know which groups are different. To know the pairs of significant different treatments, we will perform multiple pairwise comparison (post hoc comparison) analysis for all unplanned comparison using Tukey's honestly significantly differenced (HSD) test."
                                "\n\nThis will tell us which pairwise brand comparisons are significantly different, according to the below table.")
                            
                                st.table(branddf)

                            res.tukey_hsd(df=data, res_var='BRAND', xfac_var=chosenvar, anova_model='BRAND ~ C(' + chosenvar + ')')
                            st.write(res.tukey_summary)

                            st.success("In the pairwise comparisons, those with p-values less than 0.05 are significantly different to each other.")

                            # res.anova_std_residuals are standardized residuals obtained from ANOVA (check above)
                            sm.qqplot(res.anova_std_residuals, line='45')
                            plt.xlabel("Theoretical Quantiles")
                            plt.ylabel("Standardized Residuals")
                            st.pyplot(plt)

                            # histogram
                            plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k')
                            plt.ylim([0, 1.3*max(res.anova_model_out.resid)]) 
                            plt.xlabel("Residuals")
                            plt.ylabel('Frequency')
                            st.pyplot(plt)

                            st.success("We can check to see if the standardized residuals lie around the 45-degree line, and that the distribution histogram is distributed normally, to see whether residuals are distributed normally, and our conclusions are valid, especially for larger sample sizes.")

                            w, pvalue = stats.shapiro(data[chosenvar])
                            st.info("The Shapiro-Wilk test can be used to check the normal distribution of residuals. Null hypothesis: data is drawn from the normal distribution.")
                            st.write("Shapiro-Wilk test results: ", w, pvalue)
                            if pvalue<0.05:
                                st.error("The data indicates some processing may be required for more robust interpretation of results, as data is not normally distributed.")
                            elif pvalue>0.05:
                                st.success("We fail to reject the null hypothesis. Data approximates the normal distribution.")

                            st.info("We can use Bartlett's test to check the Homogeneity of variances. Null hypothesis: samples from populations have equal variances.")

                            w, pvalue = stats.bartlett(data[chosenvar], data['BRAND'])
                            st.write("Bartlett test results: ", w, pvalue)

                            if pvalue<0.05:
                                st.success("The data indicates some processing may be required for more robust interpretation of results, as samples have unequal variances.")
                            elif pvalue>0.05:
                                st.success("We fail to reject the null hypothesis. Data samples exhibit equal variance.")
                            
                            # # if you have a stacked table, you can use bioinfokit v1.0.3 or later for the bartlett's test
                            # res = stat()
                            # res.bartlett(df=data, res_var='BRAND', xfac_var=df[chosenvar])
                            # st.write(res.bartlett_summary)

                            res = stat()
                            res.levene(df=data, res_var='BRAND', xfac_var=chosenvar)
                            st.write(res.levene_summary)

                            ### End
                        
                        elif interacvar != []:

                            fig = sns.boxplot(x=chosenvar, y="BRAND", hue=interacvar, data=data, palette="Set3")

                            res = stat()
                            res.anova_stat(df=data, res_var='BRAND', anova_model='BRAND~C('+chosenvar+')+C('+interacvar+')+C('+chosenvar+'):C('+interacvar+')')
                            st.write(res.anova_summary)

                            st.success("**If the p-value is significant, we can conclude that either the explanatory variable, the interaction variable, or both through their interaction significantly impacts brand choice for consumers, in the choice between the two chosen groups.**")

                            st.info("We can also visualize the interaction plot (also called profile plot) for interaction effects:")

                            fig = interaction_plot(x=data.iloc[chosenvarindex], trace=data.iloc[interacvarindex], response=data['BRAND'])
                            st.pyplot(fig)

                            st.info("The interaction plot helps to visualize the means of the response of the two factors on one graph.")

                            st.success("The interaction effect is significant between the two interaction variables if the lines are not parallel (roughly parallel factor lines indicate no interaction - additive model). This interaction is also called ordinal interaction if the lines do not cross each other.")

                            st.info("For a more reliable conclusion of the interaction plot, it should be verified with the F-test for interaction. We do this through multiple pairwise comparisons to find out which groups of explanatory variables are most significant.")

                            res = stat()
                            # for main effect main variable
                            res.tukey_hsd(df=data, res_var='BRAND', xfac_var=chosenvar, anova_model='BRAND~C('+chosenvar+')+C('+interacvar+')+C('+chosenvar+'):C('+interacvar+')')
                            st.write("The effect of " + chosenvar + ":")
                            st.write(res.tukey_summary)

                            # for main effect secondary variable
                            res.tukey_hsd(df=data, res_var='BRAND', xfac_var=interacvar, anova_model='BRAND~C('+chosenvar+')+C('+interacvar+')+C('+chosenvar+'):C('+interacvar+')')
                            st.write("The effect of " + interacvar + ":")
                            st.write(res.tukey_summary)

                            # copy_data = data.iloc[chosenvarindex] + data.iloc[interacvarindex] + data.iloc[0]

                            # for interaction effect between chosen variable and interaction variable
                            res.tukey_hsd(df=data.head(10), res_var='BRAND', xfac_var=[chosenvar, interacvar], anova_model='BRAND~C('+chosenvar+')+C('+interacvar+')+C('+chosenvar+'):C('+interacvar+')')
                            st.write("The effect of interaction between " + chosenvar + " and " + interacvar + ":")
                            st.write(res.tukey_summary)

                            st.info("We can use visual approaches, Bartlett's or Levene's, and the Shapiro-Wilk test, to validate the assumptions for homogeneity of variances and normal distribution of residuals.")

                            # Shapiro-Wilk test
                            import scipy.stats as stats
                            w, pvalue = stats.shapiro(res.anova_model_out.resid)
                            st.write("Results of the Shapiro-Wilk test: ", w, pvalue)
                            if pvalue<0.05:
                                st.error("The data indicates some processing may be required for more robust interpretation of results, as data is not normally distributed.")
                            elif pvalue>0.05:
                                st.success("We fail to reject the null hypothesis. Data approximates the normal distribution.")
                            st.write("We should further look for the residual plots and histograms.")
                
                            # res.anova_std_residuals are standardized residuals obtained from two-way ANOVA (check above)
                            sm.qqplot(res.anova_std_residuals, line='45')
                            plt.xlabel("Theoretical Quantiles")
                            plt.ylabel("Standardized Residuals")
                            st.pyplot(plt)

                            # histogram
                            plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k')
                            plt.ylim([0, 1.3*max(res.anova_model_out.resid)])
                            plt.xlabel("Residuals")
                            plt.ylabel('Frequency')
                            st.pyplot(plt)

                            st.success(" In the residual plot, if the standardized residuals lie around the 45-degree line, it suggests that the residuals are approximately normally distributed. Besides, the histogram shows whether residuals are normally distributed.")

                            st.info("Note: The ANOVA model is remarkably robust to the violation of normality assumption, which means that it will have a non-significant effect on Type I error rate and p values will remain reliable as long as there are no outliers.")

                            st.info("We will use Levene's test to check the assumption of homogeneity of variances")

                            res = stat()
                            res.levene(df=data, res_var='BRAND', xfac_var=[chosenvar, interacvar])
                            st.write(res.levene_summary)

                            st.warning("If the p-value is non-significant (> 0.05), we fail to reject the null hypothesis and conclude that our samples have equal variances.")
                    
                    elif brandssel != [] and brandssel2 != []:

                        if "All" not in brandssel:
                            data = data[data['BRANDNAME'].isin(brands)]

                        data2 = data[data['BRANDNAME'].isin(brandssel2)]

                        data3 = data + data2

                        if interacvar == []:

                            #Create a boxplot
                            fig = data.boxplot(chosenvar, by='BRAND', figsize=(18, 18)) # Dynamic to variables of choice
                            fig.set_xticklabels(fig.get_xticklabels(),rotation=270)

                            ctrl = data[chosenvar][data.BRAND == 'Castle Lager'] # Dynamic to brand and variables of choice

                            grps = pd.unique(data.BRAND.values)
                            d_data = {grp:data[chosenvar][data.BRAND == grp] for grp in grps}

                            k = len(pd.unique(data.BRAND))  # number of conditions
                            N = len(data.values)  # conditions times participants
                            n = data.groupby('BRAND').size()[0] #Participants in each condition

                            buf = BytesIO()
                            fig.figure.savefig(buf, format="png")
                            st.image(buf)

                            # ANOVA using statsmodels

                            print(chosenvar)

                            # Start of block comment

                            # mod = ols(chosenvar + '~ BRAND',
                            #                 data=data).fit()
                                            
                            # aov_table = sm.stats.anova_lm(mod, typ=2)

                            # # aovrm = AnovaRM(df, depvar='BRAND', subject=chosenvar, within=[chosenvar], aggregate_func=mean)

                            # # res = aovrm.fit()

                            # aov_table = sm.stats.anova_lm(mod, typ=2)

                            # st.write("ANOVA table of " + chosenvar + " appears below, grouped by brand. This shows the statistical significance of " + chosenvar + " on the choice of brands.")
                            # st.table(aov_table)

                            # # Pairwise comparisons

                            # print(data.head(5))

                            # pair_t = mod.t_test_pairwise('BRAND') # can add optional method = "sidak" or "bonferroni" here
                            # st.write("The impact of " + chosenvar + " on the different brand choices appears below:")
                            # st.write(pair_t.result_frame)

                            # End of block comment

                            res = stat()
                            res.anova_stat(df=data3, res_var='BRAND', anova_model='BRAND ~ C(' + chosenvar + ')')
                            st.write(res.anova_summary)

                            st.success("**If the p-value (PR(>F)) is smaller than 0.05, we can reject the null hypothesis and conclude that the chosen group is significantly differently impacted by the chosen variables, compared to all brands.**")

                            with st.expander("Explaining the model pairwise comparison results"):
                                st.info("If we conclude that there are significant differences between groups, we do not know which groups are different. To know the pairs of significant different treatments, we will perform multiple pairwise comparison (post hoc comparison) analysis for all unplanned comparison using Tukey's honestly significantly differenced (HSD) test."
                                "\n\nThis will tell us which pairwise brand comparisons are significantly different, according to the below table.")
                            
                                st.table(branddf)

                            res.tukey_hsd(df=data3, res_var='BRAND', xfac_var=chosenvar, anova_model='BRAND ~ C(' + chosenvar + ')')
                            st.write(res.tukey_summary)

                            st.success("In the pairwise comparisons, those with p-values less than 0.05 are significantly different to each other.")

                            # res.anova_std_residuals are standardized residuals obtained from ANOVA (check above)
                            sm.qqplot(res.anova_std_residuals, line='45')
                            plt.xlabel("Theoretical Quantiles")
                            plt.ylabel("Standardized Residuals")
                            st.pyplot(plt)

                            # histogram
                            plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k')
                            plt.ylim([0, 1.3*max(res.anova_model_out.resid)])
                            plt.xlabel("Residuals")
                            plt.ylabel('Frequency')
                            st.pyplot(plt)

                            st.success("We can check to see if the standardized residuals lie around the 45-degree line, and that the distribution histogram is distributed normally, to see whether residuals are distributed normally, and our conclusions are valid, especially for larger sample sizes.")

                            w, pvalue = stats.shapiro(data3[chosenvar])
                            st.info("The Shapiro-Wilk test can be used to check the normal distribution of residuals. Null hypothesis: data is drawn from the normal distribution.")
                            st.write("Shapiro-Wilk test results: ", w, pvalue)

                            if pvalue<0.05:
                                st.error("The data indicates some processing may be required for more robust interpretation of results, as data is not normally distributed.")
                            elif pvalue>0.05:
                                st.success("We fail to reject the null hypothesis. Data approximates the normal distribution.")

                            st.info("We can use Bartlett's test to check the Homogeneity of variances. Null hypothesis: samples from populations have equal variances.")

                            w, pvalue = stats.bartlett(data3[chosenvar], data3['BRAND'])
                            st.write("Bartlett test results: ", w, pvalue)

                            if pvalue<0.05:
                                st.success("The data indicates some processing may be required for more robust interpretation of results, as samples have unequal variances.")
                            elif pvalue>0.05:
                                st.success("We fail to reject the null hypothesis. Data samples exhibit equal variance.")
                            
                            # # if you have a stacked table, you can use bioinfokit v1.0.3 or later for the bartlett's test
                            # res = stat()
                            # res.bartlett(df=data, res_var='BRAND', xfac_var=df[chosenvar])
                            # st.write(res.bartlett_summary)

                            # res = stat()
                            # res.levene(df=data, res_var='BRAND', xfac_var=df[chosenvar])
                            # st.write(res.levene_summary)

                        elif interacvar != []:

                            fig = sns.boxplot(x=chosenvar, y="BRAND", hue=interacvar, data=data, palette="Set3")

                            res = stat()
                            res.anova_stat(df=data, res_var='BRAND', anova_model='BRAND~C('+chosenvar+')+C('+interacvar+')+C('+chosenvar+'):C('+interacvar+')')
                            st.write(res.anova_summary)

                            st.success("**If the p-value is significant, we can conclude that either the explanatory variable, the interaction variable, or both through their interaction significantly impacts brand choice for consumers, in the choice between the two chosen groups.**")

                            st.info("We can also visualize the interaction plot (also called profile plot) for interaction effects:")

                            fig = interaction_plot(x=data.iloc[chosenvarindex], trace=data.iloc[interacvarindex], response=data['BRAND'])
                            st.pyplot(fig)

                            st.info("The interaction plot helps to visualize the means of the response of the two factors on one graph.")
                            
                            st.success("The interaction effect is significant between the Genotype and years if the lines are not parallel (roughly parallel factor lines indicate no interaction - additive model). This interaction is also called ordinal interaction if the lines do not cross each other.")

                            st.info("For a more reliable conclusion of the interaction plot, it should be verified with the F test for interaction. We do this through multiple pairwise comparisons to find out which groups of explanatory variables are most significant.")

                            res = stat()
                            # for main effect main variable
                            res.tukey_hsd(df=data, res_var='BRAND', xfac_var=chosenvar, anova_model='BRAND~C('+chosenvar+')+C('+interacvar+')+C('+chosenvar+'):C('+interacvar+')')
                            st.write("The effect of " + chosenvar + ":")
                            st.write(res.tukey_summary)

                            # for main effect secondary variable
                            res.tukey_hsd(df=data, res_var='BRAND', xfac_var=interacvar, anova_model='BRAND~C('+chosenvar+')+C('+interacvar+')+C('+chosenvar+'):C('+interacvar+')')
                            st.write("The effect of " + interacvar + ":")
                            st.write(res.tukey_summary)

                            # for interaction effect between chosen variable and interaction variable
                            res.tukey_hsd(df=data, res_var='BRAND', xfac_var=[chosenvar, interacvar], anova_model='BRAND~C('+chosenvar+')+C('+interacvar+')+C('+chosenvar+'):C('+interacvar+')')
                            st.write("The effect of interaction between " + chosenvar + " and " + interacvar + ":")
                            st.write(res.tukey_summary)

                            st.info("We can use visual approaches, Bartlett's or Levene's, and the Shapiro-Wilk test, to validate the assumptions for homogeneity of variances and normal distribution of residuals.")

                            # Shapiro-Wilk test
                            import scipy.stats as stats
                            w, pvalue = stats.shapiro(res.anova_model_out.resid)
                            st.write("Results of the Shapiro-Wilk test: ", w, pvalue)
                            if pvalue<0.05:
                                st.error("The data indicates some processing may be required for more robust interpretation of results, as data is not normally distributed.")
                            elif pvalue>0.05:
                                st.success("We fail to reject the null hypothesis. Data approximates the normal distribution.")
                            st.write("We should further look for the residual plots and histograms.")
                
                            # res.anova_std_residuals are standardized residuals obtained from two-way ANOVA (check above)
                            sm.qqplot(res.anova_std_residuals, line='45')
                            plt.xlabel("Theoretical Quantiles")
                            plt.ylabel("Standardized Residuals")
                            st.pyplot(plt)

                            # histogram
                            plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k')
                            plt.ylim([0, 1.3*max(res.anova_model_out.resid)])
                            plt.xlabel("Residuals")
                            plt.ylabel('Frequency')
                            st.pyplot(plt)

                            st.success(" In the residual plot, if the standardized residuals lie around the 45-degree line, it suggests that the residuals are approximately normally distributed. Besides, the histogram shows whether residuals are normally distributed.")

                            st.info("Note: The ANOVA model is remarkably robust to the violation of normality assumption, which means that it will have a non-significant effect on Type I error rate and p values will remain reliable as long as there are no outliers.")

                            st.info("We will use Levene's test to check the assumption of homogeneity of variances")

                            res = stat()
                            res.levene(df=data, res_var='BRAND', xfac_var=[chosenvar, interacvar])
                            st.write(res.levene_summary)

                            st.warning("If the p-value is non-significant (> 0.05), we fail to reject the null hypothesis and conclude that our samples have equal variances.")

                        ### End

                        st.write("")

                        # Spawn a new Quill editor
                        st.subheader("Notes on ANOVA analysis")
                        anovacontent = st_quill(placeholder="Write your notes here")

                        st.write("ANOVA analysis took ", time.time() - start_time, "seconds to run")

            if modelchoice == "Principal component analysis":

                with st.spinner('Please wait while we conduct principal component analysis'):

                    my_bar = st.progress(0)

                    time.sleep(1)

                    for percent_complete in range(100):
                        time.sleep(0.01)
                        my_bar.progress(percent_complete + 1)

                    start_time = time.time()

                    st.info("Principal component analysis is a method to reduce dimensionality, or the number of variables required for a robust analysis, while preserving as much information and explanatory power as possible.")

                    # Choose predicted variable - this will become dynamic in the app
                    regvars = list(df.columns)
                    chosenvar = st.selectbox("Please select variable for analysis", regvars)

                    ### From Jupyter - Principal component analysis

                    # Initially, visualize the important data features

                    # Scale the features
                    # Separating out the features
                    # x = df.iloc[:, 1:-1].sample(10000).values #subsampling for efficiency and speed
                    x = df.drop([chosenvar, 'BRANDNAME'], axis=1).sample(1000)
                    # Separating out the target
                    # y = df.iloc[:,0].sample(10000).values #subsampling for efficiency and speed
                    y = df[chosenvar].sample(1000)

                    print(x.shape, y.shape)
                    
                    # Initial linear regression
                    x, y = np.array(x), np.array(y)
                    x = sm.add_constant(x)
                    model = sm.OLS(y, x)
                    results = model.fit()
                    st.write("Linear regression results give an indication of which factors are most important:")
                    st.write(results.summary())
                    st.write("")
                    st.write('Predicted response:', results.fittedvalues, sep='\n')

                    # Standardizing the features
                    x = StandardScaler().fit_transform(x)

                    # Dimensionality reduction
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=10)
                    principalComponents = pca.fit_transform(x)
                    principalDf = pd.DataFrame(data = principalComponents, 
                        columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5', 'principal component 6', 'principal component 7', 'principal component 8', 'principal component 9', 'principal component 10'])
                    # Concatenate DF across axis 1
                    finalDf = pd.concat([principalDf, df[chosenvar]], axis = 1)
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
                        indicesToKeep = finalDf[chosenvar] == target
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

                # pca = st.button("Click to see how PCA can speed up machine learning and to run a new regression model")

                # if pca == True:

                #     # st.session_state.pcasession = 'True'

                #     with st.spinner('Please wait while we conduct a new linear regression using the principal components'):

                #         my_bar = st.progress(0)

                #         time.sleep(1)

                #         for percent_complete in range(100):
                #             time.sleep(0.1)
                #             my_bar.progress(percent_complete + 1)

                #         start_time = time.time()

                    st.subheader("We can now see how PCA can speed up machine learning by running a new regression model")

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
                    y = df[chosenvar].sample(750)

                    print(y.shape)
                    print(train_X.shape)

                    # Define predictor variables
                    # x = train_X

                    # train_X, y = np.array(train_X), np.array(y.sample(750))

                    train_X = sm.add_constant(train_X)

                    model = sm.OLS(y, train_X)

                    results = model.fit()

                    st.subheader("PCA regression results:")

                    st.write(results.summary())

                    st.write("")

                    st.write('\nPredicted response:', results.fittedvalues, sep='\n') # Or print('predicted response:', results.predict(x), sep='\n')

                    st.write("")

                    st.write("Conducting a new linear regression with principal components took ", time.time() - start_time, "seconds to run")

                    st.success("Now that we know how many components the model suggests, take a look at the table of explained variance, and interpret the effect on the objective variable of the relevant number of features/variables exhibiting the highest correlation with the objective variable.")

                    link = "https://builtin.com/data-science/step-step-explanation-principal-component-analysis"
                    st.warning("Note that this interpretation is only an approximation, as the true interpretation of the component contributions are as linear combinations of features, but this is a reasonable first step in determining important variables. See this [link](%s) for more." %link)
                    
                    # Spawn a new Quill editor
                    st.subheader("Notes on principal component analysis")
                    pcacontent = st_quill(placeholder="Write your notes here")
                            
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

                    st.info("Conjoint analysis is traditionally a method for determining what features of a product are most important to consumers. We extend the usage here to enable us to determine what factors are most important in determining a consumer's choice for a particular brand purchase.")

                    brands = pd.Series(branddf['BRAND NAME'].drop_duplicates()).sort_values().tolist() # Faster than .unique()

                    selbrand = st.selectbox("Please select a brand to analyze", brands)

                    # brandcode = branddf.query('BRAND NAME == %s' % str(selbrand))['BRAND']
                    brandcode = branddf.loc[branddf['BRAND NAME'] == selbrand, 'BRAND'].item()

                    y = df2.BRAND.apply(lambda x: brandcode if selbrand in x else 0).head(10000) # Dynamic to brand, subsampling for speed and efficiency
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
                    plt.title('Fcator importances') # Used to be Part Worth
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

                    st.success("The results of the conjoint analysis show us which factors should be paid particular attention, in framing a consumer's choice of purchase for the chosen brand, " + selbrand + ".")

                    # Spawn a new Quill editor
                    st.subheader("Notes on conjoint analysis")
                    conjointcontent = st_quill(placeholder="Write your notes here")

                    st.write("Running the conjoint analysis took ", time.time() - start_time, "seconds to run")

            if modelchoice == "Neural networks":

                st.warning("This model can take a long time to run. Please be patient.")

                with st.spinner('Please wait while we conduct the neural networks analysis'):

                    my_bar = st.progress(0)

                    time.sleep(10)

                    for percent_complete in range(100):
                        time.sleep(0.01)
                        my_bar.progress(percent_complete + 1)

                    start_time = time.time()

                    st.info("This neural networks model examines the effect of gender, age, and city of consumer on brand preference. It is a work in progress and will be expanded soon.")

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

                    # Find the min and max of the desired feature and set up a sequence
                    min_feature = min(X.loc[:,"GENDER"]) # Dynamic to desired feature
                    max_feature = max(X.loc[:,"GENDER"]) # Dynamic to desired feature
                    seq = np.linspace(start=min_feature,stop=max_feature,num=50)

                    # Set up a list of moving features
                    to_predict = []
                    for result in seq:
                        X_design_vec.loc[0,"GENDER"] = result # Dynamic to desired feature
                        to_predict.append(X_design_vec.copy())

                    # Convert back to dataframe
                    to_predict = pd.concat(to_predict)

                    # Scale and predict
                    to_predict = scaler.transform(to_predict)
                    predictions = clf.predict(to_predict)

                    # Plot 
                    plt.plot(seq,predictions)
                    plt.xlabel("Gender") # Dynamic to desired feature
                    plt.ylabel("Brand") # Dynamic to desired feature
                    plt.title("Response vs Gender") # Dynamic to desired feature
                    plt.show()

                    buf3 = BytesIO()
                    fig.savefig(buf3, format="png")
                    st.image(buf3)

                    # Find the min and max of the desired feature and set up a sequence
                    min_feature = min(X.loc[:,"CITY"]) # Dynamic to desired feature
                    max_feature = max(X.loc[:,"CITY"]) # Dynamic to desired feature
                    seq = np.linspace(start=min_feature,stop=max_feature,num=50)

                    # Set up a list of moving features
                    to_predict = []
                    for result in seq:
                        X_design_vec.loc[0,"CITY"] = result # Dynamic to desired feature
                        to_predict.append(X_design_vec.copy())

                    # Convert back to dataframe
                    to_predict = pd.concat(to_predict)

                    # Scale and predict
                    to_predict = scaler.transform(to_predict)
                    predictions = clf.predict(to_predict)

                    # Plot 
                    plt.plot(seq,predictions)
                    plt.xlabel("City") # Dynamic to desired feature
                    plt.ylabel("Brand") # Dynamic to desired feature
                    plt.title("Response vs City") # Dynamic to desired feature
                    plt.show()

                    buf4 = BytesIO()
                    fig.savefig(buf4, format="png")
                    st.image(buf4)

                    st.write("")

                    # Spawn a new Quill editor
                    st.subheader("Notes on neural network analysis")
                    nncontent = st_quill(placeholder="Write your notes here")

                    st.write("Running the neural networks model took ", time.time() - start_time, "seconds to run")

            if modelchoice == "Decision trees":

                with st.spinner('Please wait while we conduct the decision tree analysis'):

                    my_bar = st.progress(0)

                    time.sleep(10)

                    for percent_complete in range(100):
                        time.sleep(0.01)
                        my_bar.progress(percent_complete + 1)

                    start_time = time.time()

                    st.info("A decision tree is a decision support tool that uses a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility. It is one way to display an algorithm that only contains conditional control statements.")
                    st.warning("In this way, a decision tree shows how a particular decision was made, using particular explanatory variables, in our case, how a particular brand was chosen based on gender, age, and city of consumer.")

                    brands = pd.Series(branddf['BRAND NAME'].drop_duplicates()).sort_values().tolist() # Faster than .unique()

                    selbrand = st.selectbox("Please select a brand to analyze", brands)

                    branddf2 = branddf[~branddf["BRAND"].duplicated(keep='first')]

                    # brandcode = branddf2.loc[branddf2['BRAND NAME'] == selbrand, 'BRAND'].item()
                    brandcode = branddf.loc[branddf['BRAND NAME'] == selbrand, 'BRAND'].iloc[0]
                    print(brandcode)

                    # Split data into features (X) and response (y)
                    X = df.loc[0:10000,["GENDER", "AGE", "CITY"]] # Subsampling for speed and efficiency, dynamic to certain relevant explanatory variables (could combine with dimensionality reduction or with conjoint analysis above)
                    y = df2.BRAND.apply(lambda x: brandcode if selbrand in x else 0).head(10001) # Dynamic to brand, subsampling for speed and efficiency

                    # Fit data to tree-based regression model
                    regressor = DecisionTreeRegressor(random_state=0)
                    regressor=regressor.fit(X,y)

                    # Visualising the decision tree regression results
                    plt.figure(figsize=(6,6), dpi=150)
                    plot_tree(regressor,max_depth=3,feature_names=X.columns, impurity=False)
                    plt.show()

                    buf = BytesIO()
                    plt.savefig(buf, format="png")
                    # st.image(buf)

                    # A scatter plot of latitude vs longitude
                    df.plot.scatter(x='AGE',y='GENDER',c='DarkBlue',s=1.5) # Dynamic to chosen variables

                    plt.figure(figsize=[6,4], dpi=120)
                    cutx, cuty = 50, 50
                    plt.ylim(0,cuty)       
                    plt.xlim(0,cutx)
                    plt.xlabel('AGE')
                    plt.ylabel('GENDER')
                    plt.scatter(x=X['AGE'],y=X['GENDER'],c=df['BRAND'].head(10001)) # Dynamic to chosen variables

                    buf2 = BytesIO()
                    plt.savefig(buf2, format="png")
                    st.image(buf2)

                    # splits = regressor.tree_.threshold[:2]
                    # print(splits, cutx, cuty)
                    # plt.plot([splits[1],splits[1]], [0,cuty]) 
                    # plt.plot([splits[1],cutx], [splits[0],splits[0]])
                    # plt.colorbar()
                    # plt.show()

                    # buf3 = BytesIO()
                    # plt.savefig(buf3, format="png")
                    # st.image(buf3)

                    X, y = make_regression(n_samples=1000, n_features=2,n_informative=2,
                                    random_state=0)
                    reg = DecisionTreeRegressor(max_depth=3).fit(X, y)

                    # Parameters
                    plot_colors = "ryb"
                    plot_step = 0.02

                    # Plot the decision boundary
                    f, axes =plt.subplots(ncols=1,nrows=2,figsize=(30, 30))

                    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                                        np.arange(y_min, y_max, plot_step))
                    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

                    Z = reg.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)
                    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Blues)

                    plt.xlabel('AGE')
                    plt.ylabel('GENDER')

                    axes[1].scatter(X[:, 0], X[:, 1], c=y,
                                cmap='Oranges', edgecolor='black', s=15)

                    plot_tree(reg, filled=True, feature_names=['AGE', 'GENDER'],
                            ax=axes[0], fontsize=10,
                            class_names='Target')

                    plt.show()

                    buf3 = BytesIO()
                    plt.savefig(buf3, format="png")
                    st.image(buf3)

                    st.write("")

                    st.success("We interpret the decision tree by analyzing how a consumer moves downwards and to the left, meaning that each condition referenced is true if we move to the bottom left, whereas as we move to the bottom right in any particular branch of the tree the relevant referenced condition is false, or opposite to that stated.")

                    # Spawn a new Quill editor
                    st.subheader("Notes on decision tree analysis")
                    dectreecontent = st_quill(placeholder="Write your notes here")

                    st.write("Conducting the decision tree analysis took ", time.time() - start_time, "seconds to run")

            if modelchoice == "Ensemble methods - random forest":

                with st.spinner('Please wait while we conduct the ensemble methods analysis using the random forest algorithm'):

                    my_bar = st.progress(0)

                    time.sleep(10)

                    for percent_complete in range(100):
                        time.sleep(0.01)
                        my_bar.progress(percent_complete + 1)

                    start_time = time.time()

                    st.info("This ensemble methods random forest model examines the effect of gender, age, and city of consumer on brand preference. It is a work in progress and will be expanded soon.")

                    # test classification dataset

                    st.write("Test classification dataset:")

                    # define dataset
                    X = df.loc[0:10000,["GENDER", "AGE", "CITY"]] # Subsampling for speed and efficiency, dynamic to certain relevant explanatory variables (could combine with dimensionality reduction or with conjoint analysis above)
                    y = df2.BRAND.head(10001) #.apply(lambda x: '1' if '*astle' in x else 0).head(10001) # Dynamic to brand, subsampling for speed and efficiency
                    # summarize the dataset
                    st.write("The shape of your subsampled data, in the form of explanatory variables (gender, age, city), and predicted variable (brand):")
                    st.write(X.shape, y.shape)

                    # print(X.head())

                    # X.drop(['BRANDNAME'], axis=1)

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
                    y = df['BRAND'].head(10001)
                    # Define predictor variables
                    # X = df.iloc[:, 1:-1].head(10000)
                    X = df.loc[0:10000,["GENDER", "AGE", "CITY"]]
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

                st.write("")

                # Spawn a new Quill editor
                st.subheader("Notes on random forest analysis")
                randomforestcontent = st_quill(placeholder="Write your notes here")

                st.write("Conducting the random forest model took ", time.time() - start_time, "seconds to run")

            if modelchoice == "This will come later":

                max_depth = st.slider("What should be the max_depth of the model?", min_value=10, max_value=100, value=20, step=10)
                n_estimators = st.selectbox("How many trees?", options=[100, 200, 300, 'No limit'], index = 0)
                st.text("Here is a list of features from the database:")
                input_feature = st.text_input("What is the input feature?", 'uuid')

    ### UNTOUCHED ORIGINAL CODE