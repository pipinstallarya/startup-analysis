import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(layout='wide', page_title='Startup Analysis')

add_selectbox = st.sidebar.radio("Go to", ("Indian Startup Exploration","Startup Profit Prediction"))

if add_selectbox=="Indian Startup Exploration":
    df = pd.read_csv('startup_cleaned.csv')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    def load_investor_details(investor):
        st.title(investor)
        # load recent 5 investments
        st.subheader('Most Recent Investments')
        st.dataframe(df[df['investors'].str.contains(investor)].head()[['date', 'startup', 'vertical', 'city', 'round', 'amount']])

        #biggest investments
        col1, col2 = st.columns(2)
        with col1:
            big_investments = df[df['investors'].str.contains(investor)].groupby('startup')['amount'].sum().sort_values(ascending=False).head()
            st.subheader('Biggest Investments')
            fig, ax = plt.subplots()
            ax.bar(big_investments.index, big_investments.values)
            st.pyplot(fig)

        # investment by sectors
        with col2:
            sectors_invested = df[df['investors'].str.contains(investor)].groupby('vertical')['amount'].sum()
            st.subheader('Sectors Invested In')
            fig1, ax1 = plt.subplots()
            ax1.pie(sectors_invested, labels=sectors_invested.index, autopct='%0.01f%%')
            st.pyplot(fig1)

        col3, col4 = st.columns(2)
        # investment by rounds
        with col3:
            rounds = df[df['investors'].str.contains(investor)].groupby('round')['amount'].sum()
            st.subheader('Investment Rounds')
            fig2, ax2 = plt.subplots()
            ax2.pie(rounds, labels=rounds.index, autopct='%0.01f%%')
            st.pyplot(fig2)

        # investment by cities
        with col4:
            cities = df[df['investors'].str.contains(investor)].groupby('city')['amount'].sum()
            st.subheader('Investment By Cities')
            fig3, ax3 = plt.subplots()
            ax3.pie(cities, labels=cities.index, autopct='%0.01f%%')
            st.pyplot(fig3)

        col5, col6 = st.columns(2)
        # YoY investment
        with col5:
            YoY = df[df['investors'].str.contains(investor)].groupby('year')['amount'].sum()
            st.subheader('YoY Investment')
            fig4, ax4 = plt.subplots()
            ax4.plot(YoY.index, YoY.values)
            st.pyplot(fig4)


    def load_overall_analysis():
        st.title('Overall Analysis')
        col1, col2, col3, col4 = st.columns(4)
        # total investment
        total_investment = round(df['amount'].sum())
        # max funding
        max_funding = df.groupby('startup')['amount'].max().sort_values(ascending=False).head(1).values[0]
        # average funding
        avg_funding = df.groupby('startup')['amount'].sum().mean()
        # total funded startups
        total_startups = df['startup'].nunique()
        with col1:
            st.metric('Total', '{} Cr'.format(total_investment))
        with col2:
            st.metric('Max Funding', '{} Cr'.format(max_funding))
        with col3:
            st.metric('Average Funding', '{} Cr'.format(round(avg_funding)))
        with col4:
            st.metric('Total Funded Startups', '{}'.format(total_startups))

        # top sectors
        col5, col6 = st.columns(2)
        with col5:
            st.subheader('Top 5 Sectors')
            top5_sectors = df['vertical'].value_counts().head()
            fig1, ax1 = plt.subplots()
            ax1.bar(top5_sectors.index, top5_sectors.values)
            st.pyplot(fig1)
        with col6:
            st.subheader('Amount Invested')
            amount_invested = df.groupby('vertical')['amount'].sum().sort_values(ascending=False).head(5)
            fig2, ax2 = plt.subplots()
            ax2.bar(amount_invested.index, amount_invested.values)
            st.pyplot(fig2)

        # MoM investment
        st.header('Month on Month Investment')
        selected_option = st.selectbox('Select type', ['Amount', 'Startups'])
        if selected_option == 'Amount':
            temp_df = df.groupby(['year', 'month'])['amount'].sum().reset_index()
        else:
            temp_df = df.groupby(['year', 'month'])['amount'].count().reset_index()
        temp_df['x_axis'] = temp_df['month'].astype('str') + '-' + temp_df['year'].astype('str')
        fig3, ax3 = plt.subplots()
        ax3.plot(temp_df['x_axis'], temp_df['amount'])
        st.pyplot(fig3)


    def load_startup_details(startup):
        st.title(startup)
        col1, col2 = st.columns(2)
        with col1:
            # investment details
            industry_series = df[df['startup'].str.contains(startup)][['year', 'vertical', 'city', 'round']]
            st.subheader('About The Startup')
            st.dataframe(industry_series)

        with col2:
            inv_series = df[df['startup'].str.contains(startup)].groupby('investors').sum()
            st.subheader('Investors')
            st.dataframe(inv_series)

        # Subindustry
        col1, col2 = st.columns(2)
        with col1:
            ver_series = df[df['startup'].str.contains(startup)].groupby('vertical')['year'].sum()
            st.subheader('Industry')
            fig9, ax9 = plt.subplots()
            ax9.pie(ver_series, labels=ver_series.index, autopct="%0.01f%%")
            st.pyplot(fig9)

        with col2:
        
            sub_series = df[df['startup'].str.contains(startup)].groupby('subvertical')['year'].sum()
            st.subheader('Sub-Industry')
            fig10, ax10 = plt.subplots()
            ax10.pie(sub_series, labels=sub_series.index, autopct="%0.01f%%")
            st.pyplot(fig10)


    st.sidebar.title('Startup Funding Analysis')

    option = st.sidebar.selectbox('Select One', ['Overall Analysis', 'Startup', 'Investor'])

    if option == 'Overall Analysis':
            load_overall_analysis()


    elif option == 'Startup':
        select_startup = st.sidebar.selectbox('Select Startup', sorted(df['startup'].unique().tolist()))
        btn1 = st.sidebar.button('Find Startup Details')
        if btn1:
            load_startup_details(select_startup)

    else:
        selected_investor = st.sidebar.selectbox('Select StartUp', sorted(set(df['investors'].str.split(',').sum())))
        btn2 = st.sidebar.button('Find Investor Details')
        if btn2:
            load_investor_details(selected_investor)

if add_selectbox=="Startup Profit Prediction":
    st.title("Startup's Profit Prediction")

    dataset = pd.read_csv("startup_profit.csv")

# spliting Dataset in Dependent & Independent Variables
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 4].values

    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    X[:, 3] = labelencoder.fit_transform(X[:, 3])


    from sklearn.model_selection import train_test_split

    x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=0)

    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    testing_data_model_score = model.score(x_test, y_test)
    st.markdown("Model Score/Performance on Testing data",testing_data_model_score)

    training_data_model_score = model.score(x_train, y_train)
    st.markdown("Model Score/Performance on Training data",training_data_model_score)

    rnd_cost = st.sidebar.number_input('Insert R&D Spend')
    st.write('The R&D Spend is ', rnd_cost)

    Administration_cost = st.sidebar.number_input('Insert Administration cost Spend')
    st.write('The Administration cost Spend is ', Administration_cost)

    Marketing_cost_Spend = st.sidebar.number_input('Insert Marketing cost Spend')
    st.write('The Marketing cost Spend is ', Marketing_cost_Spend)

# no_fundingrounds = st.sidebar.slider("No. of Funding Rounds", 1, 12, step=1)
# st.write('No of Funding Rounds Selected ', no_fundingrounds)

# no_seedrounds = st.sidebar.slider("No. of Initial Seed Funding(in crores)", 5, 15, step=1)
# st.write('Initial Seed Funding(in crores) ', no_seedrounds)

# sisfs = st.sidebar.selectbox("Startup India Seed Fund Scheme",("1", "0")) 
# st.write('SISFS', sisfs)

# international_sales = st.sidebar.selectbox("International Sales",("1","0"))
# st.write('International Sales) ', international_sales)

# female = st.sidebar.selectbox("Primary Owner Female",("1","0"))
# st.write('Primary Owner Female) ', female)

    option = st.sidebar.selectbox(
        'Select the region',
        ('Delhi-NCR', 'Karnataka', 'Maharashtra'))

    st.write('You selected:', option)

    if option == "Delhi-NCR":
        optn = 0
    if option == "Karnataka":
        optn = 1
    if option == "Maharashtra":
        optn = 2   

    y_pred = model.predict([[Marketing_cost_Spend,Administration_cost,rnd_cost,optn]])

    if st.button('Predict'):
        st.success('The Profit must be {} '.format(y_pred))
    else:
        st.write('Please fill all the important details')


    fig = plt.figure()

    X = ['Total Cost Spent']
    x_value = [rnd_cost+Administration_cost+Marketing_cost_Spend]
  
    X_axis = np.arange(len(X))
  
    plt.bar(X_axis - 0.2, x_value, 0.4, label = 'cost')
    plt.bar(X_axis + 0.2, y_pred, 0.4, label = 'profit')
  
    plt.xticks(X_axis, X)
    plt.xlabel("RS")
    plt.title("Profit vs Total Cost Spent")
    plt.legend()
    plt.show()

    st.pyplot(fig)
