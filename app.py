import streamlit as st
import preprocessor,helper,senti
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import matplotlib.font_manager as fm


def set_emoji_font():
    font_path = "D:/WCA Project/SEGUIEMJ.ttf"
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = prop.get_name()

st.sidebar.title("Whatsapp Chat Analysis")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    set_emoji_font()
    bytes_data = uploaded_file.getvalue()
    data=bytes_data.decode("utf-8")
    df=preprocessor.preprocess(data)
    s_df=senti.prepro(data)

    #fetch unique users


    user_list=df['users'].unique().tolist()
    user_list.remove('grp_notification')
    user_list.sort()
    user_list.insert(0,'Overall')

    selected_user=st.sidebar.selectbox("Show analysis wrt",user_list)


    if st.sidebar.button("Show Analysis"):
        st.dataframe(df)
        #stats area

        num_messages,words,Media,Links=helper.fetch_stats(selected_user,df)
        st.title("Top Statistics")
        col1,col2,col3,col4=st.columns(4)

        with col1:
            st.header('Total Messages')
            st.title(num_messages)
        with col2:
            st.header('Total Words')
            st.title(words)
        with col3:
            st.header('Media Shared')
            st.title(Media)
        with col4:
            st.header('Links Shared')
            st.title(Links)

        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['messages'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['messages'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # activity map
        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        #finding the busiest user

        if selected_user=='Overall':
            st.title('Most Busy Users(Number and Percentage)')
            x,new_df=helper.most_busy_users(df)
            fig,ax=plt.subplots()
            col1,col2=st.columns(2)

            with col1:
                ax.bar(x.index,x.values,color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)
        # word cloud
        st.title('Word Cloud')
        df_wc=helper.create_wordcloud(selected_user,df)
        fig,ax=plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        #most common words
        most_common_df=helper.most_common_words(selected_user,df)
        fig,ax=plt.subplots()
        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')
        st.title('Most Common Words')
        st.pyplot(fig)

        # emoji analysis
        emoji_df=helper.emoji_helper(selected_user,df)
        if emoji_df.shape[0]!=0:
            col1, col2 = st.columns(2)

            with col1:
                st.title('Emoji Dataframe')
                st.dataframe(emoji_df)
            with col2:
                st.title('PieChart')
                fig, ax = plt.subplots()
                ax.pie(emoji_df[1].head(),labels=emoji_df[0].head(),autopct="%0.2f")
                st.pyplot(fig)
        else:
            st.title('This User has not shared any emojis in this chat')

    option = st.sidebar.selectbox("Show Sentiment Analysis", options=["Overall"])

    st.write("You selected:", option)

    if st.sidebar.button("Show Sentiment Analysis"):
        st.dataframe(s_df)

        st.title('Sentiment Score')
        dfI=senti.sentimentA(s_df)
        df_ss = dfI.head(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=df_ss, x='user', y='average_compound')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        st.title('Most Optimistic Users')

        df_graph_sorted_all = dfI.sort_values(by='average_compound', ascending=False)

        df_graph_top10 = df_graph_sorted_all.head(10)

        # Create the barplot for the top 10 users
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=df_graph_top10, x='user', y='average_compound')

        # Set the title and display the plot
        plt.xticks(rotation=90)  # Rotate the user names if necessary
        st.pyplot(fig)

        st.title('Most Pessimistic Users')

        df_graph_bottom10 = df_graph_sorted_all.tail(10)

        # Create the barplot for the bottom 10 users
        fig,ax=plt.subplots(figsize=(6, 4))
        sns.barplot(data=df_graph_bottom10, x='user', y='average_compound')

        # Set the title and display the plot
        plt.xticks(rotation=90)  # Rotate the user names if necessary

        st.pyplot(fig)