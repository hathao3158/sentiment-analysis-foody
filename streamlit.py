#==============================================================================================================================
#IMPORT LIBRARY
#=============================================================================================================================
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve

from sklearn.naive_bayes import BernoulliNB

import streamlit as st

import pickle

from bs4 import BeautifulSoup
import re
from urllib.request import Request, urlopen

#======================================================================================================================
# 1. CODE
#======================================================================================================================
# 1.1. Read data
data = pd.read_csv("df_pre.csv", encoding='utf-8')
data = data.dropna()

def data_understand(df):
    uploaded_file = st.file_uploader("Choose a file", type = ['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
        df.to_csv("spam_new.csv", index=False)
    
    st.write("#### 1. Some data")
    st.image("image/foody_review.png")
    st.dataframe(df[['restaurant','review_text','review_score']].head())
    #st.dataframe(data[['restaurant','review_text','review_score']].tail())
    st.write('''###### Dữ liệu review_text chứa bình luận của khách hàng sẽ là input.
    * review_text có icon emoji, có các từ viết tắt, teencode, viết sai chính tả
    * Các từ lặp đi lặp lại nhiều như: món ăn, bánh, đồ uống,...
    * Từ mang nghĩa phủ định như "không ngon", "không nên đến"... 
    * => Cần xử lý các data review_text cho ngắn gọn để đưa vào model.\n\n''')
    st.markdown("###### review_text ban đầu:")
    view4 = df.loc[4,'review_text']
    st.code(view4)
    st.markdown("###### review_text sau NLP:")
    view42 = df.loc[4,'review_text_clean']
    st.code(view42)

    st.write("#### 2. Visualize")
    fig1 =  plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    sb.distplot(df.review_score)
    plt.subplot(1,2,2)
    plt.boxplot(df.review_score)
    plt.suptitle('Distplot & Boxplot of review_score')
    st.pyplot(fig1)
    
    st.write("###### Dữ liệu tập trung ở điểm số từ 6 đến 10\n\n")

    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    df_view = df
    df_view['review_score_bins'] = pd.cut(df_view['review_score'], bins=bins)
    scr_bin = df_view.groupby('review_score_bins').size().sort_values(ascending=False)
    fig2 =  plt.figure(figsize=(12,6))
    scr_bin.plot.bar(color='c')
    st.pyplot(fig2)

    st.write("###### Đánh giá thang 7-10 chiếm số lượng nhiều, tổng số tần suất của thang điểm <7 cộng lại vẫn ít hơn thang >7. Tuy nhiên nếu lấy mốc 7 để chia dữ liệu thì các bình luận chê sẽ không được thể hiện rõ và dữ liệu vẫn mất cân bằng => sẽ dùng cách khác để cân bằng dữ liệu và lấy mốc 6 để phân class\n")
    st.write('''* Chia dữ liệu thành 2 nhóm:
    * Nhóm 0: Chê (Notlike): Thang điểm <=6
    * Nhóm 1: Khen (Like): Thang điểm > 6
    \n\n''')

    count = df['review_class'].value_counts()
    fig3 =  plt.figure(figsize=(6,6))
    count.plot.bar()
    plt.xticks(rotation=0)
    st.pyplot(fig3)

    st.write('###### Dữ liệu không cân bằng')

    for label, cmap in zip(['Like','Notlike'],['Reds','Greens']):
        text = str(df['review_text_clean'][df['review_class']== label].values)
        fig4 = plt.figure(figsize=(10, 6))
        wc = WordCloud(width=1000, height=600, background_color="#919191", colormap=cmap)
        wc.generate_from_text(text)
        plt.imshow(wc)
        plt.axis("off")
        plt.title(f"Words Commonly Used in ${label}$ Reviews", size=20)
        st.pyplot(fig4)
        

    #------Import html file but it is too big to show, page keep loading---------
    #HtmlFile = open("Foody_Sentiment_raw.html", 'r', encoding='utf-8')
    #source_code = HtmlFile.read()
    #components.html(source_code)

    #HtmlFile = open("Foody_Sentiment_clean.html", 'r', encoding='utf-8')
    #source_code = HtmlFile.read()
    #components.html(source_code)
    #-----------show screenshot instead--------------

    st.image("image/scatter_raw.png")
    st.write("###### Với text chưa qua xử lý, sử dụng công cụ scattertext ta thấy Like (màu đỏ) có nhiều icon hơn, Notlike có icon giận dữ, các từ xuất hiện nhiều ở nhóm Notlike là: không bao giờ, tệ, thái độ, thất vọng...\n\n")
    st.image("image/scatter_clean.png")
    st.write('''
    * Với clean text, Notlike có các từ: không thèm, chửi, dở, bực mình, phí tiền, phục vụ kém, giận, thất vọng, không bao giờ,  ...
    * Like có các từ: ủng hộ dài dài, thân thiện nhiệt tình, trang trí đẹp mắt, không gian thoáng mát, giá cả bình dân, tuyệt vời, vừa miệng, hợp lý''')


text_data = np.array(data['review_text_clean'])
# 1.2. CountVectorizer
count = CountVectorizer(max_features=1000)
count.fit(text_data)
bag_of_words = count.transform(text_data)
# 1.3. Data pre-processing
X = bag_of_words.toarray()
y = np.array(data['label'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# 1.4. Build model
clf = BernoulliNB()
model = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# 1.5. Evaluate model
score_train = model.score(X_train,y_train)
score_test = model.score(X_test,y_test)
acc = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
cr = classification_report(y_test, y_pred)
y_prob = model.predict_proba(X_test)
roc = roc_auc_score(y_test, y_prob[:, 1])
# 1.6. Save models
# luu model classication Like/Notlike
pkl_filename = "sentiment_analyse.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)

# luu model CountVectorizer (count)
pkl_count = "count_model.pkl"  
with open(pkl_count, 'wb') as file:
    pickle.dump(count, file)

# 1.7. Load models 
# Đọc model
# import pickle
with open(pkl_filename, 'rb') as file:  
    sentiment_model = pickle.load(file)
# doc model count len
with open(pkl_count, 'rb') as file:  
    count_model = pickle.load(file)
    
def build_model():
    st.write("#### 1. Build model: Using BernouliNB")
    
    st.write("#### 2. Evaluation")
    st.code("Score train:"+ str(round(score_train,2)) + " vs Score test:" + str(round(score_test,2)))
    st.code("Accuracy:"+str(round(acc,2)))
    st.write("###### Confusion matrix:")
    st.code(cm)
    # visualize confusion matrix with seaborn heatmap
    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                    index=['Predict Positive:1', 'Predict Negative:0'])
    fig1 = plt.figure(figsize=(5,5))
    sb.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    st.pyplot(fig1)

    st.write("###### Classification report:")
    st.code(cr)
    st.code("Roc AUC score:" + str(round(roc,2)))
    # calculate roc curve
    st.write("###### ROC curve")
    fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
    fig2, ax = plt.subplots()       
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.plot(fpr, tpr, marker='.')
    st.pyplot(fig2)

    st.write("##### 3. Kết luận")
    st.write('''
    * BernoulliNB cho kết quả cao ở cả Precision, Recall của class 0, Accuracy 86%, FP ít, f1-score 0.78, thời gian thực hiện: 2 giây.
    * Ngoài ra BernoulliNB cho kết quả tốt trên data gốc (không cần oversampling hay undersampling, undersampling sẽ làm giảm lượng data => lãng phí dữ liệu thu thập được, còn oversampling thì tạo ra data giả cũng không phải data thực tế thu thập được)
    ''')


def crawl_data(link):
    req = Request(link)
    webpage = urlopen(req).read()
    soup = BeautifulSoup(webpage, "html.parser")
    name = soup.find("h1").get_text().strip()
    avg_rate = re.findall(r'\D*(\d+\.?\d*)\D*',soup.find('div', {"class":"microsite-top-points-block"}).text)[0]
    comment = soup.find_all('div', attrs={"class":"rd-des toggle-height"})
    comment_lst = [x.get_text().replace("\n...Xem thêm","") for x in comment]
    return name, avg_rate, comment_lst

def classify_review(lines):
    if len(lines)>0:
        st.code(lines)
        x_new = count_model.transform(lines)        
        y_pred_new = sentiment_model.predict(x_new)
        if y_pred_new == 1:
            st.write(str(y_pred_new)+" - Khen 😍")
        elif y_pred_new == 0:
            st.write(str(y_pred_new)+" - Chê 😡")


#======================================================================================================================
# 2. GUI
#======================================================================================================================
st.title("PROJECT 2: SENTIMENT ANALYSIS - FOODY.VN")
st.sidebar.markdown("## Choose content")


menu = ['Business Understanding','Data Understanding','Build Model']
button1 = st.sidebar.button(menu[0])
button2 = st.sidebar.button(menu[1])
button3 = st.sidebar.button(menu[2])

choice = st.sidebar.selectbox('SENTIMENT ANALYSIS',['--','From Data Input','From Website'])
    
if button1:
    st.subheader(menu[0].upper())
    st.image("image/app_foody.jpg")
    st.write("""* Foody.vn là một kênh phối hợp với các nhà hàng/quán ăn bán thực phẩm online.
    * Để lựa chọn một nhà hàng/quán ăn mới khách hàng có xu hướng xem xét những bình luận từ những người đã thưởng thức để đưa ra quyết định có nên thử hay không?
    * Các nhà hàng/quán ăn cần nỗ lực để cải thiện chất lượng của món ăn cũng như thái độ phục vụ nhằm duy trì uy tín của nhà hàng cũng như tìm kiếm thêm khách hàng mới dựa trên bình luận từ khách hàng
    * Chúng ta có thể lên đây để xem các đánh giá, nhận xét cũng như đặt mua thực phẩm.
    * Từ những đánh giá của khách hàng, vấn đề được đưa ra là làm sao để các nhà hàng/ quán ăn hiểu được khách hàng rõ hơn, biết họ đánh giá về mình như thế nào để cải thiện hơn trong dịch vụ/sản phẩm.
    \n\n""")
    st.image("image/foody_show_review.png")
    st.write(""" Ví dụ trong hình là nhà hàng có điểm đánh giá 7.4. Đây chưa phải là điểm cao nhất nên nhà hàng vẫn cần xem đánh giá từ khách hàng để từ đó cải thiện chất lượng dịch vụ. Nhà hàng có 3 bình luận kém, 3 bình luận trung bình. """)
    st.write("##### Mục tiêu: Đánh giá nội dung các bình luận để tìm ra đâu là những bình luận chê, kém, đánh giá thấp để từ đó nhà hàng biết những chỗ cần cải thiện.")

elif button2:
    st.subheader(menu[1].upper())
    data_understand(data)
elif button3:
    st.subheader(menu[2].upper())
    build_model()

else:
    if choice == '--':
        st.image("image/emoji.png")
    if choice == 'From Data Input':
       st.subheader('Select Data')
       flag = False
       new_dt = None
       type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
       if type=="Upload":
           # Upload file
           uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
           if uploaded_file_1 is not None:
                new_dt = pd.read_csv(uploaded_file_1)
                new_dt = new_dt.dropna()
                st.dataframe(new_dt)
                if 'review_text_clean' in new_dt.columns:
                    new_dt = new_dt['review_text_clean'].head(1)
                else:
                    new_dt = new_dt.select_dtypes(include=object)
                    new_dt = new_dt.iloc[0,:]
                flag = True
       if type=="Input":        
           review = st.text_area(label="Input your review:")
           submit2 = st.button("Submit")
           if submit2 and review!="":
               new_dt = np.array([review])
               flag = True
    
       if flag:
           classify_review(new_dt)
    
    elif choice == 'From Website':
       resto = st.text_input(label="Nhập link nhà hàng:")
       if resto!= "":
            if resto.startswith("https://www.foody.vn/"):
                if resto.endswith("/binh-luan"):
                    pass
                else:
                    sufix = "/binh-luan"
                    resto =  resto+sufix
                name, rate, comments =  crawl_data(resto)
                st.write("#### Nhà hàng: "+name)
                st.write("#### Rating: "+rate)
                if len(comments)>0:
                   for comm in comments[:3]:
                       new_dt = np.array([comm])
                       classify_review(new_dt)
                else:
                   st.write("Not found data")
            else:
                st.write("Only get link restaurants on Foody.vn")
    
st.sidebar.markdown("#### by Nguyen Ha Phuong Thao")