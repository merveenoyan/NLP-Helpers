![enter image description here](https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/05df8cc2-4413-4a7c-93c7-dbf7991b18a7/de1bhi4-f4a2b7e2-cbd7-421c-bcbd-180a93a9772f.png/v1/fill/w_1280,h_498,q_80,strp/nlp_helpers_by_markdownimgmn_de1bhi4-fullview.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOiIsImlzcyI6InVybjphcHA6Iiwib2JqIjpbW3siaGVpZ2h0IjoiPD00OTgiLCJwYXRoIjoiXC9mXC8wNWRmOGNjMi00NDEzLTRhN2MtOTNjNy1kYmY3OTkxYjE4YTdcL2RlMWJoaTQtZjRhMmI3ZTItY2JkNy00MjFjLWJjYmQtMTgwYTkzYTk3NzJmLnBuZyIsIndpZHRoIjoiPD0xMjgwIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmltYWdlLm9wZXJhdGlvbnMiXX0.kade-APxd_CudJDoCw01YYzen_rjm9Gi5k5vk8XDUPE)
# NLP Helpers
Bu repo doğal dil işlemede kullandığım helper function'ları içerir. Pull request'le eklemeye açıktır.
Notlar: Bu repo'daki bazı fonksiyonlar NLTK'nın desteklediği dillerde geçerlidir. 

## Büyük Harfleri Küçük Harf Haline Getirme

    df["kucuk_harf"] = df["metin"].str.lower()
    df.head()

## Özel Karakterleri Temizleme

    import string
    
    noktalama_isaretleri = string.punctuation
    def noktalama_temizleme(metin):
        return metin.translate(str.maketrans('', '', noktalama_isaretleri))
    
    df["noktalamasiz_metin"] = df["metin"].apply(lambda text: noktalama_temizleme(metin))

str.maketrans(intab, outtab) bir metindeki intab'leri outtab'lerle değiştiriyor, bunu yaparken karakterlerin sıralamasını map ediyor.

## Stopword (Bağlaç) Temizleme (NLTK dilleri için geçerli)

    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
    def remove_stopwords(text):
    return " ".join([word for word in str(metin).split() if word not in STOPWORDS])
    df["baglacsiz_metin"] =
    df["metin"].apply(lambda metin: remove_stopwords(metin))

## Sık ve Nadir Görünen Kelimeleri Temizleme 
Kişisel not: Vektörize etme aşamasında max_df parametresiyle de temizleyebilirsiniz. Çok az görünenler için de min_df kullanılır.

    from collections import Counter
    cnt = Counter()
    for metin in df["metin"].values:
    for word in metin.split():
        cnt[word] += 1
    #en sık kullanılan on kelimeyi görmek için most_common metodunu kullanabiliriz 
    cnt.most_common(10)
Kelimeleri Temizleme

    freq = set([w for (w, wc) in cnt.most_common(10)])
    def remove_freqwords(metin):
    return " ".join([word for word in str(metin).split() if word not in freq])
    df["metin"] = df["metin"].apply(lambda metin: remove_freqwords(metin))
Aynı mantıkla .most_common metodunun döndürdüğü listenin son on elemanı bize en az kullanılan kelimeleri verir.

    n_rare_words = 10
    nadir_kelimeler = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])

## Kelimenin Kökünü Alma
Not: NLTK'nın desteklediği dillerde geçerlidir.
Snowball stemmer kullanmak isterseniz NLTK'nın stem kütüphanesinden SnowballStemmer'ı import edip aynı şekilde initialize etmeniz yeterli.

    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])
    df["kokler"] = df["metin"].apply(lambda metin: stem_words(metin))

## Kelimelerin Sözlükteki Köklerini Alma (Lemma)

    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    df["kokler"] = df["metin"].apply(lambda text: lemmatize_words(metin))
## Emojileri Silme

    # Cr: [https://gist.github.com/slowkow](https://gist.github.com/slowkow)

    def remove_emoji(string):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002500-\U00002BEF"  # chinese char
                                   u"\U00002702-\U000027B0"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001f926-\U0001f937"
                                   u"\U00010000-\U0010ffff"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u200d"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\ufe0f"  # dingbats
                                   u"\u3030"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', string)

## Regex'le URL Temizleme

    def remove_urls(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', metin)

## Vektörizasyon İşlemleri
Sklearn'ün CountVectorizer ve TFIDF-Vectorizer'ını kullanabilirsiniz. TFIDF Count Vectorizer'a göre daha hassastır. Bunu kullanmadan önce veriyi eğitim ve test olarak ayırıp ardından dönüşüm yapmalısınız.
**max_df, min_df** parametreleri metinde en çok gözüken ve en az gözüken (verilen sıklık yüzdelerine göre) kelimeleri atar.
**ngram_range** kelime öbeklerini vektörize eder, 1'den 2'ye olan ngram range size tek kelime ve çift kelimeden oluşacak şekilde kelimelerinizi alıp öbekler oluşturur.

        from sklearn.feature_extraction.text import TfidfVectorizer
        tvec= TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_df=0.9, min_df=0.05)
        t_train=tvec.fit_transform(x_train)
    t_test=tvec.fit_transform(x_test)

Count Vectorizer kullanımı:

    from sklearn.feature_extraction.text import CountVectorizer
    cvec = CountVectorizer(stop_words="english",ngram_range=(1,2), max_df=0.9, min_df=0.05)
    c_train=cvec.fit_transform(x_train)
    c_test=cvec.fit_transform(x_test)
## Keras/tf.keras Tokenizer'ıyla cümleleri küçük birimlere ayırma

    from keras.preprocessing.text import Tokenizer
    #ilk kaç kelimeyi alacak
    tokenizer=Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(df.metin)
    sekans_verisi=tokenizer.texts_to_sequences(df.metin)
    tokenized=tokenizer.texts_to_matrix(df.metin)
    #kelime indeksi oluşturma
    word_index=tokenizer.word_index
## Padding
Cümlelere (sekans verilerine) baştan ya da sondan padding uygulamalısınız ki sinir ağına girerken aynı boyutta olsunlar ve sorun çıkmasın. Bunu eğitim ve test verisi için ayrı ayrı yapın.

    X_train=tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=300)
    X_test=tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=300)

## En Çok Kullanılan Kelimeleri Plotly ile Görselleştirme
İtiraf: Aşağıdaki snippet'ı daha önce şiir veri setiyle oynarken kullanmak adına başka bir notebook'tan almıştım. Plotly kullanmıyorum. (Öğrenilsin)
Bu kod size en çok kullanılan 20 kelimeyi bar chart halinde verir.
  

      import plotly.graph_objects as go
        from plotly.offline import iplot
        words = df['metin'].str.split(expand=True).unstack().value_counts()
        data = [go.Bar(
                    x = words.index.values[2:20],
                    y = words.values[2:20],
                    marker= dict(colorscale='RdBu',
                                 color = words.values[2:40]
                                ),
                    text='Word counts'
            )]
        
        layout = go.Layout(
            title='Most used words excluding stopwords'
        )
        
        fig = go.Figure(data=data, layout=layout)
        
        iplot(fig, filename='basic-bar')

## Word Cloud Oluşturma
Word Cloud'un kendi kütüphanesi var, her dil için ayrı bağlaçlar içinde mevcut. Bağlaçları kendisi atıyor.

        from wordcloud import WordCloud, STOPWORDS
        import matplotlib.pyplot as plt
    def word_cloud(content, title):
        wc = WordCloud(background_color='white', max_words=200,
                      stopwords=STOPWORDS, max_font_size=50)
        wc.generate(" ".join(content))


> Written with [StackEdit](https://stackedit.io/).



