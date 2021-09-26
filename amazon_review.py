import pandas as pd
import math
import scipy.stats as st
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


def score_up_down_diff(up, down):
    return up - down


def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)


#WILSON_LOWER_BOUND: bu yöntem için yorumların binary olması lazım. beğendi-beğenmedi- faydalı- değil gibi.
#burada p olasılığını istatistiksel olarak genelleyelim. yani yorum sonsuza dek orda olsa, bu yorumun alacağı
#helpful oyu %95 güvenirlik ile en az kaç alacağını sonuçlarda görebilirim.
def wilson_lower_bound(helpful_yes,helpful_no,confidence=0.95):
    n = helpful_yes + helpful_no
    if n == 0:
        return 0
    z = st.norm.ppf(1-(1-confidence)/2)
    phat = 1.0 * helpful_yes/ n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

####GÖREV 1: Average Rating'i Güncel Yorumlara Göre Hesaplayınız ve Var Olan Average Rating ile Kıyaslayınız.

###################################################
# 1. Veri Setini Okutunuz ve Ürünün Ortalama Puanını Hesaplayınız.
###################################################


df = pd.read_csv("amazon_review.csv")
df.head()
# tek bir ürün var zaten.
df.asin.value_counts()   #B007WTAJTO  4915
df["reviewerID"].nunique()  #4915


# ortalama rating
df["overall"].mean()   #4.5875

#time_based_weighted_average değeri
#Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır.
#Bu görevde amacımız verilen puanları tarihe göre ağırlıklandırarak değerlendirmek.
#İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.
df["day_diff"].describe().T


def time_based_weighted_average(dataframe, w1=28, w2=26, w3= 24, w4=22):
    return df.loc[df["day_diff"] <= 30, "overall"].mean() * w1/100 + \
    df.loc[(df["day_diff"] > 30) & (df["day_diff"] <= 90), "overall"].mean() * w2/100 + \
    df.loc[(df["day_diff"] > 90) & (df["day_diff"] <= 180), "overall"].mean() * w3/100 + \
    df.loc[df["day_diff"] > 180, "overall"].mean() * w4/100

time_based_weighted_average(df)  #4.6987
# tarihe bakmadan yaptığımız hesaplama 4.5875. Tarihe göre 4.6987.  görüyoruz ki yeniler yüksek puan vermiş. Güzel:)


#Görev 2: Ürün için Ürün Detay Sayfasında Görüntülenecek 20 Review'i Belirleyiniz.

#Not: total_vote bir yoruma verilen toplam up-down sayısıdır. up, helpful demektir.
#veri setinde helpful_no değişkeni olmadıdığı için helpful_no'yu biz buluyoruz.

df["helpful_no"]= df["total_vote"]- df["helpful_yes"]
df.sort_values("helpful_no", ascending=False).head(20)

df = df[["reviewerName", "overall", "helpful_yes", "helpful_no", "total_vote", "day_diff"]]
df.head()

# score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayıp tabloya ekleyelim

# score_pos_neg_diff
df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

# score_average_rating
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

# wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)
df.head()



# 20 Yorumu Belirleyiniz ve Sonuçları Yorumlayınız.

df[["reviewerName","overall","helpful_yes","helpful_no","total_vote","score_pos_neg_diff","score_average_rating","wilson_lower_bound"]].\
    sort_values(by="wilson_lower_bound",ascending=False).head(20)