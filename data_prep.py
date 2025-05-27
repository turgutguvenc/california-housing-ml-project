import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from scipy import stats


def check_normal_distribution(dataframe, num_col, std_distance=1, z_score_distance=3):
    """
    checks the emprical rule for a normal distribution and returns the outliers indexes using Z_score

    empirical rule says that for a normal distribution:
    68% of the values fall within +/- 1 SD from the mean
    95% of the values fall within +/- 2 SD from the mean
    99.7% of the values fall within +/- 3 SD from the mean

    -----------------------------

    Knowing that your data is normally distributed is useful for analysis because many statistical tests and machine learning models assume a normal distribution.
    Plus, when your data follows a normal distribution, you can use z-scores to measure the relative position of your values and find outliers in your data.

    -----------------------------
    Compute z-scores to find outliers
    A z-score is a measure of how many standard deviations below or above the population mean a data point is.
    A z-score is useful because it tells you where a value lies in a distribution.

    Data professionals often use z-scores for outlier detection.
    Typically, they consider observations with a z-score smaller than -3 or larger than +3 as outliers.
    In other words, these are values that lie more than +/- 3 SDs from the mean.
    """
    mean_value = dataframe[num_col].mean()
    std_value = dataframe[num_col].std()

    #### Normal Distribution test
    lower_limit = mean_value - 1 * std_value
    upper_limit = mean_value + 1 * std_value
    ### Chance

    percentage_within_limits = ((dataframe[num_col] >= lower_limit) & (dataframe[num_col] <= upper_limit)).mean() * 100
    assert 63 <= percentage_within_limits <= 73, "Percentage out of expected range (65% to 72%)"

    lower_limit = mean_value - 2 * std_value
    upper_limit = mean_value + 2 * std_value
    percentage_within_limits = ((dataframe[num_col] >= lower_limit) & (dataframe[num_col] <= upper_limit)).mean() * 100
    assert 92 <= percentage_within_limits <= 98, "Percentage out of expected range (92% to 98%)"

    lower_limit = mean_value - 3 * std_value
    upper_limit = mean_value + 3 * std_value
    percentage_within_limits = ((dataframe[num_col] >= lower_limit) & (dataframe[num_col] <= upper_limit)).mean() * 100
    assert 97 <= percentage_within_limits <= 101, "Percentage out of expected range (97% to 101%)"

    print("Data is normally distributed.....")

    print("Outliers")
    data["z_score"] = stats.zscore(data[num_col], ddof=1)  # ddof=degrees of freedom correction (sample vs. population)
    print(data[(data["z_score"] > z_score_distance) | (data["z_score"] < - z_score_distance)])
    return data[(data["z_score"] > z_score_distance) | (data["z_score"] < - z_score_distance)].index


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    """-***Note if data is skewed right or left use median value.***"""
    # üst v alt limitleri hesaplayıp döndürür.
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable, q1=0.25, q3=0.75):
    """
    outliarları alt ve üst limite dönüştürüp baskılar.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=q1, q3=q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit



def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    """
    aykırı değer var mı yok sonucunu döner  
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1=q1, q3=q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False



def grab_outliers(dataframe, col_name, index=False, head=5):
    """aykırı değerleri print eder istersek aykırı değerlerin indexini döndürür."""

    low, up = outlier_thresholds(dataframe, col_name) #alt üst limtleri getien foksiyon
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10: # shape satır sayısı 10 dan büyük head ini alır
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head(head))
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index



def remove_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    """ Aykırıları siler. aykırıların silinmiş halini döndürür."""
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1=q1, q3=q3)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers



def missing_values_table(dataframe, na_name=False):
    """eksik değerlerin sayını ve oranını na_name= True olursa kolon isimleirinide verir.
    """
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False) #dataframe bu
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False) #dataframe bu ****
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


def missing_vs_target(dataframe, target, na_columns):
    """eksik değerleri 1 temsil ediyor.
    eksiklik barındıran değerlerin target değişkeni ortalamasını ve sayılarını print eder. 
    """
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


def label_encoder(dataframe, binary_col):
    """
    tek sütunda yani ilgili kolonda çevirme işlemini yapıyor.
    lable encoder ile binary(ikili) kolanları encode ediyoruz. ordinal değerlerde ise binary kolon olmasada kullanabiliriz.
    yani sadece iki sınıflıda veya ordinal verilerde kullanılır(ordinalleri 0,1,2,3,4 gibi kodlar.)
    """
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe



def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


def rare_encoder(dataframe, rare_perc, cat_cols):
    # sadece kategorik değil bütün cat_cols listesinin rare durumunu inceleyecek şekilde güncellendi.

    # burda 1 güncelleme daha var.
    # 1' den fazla rare varsa düzeltme yap. durumu göz önünde bulunduruldu.
    # rare sınıf sorgusu 0.01'e göre yapıldıktan sonra gelen True' ların sum'ı alınıyor.
    # eğer 1'den büyük ise rare cols listesine alınıyor.


    # oran rare_perc(0.01) dan küçük ve değişken sayısının toplamı birden fazla ise rare columns a ekle
    rare_columns = [col for col in cat_cols if (dataframe[col].value_counts() / len(dataframe) < rare_perc).sum() > 1] 

    for var in rare_columns:
        tmp = dataframe[var].value_counts() / len(dataframe)
        rare_labels = tmp[tmp < rare_perc].index
        dataframe[var] = np.where(dataframe[var].isin(rare_labels), 'Rare', dataframe[var])

    return dataframe

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()    