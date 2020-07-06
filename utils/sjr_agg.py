import pandas as pd

df = pd.read_csv("../dataset/SJR/scimagojr 2018.csv", sep=";",
    header=0, usecols=["Sourceid", "Title", "Type", "Country",
    "Categories"])
df = df.set_index("Sourceid")

for ano in range(2017, 1998, -1):
    new_df = pd.read_csv("../dataset/SJR/scimagojr " + str(ano) + ".csv",
        sep=";", header=0,
        usecols=["Sourceid", "Title", "Type", "Country", "Categories"])
    new_df = new_df.set_index("Sourceid")
    df = df.append(new_df.loc[~new_df.index.isin(df.index)])

df.to_csv("../dataset/SJR/scimagojr completo.csv", sep=";")
