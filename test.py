import pandas as pd

# Dosyayı oku
df = pd.read_csv('data/shape_data.csv')

# İlk birkaç satırı görüntüle
print(df.head())

# Veri setinin boyutunu kontrol et
print(df.shape)

# Sütun isimlerini kontrol et
print(df.columns)