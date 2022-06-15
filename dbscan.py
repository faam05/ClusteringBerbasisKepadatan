import pandas as pd
import numpy as np
import matplotlib.pyplot as mp

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

# membaca file csv
data = pd.read_csv("gempa.csv")

# memilih atribut yang diperlukan
data_x = data.drop(["Cluster", "Nomor_KK", "Nama_KK", "Alamat_Asli"], axis = 1)
# print(data_x)

# merubah data di file menjadi array
x_array = np.array(data_x)
# print(x_array)

# proses normalisasi
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x_array)
# print(x_scaled)

# memasukkan array ke algoritma DBSCAN
db = DBSCAN(eps=0.1, min_samples=3)
db.fit(x_scaled)

# menampilkan jumlah cluster
labels=db.labels_
n_raw=len(labels)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print("Terdapat "+str(n_clusters)+" cluster yang terbentuk")

# menampilkan data cluster di console
for i in range(0,20):
    print("Nomor KK:"+str(data.values[i,0])+", Kepala KK:"+str(data.values[i,1])+", Alamat:"+str(data.values[i,4])+", Cluster:"+str(db.labels_[i]))
    print("-----------------------------------------------------------")

# menampilkan visualisasi datanya
data["kluster"] = db.labels_
output = mp.scatter(x_scaled[:,0], x_scaled[:,1], s = 100, c = data.kluster, marker = "o", alpha = 1, )
mp.title("Hasil Clustering DBSCAN")
mp.colorbar(output)
mp.show()
