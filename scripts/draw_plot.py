import csv
import os
import numpy as np
import matplotlib.pyplot as plt

perf_data = []
with open('gemm_batch_test.csv') as csvfile:
    csv_reader = csv.reader(csvfile)  
    perf_header = next(csv_reader)  # read titles
    for row in csv_reader:  # save start second row after title row
        perf_data.append(row)

perf_dict = {}
M_dict = {}
N_dict = {}
K_dict = {}
for row in perf_data:
    perf_dict.setdefault(row[0],[]).append(float(row[4]))
    M_dict.setdefault(row[0],[]).append(int(row[1]))
    N_dict.setdefault(row[0],[]).append(int(row[2]))
    K_dict.setdefault(row[0],[]).append(int(row[3]))
    print(row)

plt.figure(figsize=(16,8)) 
for key in perf_dict:
    # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(M_dict[key], perf_dict[key],"x-",linewidth=1, label=key)

plt.xlabel("M") 
plt.ylabel("GFLOPS")  

plt.title("GEMM Performance (M==N==K)") 
plt.legend(loc='best')
# save pic
plt.savefig("gemm_batch_test_result.jpg") 
# show pic
plt.show()