import os
import shutil
# log_list = os.listdir("./logs")
# for log in log_list:
#     time_1 = [0]
#     with open("./logs/{}".format(log), "r") as f:
#         lines = f.read().splitlines()
#         for i in range(len(lines)-1):
#             if float(lines[i+1].split("In")[1].split("s")[0]) - float(lines[i].split("In")[1].split("s")[0]) > 0.04 or \
#                     lines[i+1].split(":")[1].split(",")[0] != lines[i].split(":")[1].split(",")[0]:
#                 time_1.append(i)
#         time_1.append(len(lines)-1)
#     for i in range(len(time_1)-1):
#         with open("./log.txt", 'a') as f1:
#             f1.write("人员ID:{},开始时间:{},结束时间:{},违规动作:{}\n".format(log.split(".")[0], lines[time_1[i]].split("In")[1],\
#                                 lines[time_1[i+1]].split("In")[1], lines[time_1[i]+1].split(":")[1].split(",")[0]))
if os.path.exists("./logs"):
    shutil.rmtree("./logs")
    os.mkdir("./logs")