import os
import csv
data_dir = './imagenet_adv'

with open("adv.csv", 'w+') as csvfile:
   writer = csv.writer(csvfile)
   for i in os.listdir(data_dir):
       if i.startswith("."):
           continue
       current_path = os.path.join(data_dir, i)
       for file_name in sorted(os.listdir(current_path)):
           if file_name.endswith("perturb.png"):
               continue
           if file_name.startswith("."):
               continue
           outputseg_path = file_name.split("orig")[0]+"perturb.png"
           writer.writerow([os.path.join(current_path, file_name), os.path.join(current_path, outputseg_path)])
