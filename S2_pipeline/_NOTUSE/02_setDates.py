import os, subprocess


output = subprocess.check_output("earthengine ls projects/servir-sea-landcover/assets/S2_Acolite_Hist",shell=True)

myList = output.decode('utf8').split("\n")

counter = 0
for items in myList:
        myName = items.split("/")
        fname = myName[4].split("_")
        year = fname[2]
        month = fname[3]
        day = fname[4]
        time = str(year) + "-" + str(month) + "-" + str(day) + "T00:00:00"
        print("earthengine asset set --time_start " + time + " " + items)
        os.system("earthengine asset set --time_start " + time + " " + items)
        print(fname,year,month,day)

