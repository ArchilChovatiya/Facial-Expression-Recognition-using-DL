import cv2
import pandas as pd
import glob

disc={0:"angry",1:"disgust",2:"fear",3:"happy",4:"neutral",5:"sad",6:"surprise"}
s1=[]
m1=[]
count=0
for m in range(7):
    print("Converting ",disc[m])
    for img in glob.glob("C:/Users/archi/OneDrive/Desktop/DMBI_OEP/images/"+disc[m]+"/*.jpg"):
        #print(img)
        n= cv2.imread(img)
        n=cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
        s=""
        for i in range(0,48):
            for j in range(0,48):
                s=s+' '+str(n[i][j])    
        s1.append(s)
        m1.append(m)
        count += 1
df=pd.DataFrame({"pixels":s1,"class":m1})
#writer = pd.ExcelWriter("H:/7th Sem/DBMI/5_OEP/dataset/training/train2.csv")
df.to_csv("C:/Users/archi/OneDrive/Desktop/DMBI_OEP/dataset/train.csv",index=False)
print("Total Images processed: ",count)