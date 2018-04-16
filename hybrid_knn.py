import numpy as np
import time
from sklearn.model_selection import train_test_split
import pandas as pd
import math
import heapq
start_time = time.time()
def PAA(array,M):
    result=[]
    N=len(array)
    #print(N)
    #print(M)
    for i in range(M):
        #print(i)
        result.append(0)
    for i in range(M*N):
        idx=int(i/N)
        pos=int(i/M)

        result[idx]=result[idx]+array[pos]

    for i in range(M):
        result[i]=result[i]/N

    return result

def distance(a,b):
        dx=abs(a[0]-b[0])
        dy=abs(a[2] - b[2])
        dz=abs(a[1]-b[1])
        #return dx+dz+dy
        return math.sqrt(dx*dx+dy*dy+dz*dz)

def dynamicTimeWarp_window(seqA, seqB,answer):

    window_size=5
    inf=99999999
    numRows, numCols = len(seqA), len(seqB)

    cost = [[0 for _ in range(numCols+1)] for _ in range(numRows+1)]

    for i in range(0,numCols+1):
        for j in range(0,numRows+1):
            cost[i][j]=inf

    cost[0][0]=0
    for i in range(1,numCols+1):
        row_min=9999999
        for j in range(max(1,i-window_size),min(numCols+1,i+window_size)):
            choices =min( cost[i - 1][j], cost[i][j - 1], cost[i - 1][j - 1])
            temp = choices + distance(seqA[i-1], seqB[j-1])
            if(row_min>temp):
                row_min=temp
            cost[i][j] = temp
        if(row_min>answer):
            #print("row_min = "+str(row_min)+" answer = "+str(answer))
            return -1

    return cost[-1][-1]    	




testxl=pd.ExcelFile("3D_handwriting_demo.xlsx")
test_X_excel=pd.read_excel(io=testxl,sheetname=0,header=None)
test_Y_excel=pd.read_excel(io=testxl,sheetname=1,header=None)
test_Z_excel=pd.read_excel(io=testxl,sheetname=2,header=None)
#answer_test = pd.read_excel(io=testxl, sheetname=3, header=None)
test_X = np.array(test_X_excel.values)
test_Y = np.array(test_Y_excel.values)
test_Z = np.array(test_Z_excel.values)

test_newX = []
for row in test_X:
    tempX = []
    for i in row:
        if not math.isnan(i):
            tempX.append(i)
    test_newX.append(tempX)

test_newY = []
for row in test_Y:
    tempX = []
    for i in row:
        if not math.isnan(i):
            tempX.append(i)
    test_newY.append(tempX)

test_newZ = []
for row in test_Z:
    tempX = []
    for i in row:
        if not math.isnan(i):
            tempX.append(i)
    test_newZ.append(tempX)

min_dim=99999999999

for i in range(len(test_newX)):
    if min_dim>len(test_newX[i]):
        min_dim=len(test_newX[i])

paa_test_X=[]
paa_test_Y=[]
paa_test_Z=[]
if (min_dim<20):
    for row in test_newX:
        if len(row)<20:
            for i in range(20-len(row)):
                row.append(np.nan)
            temp = pd.Series(row)
            row=temp.interpolate()
            paa_test_X.append(row)
        else:
            row = PAA(row, 20)
            paa_test_X.append(row)
    for row in test_newY:
        if len(row)<20:
            for i in range(20-len(row)):
                row.append(np.nan)
            temp = pd.Series(row)
            row = temp.interpolate()
            paa_test_Y.append(row)
        else:
            row = PAA(row, 20)
            paa_test_Y.append(row)
    for row in test_newZ:
        if len(row)<20:
            for i in range(20-len(row)):
                row.append(np.nan)
            temp = pd.Series(row)
            row = temp.interpolate()
            paa_test_Z.append(row)
        else:
            row = PAA(row, 20)
            paa_test_Z.append(row)

else:
    for row in test_newX:
        row = PAA(row, 20)
        paa_test_X.append(row)
    for row in test_newY:
        row = PAA(row, 20)
        paa_test_Y.append(row)
    for row in test_newZ:
        row = PAA(row, 20)
        paa_test_Z.append(row)

test_data=[]
for i in range(len(paa_test_X)):
    tetemp=[]
    for j in range(len(paa_test_X[i])):
        temp=[paa_test_X[i][j],paa_test_Y[i][j],paa_test_Z[i][j]]
        tetemp.append(temp)
    test_data.append(tetemp)

min_dim=20

xl = pd.ExcelFile("3D_handwriting_train.xlsx")
X_excel = pd.read_excel(io=xl, sheetname=0, header=None)
Y_excel = pd.read_excel(io=xl, sheetname=1, header=None)
Z_excel = pd.read_excel(io=xl, sheetname=2, header=None)
answer_excel = pd.read_excel(io=xl, sheetname=3, header=None)

X = np.array(X_excel.values)
Y = np.array(Y_excel.values)
Z = np.array(Z_excel.values)
train_answer = np.array(answer_excel.values).flatten().transpose()

newX=[]
for row in X:
    tempX=[]
    for i in row:
        if not math.isnan(i):
            tempX.append(i)
    newX.append(tempX)

newY=[]
for row in Y:
    tempX=[]
    for i in row:
        if not math.isnan(i):
            tempX.append(i)
    newY.append(tempX)

newZ=[]
for row in Z:
    tempX=[]
    for i in row:
        if not math.isnan(i):
            tempX.append(i)
    newZ.append(tempX)



paa_X=[]
for row in newX:
    row=PAA(row,min_dim)
    paa_X.append(row)

paa_Y=[]
for row in newY:
    row=PAA(row,min_dim)
    paa_Y.append(row)

paa_Z=[]
for row in newZ:
    row=PAA(row,min_dim)
    paa_Z.append(row)

train_data=[]
for i in range(len(paa_X)):
    tetemp=[]
    for j in range(len(paa_X[i])):
        temp=[paa_X[i][j],paa_Y[i][j],paa_Z[i][j]]
        tetemp.append(temp)
    train_data.append(tetemp)


#train_data, test_data, train_answer, test_answer = train_test_split(total_XYZ, answer, test_size=0.2)




result=[]
k=5
for i in range(len(test_data)):
    min_distance=999999999999999

    heap=[]
    index=[[0 for _ in range(k)] for _ in range(2)]
    size=0

    for j in range(int(len(train_data))):
        now=dynamicTimeWarp_window(train_data[j],test_data[i],min_distance)

        if min_distance>now and now!=-1:
            if size<k:
                heapq.heappush(heap,-now)

                index[0][size]=now
                index[1][size]=ord(train_answer[j])-97
                size=size+1
                if size==k:
                    min_distance=-heap[0]
            else:
                temp=heapq.heappop(heap)
                heapq.heappush(heap,-now)
                min_distance=-heap[0]
                for l in range(k):
                    if(index[0][l]==-temp):
                        index[0][l]=now
                        index[1][l]=ord(train_answer[j])-97

    same=False
    alpha=[0 for _ in range(26)]
    for l in range(26):
        alpha[l]=0
    for l in range(k):
        alpha[index[1][l]]=alpha[index[1][l]]+1

    MAX_K=-1
    index_k=-1
    candi=[]
    answer='a'
    for l in range(26):

        temp = alpha[l]
        if (temp > MAX_K):
            MAX_K = temp
            index_k=l
    count=0
    for l in range(26):
        if(alpha[l]==MAX_K):
            if(count>0):
                same=True
            candi.append(l)
            count=count+1

    if(same==False):
        answer=chr(candi[0]+97)
    else:
        min_heap=[]
        for e in range(k):
            temp=-heapq.heappop(heap)
            heapq.heappush(min_heap, temp)
        for e in range(k):
            find=False
            temp = heapq.heappop(min_heap)
            for o in range(k) :
                if (index[0][o]==temp):
                    for u in range(len(candi)):
                        if (index[1][o]==candi[u]):
                            find=True
                            answer = chr(candi[u] + 97)
                            break

                if (find==True):
                    break
            if (find==True):
                break

    print("예측 " + str(answer))

    result.append(answer)

end_time = time.time()
print (end_time - start_time)
result=np.array(result)
f = open('result3.txt','w')
f.writelines('\n'.join([str(i) for i in result.tolist()]))
f.close()