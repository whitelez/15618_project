import xgboost as xgb
import time

# read in data
dtrain = xgb.DMatrix('data/agaricus.txt.train2')
dtest = xgb.DMatrix('data/agaricus.txt.test2')

# specify parameters via map
param = {'max_depth':3, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
num_round = 3

#calculate time
start_time = time.time()

# make xgb train
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)

#calculate end time
end_time = time.time()
duration = end_time - start_time

round_preds = [int(round(elem, 0)) for elem in preds ]
observe = list();

with open('data/agaricus.txt.test2') as f:
    lines = f.readlines()


for line in lines:
    mylist = line.split(" ")
    observe.append(int(mylist[0]))


error = 0;
total = 0;
error_index = list();
for j in xrange(0, len(observe)):
    if(observe[j] != round_preds[j]):
        error += 1;
        error_index.append(j)
    total += 1;

rate = float(error) / float(total)
print "Total "+str(total) + " error "+ str(error) + " rate " + str(rate) + " duration " + str(duration)
print error_index
