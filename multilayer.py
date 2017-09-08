import pandas as pd
import numpy as np

alpha = 0.1
threshold = 0.5
df = pd.read_csv('IRIS.csv')
df = df.sample(frac=1)

y = []
for item in df['class']:
    if item == "Iris-setosa":
        y.append(0)
    else:
        y.append(1)

X = df.as_matrix()
n = 5
X = X[:, :-1]

weights_input_h1 = [[1/(n*5+5)]*4]*5
weights_input_h1 = np.asarray(weights_input_h1)# weights from input to hidden layer 1

weights_h1_output = [1/(5+1)]*5
bias_hidden = [5]*5
bias_output = 1

def get_fold(dataset, k):  # function to return the appropriate train and test sets
    dataset = np.array(dataset)

    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for i in range(len(dataset)):
        if i % 10 == k:
            x_test.append(dataset[i])
            y_test.append(y[i])
        else:
            x_train.append(dataset[i])
            y_train.append(y[i])
    return x_train, x_test, y_train, y_test
overall_sum = 0

for i in range(10):  # this is for the cross validation
    total_sum = 0
    false_positive = 0
    false_negative = 0
    true_positive = 0
    true_negative = 0
    num_positive = 0
    num_negative = 0
    x_train, x_test, y_train, y_test = get_fold(X, i)
    # Need to find the output of the first outer layer
    outputlayer_1 = [0]*5
    for i in range (500):
        for num, data in enumerate(x_train):

            for ide in range(5):  # 5 is the number of hidden neurons
                val = (np.sum(np.multiply(weights_input_h1[ide],data)))+bias_hidden[ide]
                outputlayer_1[ide] = 1/(1+np.exp(-1*val))

            val = (np.sum(np.multiply(outputlayer_1, weights_h1_output)))+bias_output
            outputlayer_2 = 1/(1+np.exp(-1*val))

            # so now we have both the output and input layers done. We need to calculate errors
            error_outputlayer_1 = [0]*5
            error_outputlayer_2 = outputlayer_2*(1-outputlayer_2)*(y_train[num]-outputlayer_2)
            for x in range(5):
                error_outputlayer_1[x] = outputlayer_1[x]*(1-outputlayer_1[x])*error_outputlayer_2*weights_h1_output[x]
            # Now we need to update the weights starting from the outermost weights
            for x, weight in enumerate(weights_h1_output):
                weights_h1_output[x] = weight+alpha*outputlayer_1[x]*error_outputlayer_2
            # Now we need to update the next set of weights
            for row_num, rows in enumerate(weights_input_h1):
                for _, weight in enumerate(rows):
                    weights_input_h1[row_num][_] = weight+alpha*error_outputlayer_1[row_num]*data[_]
                #     rows[_] = weight+alpha*x_train[_]*error_outputlayer_1[row_num]
            # Now comes the bias updation part
            bias_output = bias_output+alpha*error_outputlayer_2
            for _, item in enumerate(bias_hidden):
                bias_hidden[_] = item+alpha*error_outputlayer_1[_]
        # Now that the weight tweaking is complete we check if whatever we produce is alright
    for num, data in enumerate(x_test):
        for ide in range(5):  # 5 is the number of hidden neurons
            outputlayer_1[ide] = 1 / (1 + (np.exp(-1 * (np.sum(np.multiply(data, weights_input_h1[ide])) + bias_hidden[ide]))))
        outputlayer_2 = 1 /(1 + np.exp(-1 * (np.sum(np.multiply(outputlayer_1, weights_h1_output)) + bias_output)))
        #     outputlayer_1[ide] = np.sum(np.multiply(data, weights_input_h1[ide]))+bias_hidden[ide]
        # outputlayer_2 = np.sum(np.multiply(outputlayer_1, weights_h1_output))+bias_output

        if outputlayer_2>0.6:
            outputlayer_2 = 1
        else:
            outputlayer_2 = 0
        if y_test[num] == outputlayer_2:
            total_sum += 1
    if y_test[num] == 1 and outputlayer_2 == 1:
        true_positive += 1
        num_positive += 1
    if y_test[num] == 0 and outputlayer_2 == 0:
        true_negative += 1
        num_negative += 1
    if y_test[num] == 0 and outputlayer_2 == 1:
        false_positive += 1
        num_negative += 1
    if y_test[num] == 1 and outputlayer_2 == 0:
        false_negative += 1
        num_positive += 1
    try:
        precision = true_positive / (true_positive + false_positive)
    except:
        precision = 0
    try:
        recall = true_positive / (true_positive + false_negative)
    except:
        recall = 0
    print("Accuracy: "+str(total_sum/len(x_test))+" Precision: "+str(precision)+" Recall: "+str(recall))
    overall_sum += (total_sum/len(x_test))
