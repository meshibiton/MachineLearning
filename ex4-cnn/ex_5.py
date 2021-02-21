import os

from gcommand_dataset import GCommandLoader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


labels_num = 30
# hyper parameters
batch_size = 30
num_workers = 20
lr = 0.001
epochs = 12


class convNet(nn.Module):
    def __init__(self):
        super(convNet, self).__init__()
        # build conv layers
        # first parm- dim, second-channels,third-filter size
        self.conv0 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        # take the max
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        # build dropout
        self.dropout = nn.Dropout()
        # build linear layout
        self.fc0=nn.Linear(64*10*6,256)
        self.fc1 = nn.Linear(256,128)
        self.fc2 = nn.Linear(128, labels_num)
        self.bn0 = nn.BatchNorm2d(32)
        self.bn1=nn.BatchNorm2d(64)
        self.bn2=nn.BatchNorm2d(64)
        self.bn3=nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.conv0(x)
        x = F.relu(self.bn0(x))
        x = self.pool(x)
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# the function train the model
def train(model, train_input, optimizer, epoch):
    # pass through all the audios of train_input
    for batch_idx, (data, labels) in enumerate(train_input):
        correct = 0
        optimizer.zero_grad()
        # forward propagation
        output = model(data)
        # calculate loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, labels)
        # calculate back propagation
        loss.backward()
        optimizer.step()
        # find y_hat
        pred = output.data.max(1, keepdim=False)[1]
        # calculate accuracy
        correct += pred.eq(labels.view_as(pred)).cpu().sum().item()
    # print acc for each epoch
    # if (batch_idx + 1) % 1000 == 0:
    # 	print("epoch number %d" % epoch)
    # 	print('acc: {:.2f}%'.format((correct / len(labels)) * 100))


# the function do the validation algorithm
def validation(model, valid_input):
    correct = 0
    total = 0
    # pass through all the audios of valid_input
    for data, labels in valid_input:
        # forward propagation
        output = model(data)
        # find y_hat
        pred = output.data.max(1, keepdim=False)[1]
        # calculate accuracy
        for i in range(0, len(labels)):
            if (pred[i] == labels[i]):
                correct = correct + 1
        total += len(labels)

    # print the validation acc
    print('Accuracy of the validation : {} %'.format((correct / total) * 100))


# the function find the prediction of the test
def find_prediction(model, test_input):
    # set prediction array
    prediction_arr = []
    for batch_idx, (data, labels) in enumerate(test_input):
        # forward propagation
        output = model(data)
        # find y_hat
        y_hat = output.data.max(1, keepdim = False)[1]
        # add the y_hat (the current prediction) to prediction_arr
        for j in range(0, len(y_hat)):
            prediction_arr.append(int(y_hat[j]))
    return prediction_arr


# the function get the audio files names
def get_files_name(test_input_files):
    # set array of the audio files name
    name_arr = []
    for audio_name in test_input_files:
        curr_audio_name = audio_name[0].split('/')[len(audio_name[0].split('/')) - 1]
        curr_audio_name = curr_audio_name.split('\\')[len(curr_audio_name.split('\\')) - 1]
        # add another audio file name to name_arr
        name_arr.append(curr_audio_name)
    return name_arr


# the test function,return y_hat
def test(model, test_input, test_input_files):
        # find y_hat
        prediction_arr = find_prediction(model, test_input)
        # extract audio name
        name_arr = get_files_name(test_input_files)
        return name_arr, prediction_arr


# the function load the input files to train, validation and test
def load_data():
    dataset = GCommandLoader('gcommands/train')
    train_input = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True,
                                              num_workers = num_workers, pin_memory = True, sampler = None)

    dataset = GCommandLoader('gcommands/valid')
    valid_input = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True,
                                              num_workers = num_workers, pin_memory = True, sampler = None)

    datasetTest = GCommandLoader('gcommands/test')
    test_input = torch.utils.data.DataLoader(datasetTest, batch_size = batch_size, shuffle = False,
                                             num_workers = num_workers, pin_memory = True, sampler = None)

    return train_input, valid_input, test_input, datasetTest.spects


# the func change the order of the lines result in asc order by the file name
# and change the y_hat from num to category
def orderTest(name_list,y_hat_arr):
    # get the categories
    classes = [d for d in os.listdir('gcommands/train') if os.path.isdir(os.path.join('gcommands/train', d))]
    classes.sort()
    indx_to_classes = dict(enumerate(classes))
    # go through y_hat and change each one
    y_hat_newArr = []
    for y in y_hat_arr:
        lable_y = indx_to_classes[y]
        y_hat_newArr.append(lable_y)
    # put the line's result into array and sort it
    result = []
    for i in range(0, len(name_list)):
        line = name_list[i] + "," + y_hat_newArr[i]
        result.append(line)
    result = sorted(result, key=lambda x: int(x.split('.')[0]))
    return result


# the main function
def main():
    train_input, valid_input, test_input, datasetTest = load_data()
    model = convNet()
    optimizer = optim.Adam(model.parameters() , lr = lr)
    # train
    model.train()
    for epoch in range(0 ,epochs):
        train(model, train_input, optimizer, epoch)
    # validation
    model.eval()
    with torch.no_grad():
        validation(model, valid_input)
    # test
    name_list, y_hat_arr = test(model, test_input, datasetTest)
    results=orderTest(name_list,y_hat_arr)
    # write the test's prediction to "test_y" file
    test_file_result = open("test_y", "w")
    for result in results:
        test_file_result.write(result)
        test_file_result.write('\n')
    test_file_result.close()


if __name__=='__main__':
    main()