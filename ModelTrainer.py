"""
IN:../PostProcessedData/xxxx.dat
OUT:./ONNX/xxxx.onnx

COMMENT:
Training a model.

"""


import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing
import random
import torch.onnx

#-----------------------#
#  FUNCTION AND CLASS   #
#-----------------------#

# 乱数シード固定
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# データローダーのサブプロセスの乱数seedが固定
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
     

class PostProcessedDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
 

# WENO判定をゆるく，THINC判定をきびしく．連続解でTHINC判定だと数値散逸が大きいため
# ↓
# false-positiveを少なくしたい
# lossが大きいほど最適化が進む
class WeightedFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, omega0=0.25, omega1=0.25, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.gamma = gamma 
        self.omega1 = omega1
        self.omega0 = omega0
        self.reduction = reduction
    
    def forward(self, predict, target):
        pt = predict
        loss0 = -self.omega0 * pt ** self.gamma * (1-target) * torch.log(1-pt+1e-10)
        loss1 = -self.omega1 * (1-pt) ** self.gamma * target * torch.log(pt+1e-10)
        if self.reduction == 'mean':
            loss0 = torch.mean(loss0)
            loss1 = torch.mean(loss1)
            loss = loss0 + loss1
        elif self.reduction == 'sum':
            loss = torch.sum(loss0 + loss1)
        return loss
     

class WeightCrossEntropyLoss(torch.nn.Module):
    def __init__(self, weight, reduction='mean'):
        super(WeightCrossEntropyLoss, self).__init__()
        self.weight = weight 
        self.reduction = reduction

    def forward(self, predict, target):
        log_softmax_outputs = torch.log_softmax(predict, dim = 1)
        # print(f"log_sofmax:{log_softmax_outputs}")
        weight_softmax_outputs = self.weight * log_softmax_outputs
        # print(f"weight_soft:{weight_softmax_outputs}")
        loss = - torch.sum(target * weight_softmax_outputs, dim = 1)
        # print(f"loss:{loss}, target{target}")
        if self.reduction == 'mean':
            loss = torch.mean(loss)
            # print("final loss:{loss}")
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_hidden2, n_output):
        super(Net, self).__init__()
        activate = torch.nn.Sigmoid()
        self.model = torch.nn.Sequential(torch.nn.Linear(n_input, n_hidden),
                                         activate,
                                         torch.nn.Linear(n_hidden, n_hidden2),
                                         activate,
                                         torch.nn.Linear(n_hidden2, n_output),
                                         torch.nn.Sigmoid()
                                        )
    def forward(self, x):
        y = self.model(x)
        return y 
    
class smallnet(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(smallnet, self).__init__()
        activate = torch.nn.Sigmoid()
        self.model = torch.nn.Sequential(torch.nn.Linear(n_input, n_output),
                                        # activate, 
                                        # torch.nn.Linear(n_hidden, n_output),
                                        # activate,
                                        # torch.nn.Linear(n_hidden, n_output)
                                        ) 
    def forward(self, x):
       y = self.model(x)
       return y
class Net2(torch.nn.Module):
    def __init__(self, n_input1, n_hidden1, n_output1, n_input, n_hidden, n_output):
        super(Net2, self).__init__()
        activate = torch.nn.Sigmoid()
        self.subnet = smallnet(n_input1, n_hidden1, n_output1)
        self.mainet = torch.nn.Sequential(torch.nn.Linear(n_input, n_hidden),
                                        activate, 
                                        torch.nn.Linear(n_hidden, n_hidden),
                                        activate, 
                                        torch.nn.Linear(n_hidden, n_output),
                                        torch.nn.Sigmoid()
                                        )
    def forward(self, x):
        x1 = x[:,0:3]
        x2 = x[:,1:4]
        x3 = x[:,2:5]
        # print(x1, x2, x3)
        y1 = self.subnet(x1)
        y2 = self.subnet(x2)
        y3 = self.subnet(x3)
        y4 = x[:,-1].reshape(-1,1)
        # print(f"{y1}\n{y2}\n{y3}\n{y4}")
        input = torch.cat((y1, y2, y3, y4),1)
        # print(input)
        y = self.mainet(input)
        return y

# モデル訓練
def train_model(model, train_loader, test_loader):
    # Train loop ----------------------------
    model.train()  # 学習モードをオン
    train_batch_loss = []
    for data, label in train_loader: # バッチ毎に
        # GPUへの転送
        data, label = data.to(device), label.to(device)
        # 1. 勾配リセット
        optimizer.zero_grad()
        # 2. 推論
        output = model(data)
        # 3. 誤差計算
        loss = criterion(output, label)
        # 4. 誤差逆伝播
        loss.backward()
        # 5. パラメータ更新
        optimizer.step()
        # train_lossの取得
        train_batch_loss.append(loss.item())

    # Test(val) loop ----------------------------
    model.eval()  # 学習モードをオフ
    test_batch_loss = []
    with torch.no_grad():  # 勾配を計算なし
        for data, label in test_loader: # バッチ毎に
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)
            test_batch_loss.append(loss.item())

    return model, np.mean(train_batch_loss), np.mean(test_batch_loss)

def plot_loss(loss_train, loss_test, output_file):
    ep = np.arange(epoch)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ep, loss_train, label="train")
    ax.plot(ep, loss_test, label="test")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Loss")
    ax.set_xlim(0, epoch)
    ax.grid()
    ax.get_xaxis().set_tick_params(pad=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.clf()

def plot_confusion_matrix(cm, classes, output_file, normalize=False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(output_file)
    plt.clf()
    

#--------------#
#  PARAMETER   #
#--------------#
fgs = 6 # short side length of figure based silver ratio (1 : square root of 2)
fmr = 0.125 # figure margin ratio
wsp = 0.2 # the amount of width reserved for space between subplots
hsp = 0.2 # the amount of height reserved for space between subplots
llw = 2 # lines linewidth
alw = 1 # axes linewidth, tick width
mks = 2 ** 8 # marker size
fts = 16 # font size
#ftf = "Times New Roman" # font.family

plt.rcParams["figure.subplot.wspace"] = wsp
plt.rcParams["figure.subplot.hspace"] = hsp
plt.rcParams["lines.linewidth"] = llw
plt.rcParams["lines.markeredgewidth"] = 0
plt.rcParams["axes.linewidth"] = alw
plt.rcParams["xtick.major.width"] = alw
plt.rcParams["ytick.major.width"] = alw
plt.rcParams["xtick.minor.width"] = alw
plt.rcParams["ytick.minor.width"] = alw
plt.rcParams["xtick.minor.visible"] = 'True'
plt.rcParams["ytick.minor.visible"] = 'True'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["font.size"] = fts
plt.rcParams["font.family"] = 'serif'
#plt.rcParams["font.serif"] = ftf
plt.rcParams["mathtext.fontset"] = 'cm'
plt.rcParams["legend.fontsize"] = "small"




#--------------#
#  MAIN CODE   #
#--------------#
if __name__ == '__main__':
    # Setting GPU by Pytorch
    #print(os.cpu_count())
    #mp.set_start_method("spawn", force=True)

    print("Loading device...",end="")
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    #device = torch.device("mps") # for mac
    print("Recognize")
    #print("# of GPU: {}".format(torch.cuda.device_count()))
    #print("Num_0 of GPU Name: {}".format(torch.cuda.get_device_name(torch.device("cuda:0"))))

    #print(torch.cuda.is_available())

    print("Setting seed...",end="")
    setup_seed(1000)
    print("OK")

    # Load dataset
    Data = np.loadtxt("./PostProcessedData/MTBVDdataset.dat") # 5 stencil
    #Data = np.loadtxt("./PostProcessedData/W3TBVDdataset.dat") # 5 stencil
    X = Data[:,0:5]
    sign = Data[:,5].reshape(-1,1)
    label = Data[:,6].reshape(-1,1)

    # Normalize dataset
    X_trans = X.transpose()
    scale = preprocessing.MinMaxScaler(feature_range=(0,1))
    X_trans_scale = scale.fit_transform(X_trans)
    X_scale = X_trans_scale.transpose()
    #X_scale = X

    # Convert to Pytorch tensor and send it to the GPU
    X_unsplit = np.hstack([X_scale, sign])
    X_data = torch.from_numpy(X_unsplit).float()
    y_data = torch.from_numpy(label).float()

    # Split dataset into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=712)
    #X_train = X_train.to(device)
    #X_test = X_test.to(device)
    #y_train = y_train.to(device)
    #y_test = y_test.to(device)

    batch_size = 128
    epoch = 2000
    learning_rate = 0.001
    #width = 8
    train_data = PostProcessedDataset(X_train, y_train)
    test_data = PostProcessedDataset(X_test, y_test)
    #train_data = torch.utils.data.Dataset(X_train, y_train)
    #test_data= torch.utils.data.Dataset(X_test, y_test)
    train_dataLoader = DataLoader(train_data, batch_size, shuffle=True, pin_memory=True)
    test_dataLoader  = DataLoader(test_data,  batch_size, shuffle=True, pin_memory=True)
    #train_dataLoader = DataLoader(train_data, batch_size, shuffle=True, drop_last=True, persistent_workers=True, pin_memory=True, num_workers=2) # for windows
    #test_dataLoader  = DataLoader(test_data,  batch_size, shuffle=True, persistent_workers=True, pin_memory=True, num_workers=2)

    net = Net(6, 6, 6, 1).to(device) # Huangnet
    #net = Net2(3, 8, 2, 7, 8, 1).to(device) # simplenet
    # optimizer = torch.optim.LBFGS(net.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # loss_fn = torch.nn.BCELoss()
    # loss_fn = BCE_WITH_WEIGHT()
    criterion = WeightedFocalLoss()
    train_loss = []
    test_loss = []

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    for i in range(epoch):
        net, train_l, test_l = train_model(net, train_dataLoader, test_dataLoader)
        train_loss.append(train_l)
        test_loss.append(test_l)
        # 10エポックごとにロスを表示
        if i % 10 == 0:
            print("epoch: {0}/{1}, Train loss: {2:.3f}, Test loss: {3:.3f}" \
                .format(i, epoch, train_loss[-1], test_loss[-1]))
        if i % 100 == 0:
            lr_scheduler.step()

    plot_loss(train_loss, test_loss, "./loss_plot.png")

    y_pred = net(X_test.to(device))
        
    for i in range(len(y_pred)):
        if (y_pred[i] < 0.5) :
            y_pred[i] = 0
        else :
            y_pred[i] = 1

    cm = confusion_matrix(y_test, y_pred.detach().cpu().numpy(), labels=[0,1])
    print(cm)
    tn, fp, fn, tp = cm.flatten()
    precision = (tn+tp)/(tn+fp+fn+tp)*100.0
    accuracy = tp/(tp+fp)*100.0
    print('Precision: %.4f%%, Accuracy: %.4f%%' % (precision, accuracy))
    plot_confusion_matrix(cm, [0,1], "./cm_plot.png")

    """
    print(classification_report(y_test.detach().cpu().numpy(), y_pred.detach().cpu().numpy()))
    """

    # Export onnx
    net_cpu = net.cpu()
    net.eval()
    dmmy_input = torch.tensor([[0, 0.166667, 0.333333, 0.5, 0.666667, 1]])
    #print(dmmy_input)
    torch.onnx.export(net_cpu, dmmy_input, "./ONNX/Huangnet_MTBVD.onnx", input_names=['input'],
                    output_names=['output'], dynamic_axes= {'input':
                                {0: 'batch_size'},
                        'output':
                                {0: 'batch_size'}
                        })
