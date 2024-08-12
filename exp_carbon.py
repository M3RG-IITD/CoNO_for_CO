import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from utils.utilities3 import *
from utils.adam import Adam
from utils.params import get_args
from model_dict import get_model
import math
import os

#import wandb
#wandb.login()

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = False
print('imports done')
################################################################
# configs
################################################################
args = get_args()

TRAIN_PATH_1 = os.path.join(args.data_path, './train1.mat')
TRAIN_PATH_2 = os.path.join(args.data_path, './train2.mat')
TEST_PATH = os.path.join(args.data_path, './test_data.mat')

ntrain = args.ntrain
ntest = args.ntest
N = args.ntotal
in_channels = args.in_dim
out_channels = args.out_dim
r1 = args.h_down
r2 = args.w_down
s1 = int(((args.h - 1) / r1) + 1)
s2 = int(((args.w - 1) / r2) + 1)
T_in = args.T_in
T_out = args.T_out

batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
step_size = args.step_size
gamma = args.gamma

model_save_path = args.model_save_path
model_save_name = args.model_save_name

#run = wandb.init(
    # Set the project where this run will be logged
    #project = args.project_name,
    # Provide your desired run name here
   # name = args.run_name)

################################################################
# models
################################################################
model = get_model(args)
print(model)
print(get_num_params(model))

################################################################
# load data and data normalization
################################################################
# reader = MatReader(TRAIN_PATH_1)
# train_a = reader.read_field('a')[:ntrain, ::r1, ::r2, :T_in]
# train_u = reader.read_field('a')[:ntrain, ::r1, ::r2, T_in:T_in + T_out]

# test_a = reader.read_field('a')[-ntest:, ::r1, ::r2, :T_in]
# test_u = reader.read_field('a')[-ntest:, ::r1, ::r2, T_in:T_in + T_out]

# train_a = train_a.reshape(ntrain, s1, s2, T_in)
# test_a = test_a.reshape(ntest, s1, s2, T_in)

reader = MatReader(TRAIN_PATH_1)
train_a_1 = reader.read_field('a')[:, :T_in, ::r1, ::r2]
train_u_1 = reader.read_field('a')[:, T_in:T_in + T_out, ::r1, ::r2]

reader = MatReader(TRAIN_PATH_2)
train_a_2 = reader.read_field('a')[:, :T_in, ::r1, ::r2]
train_u_2 = reader.read_field('a')[:, T_in:T_in + T_out, ::r1, ::r2]

train_a = torch.cat((train_a_1, train_a_2), 0)
train_u = torch.cat((train_u_1, train_u_2), 0)

train_a = train_a.permute(0, 2, 3, 1)
train_u = train_u.permute(0, 2, 3, 1)

reader = MatReader(TEST_PATH)
test_a = reader.read_field('a')[:, :T_in, ::r1, ::r2]
test_u = reader.read_field('a')[:, T_in:T_in + T_out, ::r1, ::r2]

test_a = test_a.permute(0, 2, 3, 1)
test_u = test_u.permute(0, 2, 3, 1)

print(train_a.shape)
print(test_a.shape)
print(train_u.shape)
print(test_u.shape)

# x_normalizer = UnitGaussianNormalizer(train_a)
# x_train = x_normalizer.encode(train_a)
# x_test = x_normalizer.encode(test_a)

# y_normalizer = UnitGaussianNormalizer(train_u)
# y_train = y_normalizer.encode(train_u)
# y_normalizer.cuda()
# y_test = test_u

# train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
#                                            shuffle=True)
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
#                                           shuffle=False)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size,
                                          shuffle=False)
del train_a, train_u, test_u, test_a
del train_a_1, train_a_2, train_u_2, train_u_1
del reader
################################################################
# training and evaluation
################################################################
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)
print('model train')
step = 1
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T_out, step):
            y = yy[..., t:t + step]
            im = model(xx)
            
            # out = y_normalizer.decode(im)
            # y = y_normalizer.decode(y)
        
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_l2_step = 0
    test_l2_full = 0
    model.eval()
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            for t in range(0, T_out, step):
                y = yy[..., t:t + step]
                im = model(xx)
                
                # out = y_normalizer.decode(im)
                
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    t2 = default_timer()
    scheduler.step()
    print(ep, t2 - t1, train_l2_step / ntrain / (T_out / step), train_l2_full / ntrain,
          test_l2_step / ntest / (T_out / step),
          test_l2_full / ntest)
    
   #wandb.log({"epoch": ep, "train_l2_full": train_l2_full / ntrain, "train_l2_step": train_l2_step / ntrain / (T_out / step), "test_l2_full": test_l2_full / ntest,  "test_l2_step": test_l2_step / ntest / (T_out / step),"time": t2-t1})
    
    if ep % step_size == 0:
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        print('save model')
        torch.save(model.state_dict(), os.path.join(model_save_path, model_save_name))
