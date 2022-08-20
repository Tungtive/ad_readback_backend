import numpy
import torch

X = numpy.random.uniform(-10, 10, 270).reshape(1, 27, -1)
Y = numpy.random.uniform(-10, 10, 30).reshape(1, 3, -1)
# Y = np.random.randint(0, 9, 10).reshape(1, 1, -1)

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv1 = torch.nn.Conv1d(18, 200, kernel_size = 3, padding='same')
        self.conv2 = torch.nn.Conv1d(200, 200, kernel_size = 3, padding = 'same')
        self.conv3 = torch.nn.Conv1d(200, 3, kernel_size=3, padding='same')
        self.tanh = torch.nn.Tanh()
        self.fc1 = torch.nn.Linear(10, 10)
        # self.tanh2 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(10, 3)
        # self.layer1 = torch.nn.Conv1d(in_channels=18, out_channels=1800, kernel_size=3, padding='same')
        # self.act1 = torch.nn.Tanh()
        #
        # self.layer2 = torch.nn.Conv1d(in_channels=1800, out_channels=100, kernel_size=3, padding='same')
        # self.layer3 = torch.nn.Conv1d(in_channels=100, out_channels=20, kernel_size=2, padding='same')
        # # self.layer4 = torch.nn.Conv1d(in_channels=200, out_channels=20, kernel_size=3, padding='same')
        #
        # # torch.nn.init.uniform(self.layer2.weight, -10, 10)
        # self.fc1 = torch.nn.Linear(in_features=20, out_features=50)
        # self.act2 = torch.nn.Tanh()
        # self.fc2 = torch.nn.Linear(in_features=50, out_features=3)

    def forward(self, x):

        x = self.conv1(x)
        x = self.tanh(x)
        x = self.conv2(x)

        x = self.conv3(x)
        # x = self.relu(x)
        # # print(x.shape)
        # x = x.view(-1, 10)
        # # # # print(x.shape)
        # # x = self.fc1(x)
        # # x = self.tanh2(x)
        # # # x = self.relu(x)
        # x = self.fc1(x)
        # # x = self.tanh(x)
        #
        # x = self.fc2(x)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.act1(x)
        #
        # x = self.layer3(x)
        # x =x.view(-1)
        # x = self.fc1(x)
        # x = self.act2(x)
        # x = self.fc2(x)
        # log_probs = torch.nn.functional.log_softmax(x, dim=1)

        return x.reshape((1, 3, -1))# torch.unsqueeze(x.transpose(0, 1), 0)#

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = torch.nn.Conv1d(in_channels=3, out_channels=200, kernel_size=3, padding='same')
        self.pool1 = torch.nn.MaxPool1d(3, stride = 4)
        # self.act = torch.nn.LeakyReLU()
        self.layer2 = torch.nn.Conv1d(in_channels=200, kernel_size=3, out_channels=200, padding='same')
        self.pool2 = torch.nn.MaxPool1d(3, stride = 4)

        # self.dropout = torch.nn.Dropout()
        # self.layer3 = torch.nn.Conv1d(in_channels=10, kernel_size=3, out_channels=10, padding = 'same')
        self.act2 = torch.nn.Tanh()
        # self.fc1 = torch.nn.Linear(in_features=10, out_features=50)
        # self.act3 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(in_features=200, out_features=50)
        self.fc3 = torch.nn.Linear(in_features=50, out_features=1)


    def forward(self, x):
        x = self.layer1(x)
        # x = self.act(x)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        # x = self.dropout(x)
        # x = self.layer3(x)
        # x = self.act(x)
        x = x.view(-1, 200)#torch.flatten(x).reshape((-1, 10))#(-1, 10))
        # x = self.fc1(x)
        # x = self.act2(x)
        x = self.fc2(x)
        x = self.act2(x)
        # x = self.act3(x)
        x = self.fc3(x)
        return (torch.sigmoid(x))#.sigmoid(x)


class GAN1dCONV(object):

    def __init__(self):
        self.gen = Generator().double()
        self.dis = Discriminator().double()
        self.gen_optim = torch.optim.Adam(list(self.gen.parameters()), lr=0.000001)
        self.dis_optim = torch.optim.Adam(list(self.dis.parameters()), lr = 0.000001)


    # def disloss(self, real_out, fake_out):
    def train(self, X_real_T, Y_real_T, n_epochs = 100):

        batch_size = 50
        n_batches = X_real_T.shape[-1]//batch_size + 1
        loss = torch.nn.BCELoss()

        for e in range(n_epochs):
            loss_g = []
            loss_d = []
            for b in range(n_batches):
                st = b * batch_size
                et = min((b + 1) * batch_size, X_real_T.shape[-1])

                X_real = X_real_T[:, :, st:et]

                Y_real = Y_real_T[:, :, st:et]
                Y_fake = self.gen(torch.tensor(X_real))
                # print(Y_fake.shape)
                lab_real = self.dis(torch.tensor(Y_real))
                lab_fake = self.dis(Y_fake)
                gen_loss = -torch.mean(torch.log(lab_fake))
                rec_loss = torch.mean((Y_fake - Y_real)**2)
                gen_loss = rec_loss
                gen_loss.backward()

                # rec_loss.backward()
                self.gen_optim.step()
                self.dis_optim.zero_grad()

                dis_loss = -torch.mean(torch.log(lab_real) + torch.log(1-self.dis(Y_fake.detach())))
                dis_loss.backward()
                # self.dis_optim.step()
                loss_g.append(gen_loss.detach().cpu().numpy())
                loss_d.append(dis_loss.detach().cpu().numpy())

            print(f' epoch {e + 1} / {n_epochs}, Gen Loss : {sum(loss_g)}, Dis Loss : {sum(loss_d)}')



# model = GAN1dCONV()
# model.train(X, Y)
