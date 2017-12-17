import torch
from torch import nn
from torch.autograd import Variable
from crossdomainnet import CrossDomainNet, Discriminator
import numpy as np
import pickle, os

class Adversarial(nn.Module):

	def __init__(self, discriminator, generator):
		super(Adversarial, self).__init__()
		self.batch_size = 128
		self.discriminator = discriminator
		self.generator = generator
		self.doptimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)
		self.goptimizer = torch.optim.Adam(self.generator.parameters(), lr=0.001)
		self.loss = torch.nn.CrossEntropyLoss()

	def train(self, source_data, target_data, grad=True):
		epochs = 5000
		for iter in range(epochs):
			self.dloss = Variable(torch.from_numpy(np.array([2])), requires_grad=False)
			self.gloss = Variable(torch.from_numpy(np.array([5])), requires_grad=False)
			print("Discrim loss during d_training")
			while float(self.dloss.data)>0.7:
				sdata = get_batch(source_data, self.batch_size)
				slabel = Variable(torch.zeros(self.batch_size).type(torch.LongTensor), requires_grad=False)
				self.pred = self.discriminator.forward(sdata)
				self.dloss = self.loss(self.pred, slabel)
				print(self.dloss)
				if grad:
					self.doptimizer.zero_grad()
					print("Fine till here")
					self.dloss.backward(retain_graph=True)
					print("Reaches here")
					self.doptimizer.step()

			print("Generator loss during g_training")
			while float(self.gloss.data)>0.7:
				gdata = get_batch(target_data, self.batch_size)
				gdata = self.generator.forward(gdata)
				glabel = Variable(torch.ones(self.batch_size).type(torch.LongTensor), requires_grad=False)
				self.pred = self.discriminator.forward(gdata)
				self.dloss = self.loss(self.pred, glabel)
				print(self.dloss)
				if grad:
					self.doptimizer.zero_grad()
					self.dloss.backward(retain_graph=True)
					self.doptimizer.step()

				#The generator wants discriminator to think the data is from source distribution
				self.gloss = self.loss(self.pred, slabel)
				print(self.gloss)
				if grad:
					self.goptimizer.zero_grad()
					self.gloss.backward(retain_graph=True)
					self.goptimizer.step()

				with open('../models/discrim%i.pt'%iter, 'w') as f:
					torch.save(self.discriminator.state_dict(), f)
				with open('../models/generator%i.pt'%iter, 'w') as f:
					torch.save(self.generator.state_dict(), f)


class Reconstruction(nn.Module):

	def __init__(self, s2t, t2s):
		super(Reconstruction, self).__init__()
		self.s2t = s2t
		self.t2s = t2s
		self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

	def train(self, source_data, target_data):
		epochs = 100
		loss = torch.nn.MSELoss()
		for iter in range(epochs):
			sdata = get_batch(source_data)
			srecon = self.t2s.forward(self.s2t.forward(sdata))
			self.sloss = loss(srecon, sdata)

			tdata = get_batch(target_data)
			trecon = self.s2t.forward(self.t2s.forward(tdata))
			self.tloss = loss(trecon, tdata)

			self.total_loss = self.sloss+self.tloss
			self.optimizer.zero_grad()
			self.total_loss.backward()
			self.optimizer.step()


def get_batch(data_path, batch_size):
	files = os.listdir(data_path)
	rand = np.random.randint(len(files))
	print(data_path+files[rand])
	with open(data_path+files[rand], 'r') as f:
		data = pickle.load(f)
	l = data.shape[0]
	indices = np.random.randint(l, size=[batch_size])
	if data_path.endswith('target_data/'):
		batch = np.zeros((batch_size, 3, 80, 160))
		images = data[indices]
		for i in range(batch_size):
			batch[i, 0, :, :] = images[i, :, :, 0]/np.max(images[i, :, :, 0])
			batch[i, 1, :, :] = images[i, :, :, 1]/np.max(images[i, :, :, 1])
			batch[i, 2, :, :] = images[i, :, :, 2]/np.max(images[i, :, :, 2])
	else:
		batch = data[indices]
	return Variable(torch.from_numpy(batch).type(torch.FloatTensor), requires_grad=False)


def main():

	source_data = "/mnt/nfs/scratch1/manpreetkaur/steer/source_data/"
	target_data = "/mnt/nfs/scratch1/manpreetkaur/steer/target_data/"

	t2s_d = Discriminator()
	t2s_g = CrossDomainNet()
	t2s = Adversarial(t2s_d, t2s_g)
	t2s.train(source_data, target_data)

	s2t_d = Discriminator()
	s2t_g = CrossDomainNet()
	s2t = Adversarial(s2t_d, s2t_g)
	s2t.train(target_data, source_data)

	autoencoder = Reconstruction(s2t, t2s)
	autoencoder.train(source_data, target_data)

if __name__ == '__main__':
	main()
