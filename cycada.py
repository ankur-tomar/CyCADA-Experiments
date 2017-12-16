import torch
from torch.autograd import Variable
from crossdomainnet import CrossDomainNet, Discriminator

class Adversarial(nn.Module):

	def __init__(self, discriminator, generator):
        super(Adversarial, self).__init__()
		self.batch_size = 32
		self.discriminator = discriminator
		self.generator = generator
		self.doptimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)
		self.goptimizer = torch.optim.Adam(self.generator.parameters(), lr=0.001)
		self.loss = torch.nn.CrossEntropyLoss()

	def train(self, source_data, target_data, grad=True):
		epochs = 100
		for iter in range(epochs):
			sdata = get_batch(source_data)
			slabel = Variable(torch.zeros(self.batch_size), requires_grad=False)
			self.pred = self.discriminator.forward(sdata)
			self.dloss = loss(self.pred, slabel)
			if grad:
				self.doptimizer.zero_grad()
				self.dloss.backward()
				self.doptimizer.step()

			gdata = get_batch(target_data)
			gdata = self.generator.forward(gdata)
			glabel = Variable(torch.ones(self.batch_size), requires_grad=False)
			self.pred = self.discriminator.forward(gdata)
			self.dloss = loss(self.pred, glabel)
			if grad:
				self.doptimizer.zero_grad()
				self.dloss.backward()
				self.doptimizer.step()

			#The generator wants discriminator to think the data is from source distribution
			self.gloss = loss(self.pred, slabel)
			if grad:
				self.goptimizer.zero_grad()
				self.gloss.backward()
				self.goptimizer.step()


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


def get_batch(data):
	l = data.shape[0]
	index = np.random.randint(l, size=[self.batch_size])
	return Variable(torch.from_numpy(data[index]).type(torch.FloatTensor), requires_grad=False)


def main():

	source_data = 
	target_data = 

	t2s_d = Discriminator()
	t2s_g = CrossDomainNet()
	t2s = Adversarial(discriminator, generator)
	t2s.train(source_data, target_data)

	s2t_d = Discriminator()
	s2t_g = CrossDomainNet()
	s2t = Adversarial(s2t_d, s2t_g)
	s2t.train(target_data, source_data)

	autoencoder = Reconstruction(s2t, t2s)
	autoencoder.train(source_data, target_data)

if __name__ == '__main__':
	main()