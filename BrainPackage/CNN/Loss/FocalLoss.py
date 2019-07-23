import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

"""
class FocalLoss(nn.Module):
	def __init__(self, classes, alpha=None, gamma=2, average=True):
		super(FocalLoss, self).__init__()

        if alpha is None:
        	self.alpha = Variable(torch.ones(classes, 1))
        else:
        	if isinstance(alpha, Variable):
        		self.alpha = alpha
        	else:
        		self.alpha = Variable(alpha)

		self.gamma = gamma
		self.average = average

	def forward(self, inputs, targets):
		batches_num = inputs.size(0)
		classes_num = inputs.size(1)

		p = F.softmax(inputs)
		class_mask = inputs.data.new(batches_num, classes_num).fill_(0)
		class_mask = Variable(class_mask)
		indexes = targets.view(-1, 1)
		class_mask.scatter_(1, indexes.data, 1.0)

		if inputs.is_cuda and not self.alpha.is_cuda:
			self.alpha = self.alpha.cuda()
		alpha = self.alpha[indexes.data.view(-1)]

		probs = (p*class_mask).sum(1).view(-1, 1)
		log_p = probs.log()

		batch_loss = -alpha*(torch.pow((1 - probs), self.gamma))*log_p

		if self.average:
			loss = batch_loss.mean()
		else:
			loss = batch_loss.sum()


		return loss
"""


"""
# TODO: version of pytorch for cuda 7.5 doesn't have the latest features like
# reduce=False argument -- update cuda on the machine and update the code

# TODO: update the class to inherit the nn.Weighted loss with all the additional
# arguments

def convert_labels_to_one_hot_encoding(labels, number_of_classes):

    labels_dims_number = labels.dim()

    # Add a singleton dim -- we need this for scatter
    labels_ = labels.unsqueeze(labels_dims_number)
    
    # We add one more dim to the end of tensor with the size of 'number_of_classes'
    one_hot_shape = list(labels.size())
    one_hot_shape.append(number_of_classes)
    one_hot_encoding = torch.zeros(one_hot_shape).type(labels.type())
    
    # Filling out the tensor with ones
    one_hot_encoding.scatter_(dim=labels_dims_number, index=labels_, value=1)
    
    return one_hot_encoding.byte()

"""
"""
class FocalLoss(nn.Module):
   
    def __init__(self, gamma=1):
        
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, flatten_logits, flatten_targets):
        
        flatten_targets = flatten_targets.data
        
        number_of_classes = flatten_logits.size(1)
        
        flatten_targets_one_hot = convert_labels_to_one_hot_encoding(flatten_targets, number_of_classes)

        all_class_probabilities = F.softmax(flatten_logits)

        probabilities_of_target_classes = all_class_probabilities[flatten_targets_one_hot]

        elementwise_loss =  - (1 - probabilities_of_target_classes).pow(self.gamma) * torch.log(probabilities_of_target_classes)
        
        return elementwise_loss.sum()
"""


class FocalLoss(nn.Module):
    def __init__(self, gamma, alpha_map):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha_map = alpha_map

    def forward(self, output, target):
        P = F.softmax(output)
        f_out = F.log_softmax(output)
        Pt = P.gather(1,torch.unsqueeze(target,1))
        focus_p = torch.pow(1-Pt, self.gamma)
        # alpha = 1.0 #0.25
        alpha = self.alpha_map.gather(1,torch.unsqueeze(target,1)) # add by hanbing
        nll_feature=-f_out.gather(1,torch.unsqueeze(target,1))
        weight_nll = alpha*focus_p*nll_feature
        loss = weight_nll.mean()
        return loss