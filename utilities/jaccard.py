import torch
from torch import nn

def onehot(gt,shape):
    with torch.no_grad():
        shp_y = gt.shape
        #gt = gt.view((shp_y[0], *shp_y[1:]))

        gt = gt.long()
        y_onehot = torch.zeros(shape)
        y_onehot = y_onehot.cuda()
        y_onehot.scatter_(1, gt, 1)
    return y_onehot


def jaccard(pred, gt, is_prob=False):
	b_size, n_classes, x_, y_, z_ = pred.shape
	if not is_prob:
		gt = onehot(gt,pred.shape)

	gt = gt.reshape(b_size, n_classes, -1)
	pred = pred.reshape(b_size, n_classes, -1)
	numerator =  torch.sum(torch.abs(pred - gt),-1)
	denominator = torch.sum(torch.abs(gt)+torch.abs(pred),-1) + numerator + 1e-5

	jaccard = 2*numerator/denominator
	output = torch.mean(jaccard)
	return output


def jaccard_tissue(pred, gt, is_prob=False,mask=None):
	b_size, n_classes, x_, y_, z_ = pred.shape
	if not is_prob:
		gt = onehot(gt,pred.shape)

	gt = gt.reshape(b_size, n_classes, -1)
	pred = pred.reshape(b_size, n_classes, -1)
	if mask is None:
		numerator =  torch.sum(torch.abs(pred - gt),-1)
		denominator = torch.sum(torch.abs(gt)+torch.abs(pred),-1) + numerator + 1e-5
	else:
		mask = torch.stack([mask.reshape(b_size,-1) for i in range(n_classes)],1).to(torch.float)
		numerator =  torch.sum(mask*torch.abs(pred - gt),-1)
		denominator = torch.sum(mask*torch.abs(gt)+mask*torch.abs(pred),-1) + numerator + 1e-5

	jaccard = 2*numerator/denominator

	output = torch.mean(jaccard[:,:7])

	return output


def jaccard_lesion(pred, gt, is_prob=False):
	b_size, n_classes, x_, y_, z_ = pred.shape
	if not is_prob:
		gt = onehot(gt,pred.shape)

	gt = gt.reshape(b_size, n_classes, -1)
	pred = pred.reshape(b_size, n_classes, -1)
	numerator =  torch.sum(torch.abs(pred - gt),-1)
	denominator = torch.sum(torch.abs(gt)+torch.abs(pred),-1) + numerator + 1e-5

	jaccard = 2*numerator/denominator
	output = torch.mean(jaccard[:,7:])
	return output


def jaccard2D(pred, gt, is_prob=False):
	b_size, n_classes, x_, y_ = pred.shape
	if not is_prob:
		gt = onehot(gt,pred.shape)

	gt = gt.reshape(b_size, n_classes, -1)
	pred = pred.reshape(b_size, n_classes, -1)
	numerator =  torch.sum(torch.abs(pred - gt),-1)
	denominator = torch.sum(torch.abs(gt)+torch.abs(pred),-1) + numerator + 1e-5

	jaccard = 2*numerator/denominator
	output = torch.mean(jaccard)
	return output
