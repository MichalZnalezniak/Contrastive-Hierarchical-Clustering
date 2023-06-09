
import torch

def probability_vec_with_level(feature, level):
        prob_vec = torch.tensor([], requires_grad=True).cuda()
        for u in torch.arange(2**level-1, 2**(level+1) - 1, dtype=torch.long):
            probability_u = torch.ones_like(feature[:, 0], dtype=torch.float32).cuda()
            while(u > 0):
                if u/2 > torch.floor(u/2):
                    # Go left
                    u = torch.floor(u/2) 
                    u = u.long()
                    probability_u *= feature[:, u]
                elif u/2 == torch.floor(u/2):
                    # Go right
                    u = torch.floor(u/2) - 1
                    u = u.long()
                    probability_u *=  1 - feature[:, u]
            prob_vec = torch.cat((prob_vec, probability_u.unsqueeze(1)), dim=1)
        return prob_vec

def tree_loss(tree_output1, tree_output2, batch_size, mask_for_level, mean_of_probs_per_level_per_epoch, tree_level):
    ## TREE LOSS
    loss_value = torch.tensor([0], dtype=torch.float32, requires_grad=True).cuda()

    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda()
    
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels * ~mask
    out_tree = torch.cat([tree_output1, tree_output2], dim=0)

    for level in range(1, tree_level + 1):
        prob_features = probability_vec_with_level(out_tree, level)
        prob_features = prob_features * mask_for_level[level]
        if level == tree_level:
            mean_of_probs_per_level_per_epoch[tree_level] += torch.mean(prob_features, dim=0)
        # Calculate loss on positive classes
        # To avoid nan while calculating sqrt https://discuss.pytorch.org/t/runtimeerror-function-sqrtbackward-returned-nan-values-in-its-0th-output/48702  https://github.com/richzhang/PerceptualSimilarity/issues/69
        loss_value -= torch.mean((torch.bmm(torch.sqrt(prob_features[torch.where(labels > 0)[0]].unsqueeze(1) +  1e-8), torch.sqrt(prob_features[torch.where(labels > 0)[1]].unsqueeze(2) + 1e-8))))
        # Calculate loss on negative classes
        loss_value += torch.mean((torch.bmm(torch.sqrt(prob_features[torch.where(labels == 0)[0]].unsqueeze(1) + 1e-8), torch.sqrt(prob_features[torch.where(labels == 0)[1]].unsqueeze(2) + 1e-8))))
    return loss_value

def regularization_loss(tree_output1, tree_output2,  masks_for_level, tree_level):
    out_tree = torch.cat([tree_output1, tree_output2], dim=0)
    loss_reg = torch.tensor([0], dtype=torch.float32, requires_grad=True).cuda()
    for level in range(1, tree_level+1):
        prob_features = probability_vec_with_level(out_tree, level)
        probability_leaves = torch.mean(prob_features, dim=0)
        probability_leaves_masked = masks_for_level[level] * probability_leaves
        for leftnode in range(0,int((2**level)/2)):
            if not (masks_for_level[level][2*leftnode] == 0 or masks_for_level[level][2*leftnode+1] == 0):
                loss_reg -=   (1/(2**level)) * (0.5 * torch.log(probability_leaves_masked[2*leftnode]) + 0.5 * torch.log(probability_leaves_masked[2*leftnode+1]))
    return loss_reg
