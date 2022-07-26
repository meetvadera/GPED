## Below code is taken from: https://github.com/dizam92/pyTorchReg. We do not own the copyright for this code.

class _Regularizer(object):
    """
    Parent class of Regularizers
    """

    def __init__(self, model):
        super(_Regularizer, self).__init__()
        self.model = model

    def regularized_param(self, param_weights, reg_loss_function):
        raise NotImplementedError

    def regularized_all_param(self, reg_loss_function):
        raise NotImplementedError


class L1Regularizer(_Regularizer):
    """
    L1 regularized loss
    """
    def __init__(self, model, lambda_reg=0.01):
        super(L1Regularizer, self).__init__(model=model)
        self.lambda_reg = lambda_reg

    def regularized_param(self, param_weights, reg_loss_function):
        reg_loss_function += self.lambda_reg * L1Regularizer.__add_l1(var=param_weights)
        return reg_loss_function

    def regularized_all_param(self, reg_loss_function):
        for model_param_name, model_param_value in self.model.named_parameters():
            if model_param_name.endswith('weight'):
                reg_loss_function += self.lambda_reg * L1Regularizer.__add_l1(var=model_param_value)
        return reg_loss_function

    @staticmethod
    def __add_l1(var):
        return var.abs().sum()


class L2Regularizer(_Regularizer):
    """
       L2 regularized loss
    """
    def __init__(self, model, lambda_reg=0.01):
        super(L2Regularizer, self).__init__(model=model)
        self.lambda_reg = lambda_reg

    def regularized_param(self, param_weights, reg_loss_function):
        reg_loss_function += self.lambda_reg * L2Regularizer.__add_l2(var=param_weights)
        return reg_loss_function

    def regularized_all_param(self, reg_loss_function):
        for model_param_name, model_param_value in self.model.named_parameters():
            if model_param_name.endswith('weight'):
                reg_loss_function += self.lambda_reg * L2Regularizer.__add_l2(var=model_param_value)
        return reg_loss_function

    @staticmethod
    def __add_l2(var):
        return var.pow(2).sum()


class ElasticNetRegularizer(_Regularizer):
    """
    Elastic Net Regularizer
    """
    def __init__(self, model, lambda_reg=0.01, alpha_reg=0.01):
        super(ElasticNetRegularizer, self).__init__(model=model)
        self.lambda_reg = lambda_reg
        self.alpha_reg = alpha_reg

    def regularized_param(self, param_weights, reg_loss_function):
        reg_loss_function += self.lambda_reg * \
                                     (((1 - self.alpha_reg) * ElasticNetRegularizer.__add_l2(var=param_weights)) +
                                      (self.alpha_reg * ElasticNetRegularizer.__add_l1(var=param_weights)))
        return reg_loss_function

    def regularized_all_param(self, reg_loss_function):
        for model_param_name, model_param_value in self.model.named_parameters():
            if model_param_name.endswith('weight'):
                reg_loss_function += self.lambda_reg * \
                                 (((1 - self.alpha_reg) * ElasticNetRegularizer.__add_l2(var=model_param_value)) +
                                  (self.alpha_reg * ElasticNetRegularizer.__add_l1(var=model_param_value)))
        return reg_loss_function

    @staticmethod
    def __add_l1(var):
        return var.abs().sum()

    @staticmethod
    def __add_l2(var):
        return var.pow(2).sum()


class GroupSparseLassoRegularizer(_Regularizer):
    """
    Group Sparse Lasso Regularizer
    """
    def __init__(self, model, lambda_reg=0.01):
        super(GroupSparseLassoRegularizer, self).__init__(model=model)
        self.lambda_reg = lambda_reg
        self.reg_l2_l1 = GroupLassoRegularizer(model=self.model, lambda_reg=self.lambda_reg)
        self.reg_l1 = L1Regularizer(model=self.model, lambda_reg=self.lambda_reg)

    def regularized_param(self, param_weights, reg_loss_function):
#         reg_loss_function = self.lambda_reg * (
#                     self.reg_l2_l1.regularized_param(param_weights=param_weights, reg_loss_function=reg_loss_function)
#                     + self.reg_l1.regularized_param(param_weights=param_weights, reg_loss_function=reg_loss_function))
        reg_loss_function = self.lambda_reg * (
                    self.reg_l2_l1.regularized_param(param_weights=param_weights, reg_loss_function=reg_loss_function))

        return reg_loss_function

    def regularized_all_param(self, reg_loss_function):
#         reg_loss_function = self.lambda_reg * (
#                 self.reg_l2_l1.regularized_all_param(reg_loss_function=reg_loss_function)
#                 + self.reg_l1.regularized_all_param(reg_loss_function=reg_loss_function))
        reg_loss_function = self.lambda_reg * (
                self.reg_l2_l1.regularized_all_param(reg_loss_function=reg_loss_function))

        return reg_loss_function


class GroupLassoRegularizer(_Regularizer):
    """
    GroupLasso Regularizer:
    The first dimension represents the input layer and the second dimension represents the output layer.
    The groups are defined by the column in the matrix W
    """
    def __init__(self, model, lambda_reg=0.01):
        super(GroupLassoRegularizer, self).__init__(model=model)
        self.lambda_reg = lambda_reg

    def regularized_param(self, param_weights, reg_loss_function, group_name='input_group'):
        if group_name == 'input_group':
            reg_loss_function += self.lambda_reg * GroupLassoRegularizer.__inputs_groups_reg(
                        layer_weights=param_weights)  # apply the group norm on the input value
        elif group_name == 'hidden_group':
            reg_loss_function += self.lambda_reg * GroupLassoRegularizer.__inputs_groups_reg(
                        layer_weights=param_weights)  # apply the group norm on every hidden layer
        elif group_name == 'bias_group':
            reg_loss_function += self.lambda_reg * GroupLassoRegularizer.__bias_groups_reg(
                        bias_weights=param_weights)  # apply the group norm on the bias
        else:
            print(
                'The group {} is not supported yet. Please try one of this: [input_group, hidden_group, bias_group]'.format(
                    group_name))
        return reg_loss_function

    def regularized_all_param(self, reg_loss_function):
        for model_param_name, model_param_value in self.model.named_parameters():
            if model_param_name.endswith('weight'):
                reg_loss_function += self.lambda_reg * GroupLassoRegularizer.__inputs_groups_reg(
                    layer_weights=model_param_value)
            if model_param_name.endswith('bias'):
                    reg_loss_function += self.lambda_reg * GroupLassoRegularizer.__bias_groups_reg(
                        bias_weights=model_param_value)
        return reg_loss_function

    @staticmethod
    def __grouplasso_reg(groups, dim):
        if dim == -1:
            # We only have single group
            return groups.norm(2)
        return groups.norm(2, dim=dim).sum()

    @staticmethod
    def __inputs_groups_reg(layer_weights):
        return GroupLassoRegularizer.__grouplasso_reg(groups=layer_weights, dim=1)

    @staticmethod
    def __bias_groups_reg(bias_weights):
        return GroupLassoRegularizer.__grouplasso_reg(groups=bias_weights, dim=-1)  # ou 0 i dont know yet

def load_model(best_model, seuil=1e-3):
    """We load the best model and applied the particularity of the paper by putting weights < abs(seuil) at 0"""
    # Use their unravel things and test that after
    model = best_model
    weights = []
    sparsity_neurons = []
    for param_name, param_weights in model.named_parameters():
        if param_name.endswith('weight'):
            weights.append(param_weights.data.cpu().numpy())
    weights_copy = deepcopy(weights)
    for i in range(len(weights)):
        for j in range(weights[i].shape[0]):
            weights[i][j][np.abs(weights[i][j]) < seuil] = 0
    for i in range(len(weights)):
        somme = np.sum(weights[i], axis=1)
        sparsity_neurons.append(str(np.where(somme == 0)[0].size))
        print('layer {} got {} amount of sparse neurons'.format(i, (np.where(somme == 0)[0].size)))
    return sparsity_neurons

def get_sparsity(model):
    sparsity_neurons = []
    masks = []
    for param_name, param_weights in model.named_parameters():
        if param_name.endswith('weight'):
            masks.append(model.mask[param_name].cpu().numpy())
    for i, mask in enumerate(masks):
        somme = np.sum(mask, axis=1)
        sparsity_neurons.append(str(np.where(somme == 0)[0].size))
        print('layer {} got {} amount of sparse neurons'.format(i, (np.where(somme == 0)[0].size)))
    return sparsity_neurons

def prune_model(model, threshold=1e-3):
    setattr(model, 'mask', OrderedDict())
    for param_name, param_weights in model.named_parameters():
        if param_name.endswith('weight'):
            zeros = torch.zeros_like(param_weights).cuda()
            mask = ((torch.abs(param_weights.data) > threshold).float() * 1).detach()
            model.mask[param_name] = mask
            param_weights.data.mul_(model.mask[param_name])

def prune_model_on_unit(model, threshold=1e-3):
    setattr(model, 'mask', OrderedDict())
    for param_name, param_weights in model.named_parameters():
        if param_name.endswith('weight'):
            zeros = torch.zeros_like(param_weights).cuda()
            mask = ((torch.abs(param_weights.data) > threshold).float() * 1).detach()
            mask_sum = torch.sum(mask, dim=1)
            new_mask = ((mask_sum > 0.).float() * 1)
            new_mask = torch.transpose(new_mask.repeat(param_weights.shape[1], 1), 1, 0)
            model.mask[param_name] = new_mask
            param_weights.data.mul_(model.mask[param_name])

def reset_pruned_weights(model):
    for param_name, param_weights in model.named_parameters():
        if param_name.endswith('weight'):
            param_weights.data.mul_(model.mask[param_name])