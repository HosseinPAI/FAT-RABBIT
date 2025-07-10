import torch
from tqdm import tqdm
#import matplotlib.pyplot as plt
import resnet20
import numpy as np
import heapq


def train(train_loader, model, device, criterion, optimizer, top_weights, num_of_weight=10):
    correct = 0
    total_loss = 0
    iteration = 0

    model.train()
    for data, target in tqdm(train_loader, total=len(train_loader)):
        data = data.to(device)
        target = target.to(device)

        # Restore Previous Weight
        prev_linear_weight = model.linear.weight.clone().to(device)

        # Start Training Model
        optimizer.zero_grad()
        target_pred = model(data)
        loss = criterion(target_pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        target_prediction = target_pred.argmax(dim=1, keepdim=True)
        correct += target_prediction.eq(target.view_as(target_prediction)).sum().item()
        iteration += 1

        # Check The important Weights
        updated_weights = model.linear.weight.clone().to(device)
        diff = torch.abs(updated_weights - prev_linear_weight).to(device)
        diff_flat = diff.view(diff.shape[0] * diff.shape[1]).sort(descending=True)[0][:num_of_weight].to(device)
        for item in diff_flat:
            diff_copy = diff.copy_(diff).to(device)
            diff_copy = torch.where(diff_copy == item, 1., 0.)
            top_weights = top_weights + diff_copy
        '''
        if torch.eq(target, 3).all():
            for item in diff_flat:
                diff_copy = diff.copy_(diff).to(device)
                diff_copy = torch.where(diff_copy == item, 1., 0.)
                cat_top_weights = cat_top_weights + diff_copy
        else:
            for item in diff_flat:
                diff_copy = diff.copy_(diff).to(device)
                diff_copy = torch.where(diff_copy == item, 1., 0.)
                other_top_weights = other_top_weights + diff_copy
        '''
    total_loss /= iteration
    acc = 100. * correct / len(train_loader.sampler)
    return total_loss, acc, top_weights


def test(test_loader, model, device, criterion):
    correct = 0
    iteration = 0
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data = data.to(device)
            target = target.to(device)

            target_pred = model(data)
            loss = criterion(target_pred, target)
            total_loss += loss.item()
            target_prediction = target_pred.argmax(dim=1, keepdim=True)
            correct += target_prediction.eq(target.view_as(target_prediction)).sum().item()
            iteration += 1

        total_loss /= iteration
        acc = 100. * correct / len(test_loader.sampler)

    return total_loss, acc


def final_test(test_loaders, final_model, top_weights, device, class_name, save_path):
    #with torch.no_grad():
    #    model.fc.weight[0, 0] = 1.
    acc = []
    final_model.eval()
    linear_weight = final_model.linear.weight.clone()
    weight_mask = linear_weight * top_weights
    indeces = torch.nonzero(weight_mask)

    with torch.no_grad():
        for item in test_loaders:
            correct = 0
            for data, target in tqdm(item):
                data = data.to(device)
                target = target.to(device)

                target_pred = final_model(data)
                target_prediction = target_pred.argmax(dim=1, keepdim=True)
                correct += target_prediction.eq(target.view_as(target_prediction)).sum().item()

            acc.append(100. * correct / len(item.sampler))

        for i in range(len(class_name)):
            print("Accuracy For Class Without Any Change[{0}]: {1:.3f}".format(class_name[i], acc[i]))

        for idx in indeces:
            final_model.load_state_dict(torch.load(save_path + "Resnet20_FP_CIFAR10_BestModel.pt",
                                                   map_location=torch.device('cpu')))
            #print("Before Changing: ", final_model.linear.weight)
            read_w = weight_mask[idx[0]][idx[1]]
            samples = (read_w * torch.rand(10) + (read_w - read_w / 2.))*10.
            for sample in samples:
                final_model.linear.weight[idx[0], idx[1]] = sample
                #print("After Changing: ", final_model.linear.weight)
                acc = []
                for item in test_loaders:
                    correct = 0
                    for data, target in tqdm(item):
                        data = data.to(device)
                        target = target.to(device)

                        target_pred = final_model(data)
                        target_prediction = target_pred.argmax(dim=1, keepdim=True)
                        correct += target_prediction.eq(target.view_as(target_prediction)).sum().item()

                    acc.append(100. * correct / len(item.sampler))

                print("Result For Linear Weight Index [{0}, {1}] with the "
                      "Original Weight value = {2:.5f} "
                      "and New Weight Value = {3:.5f}\n".format(idx[0], idx[1], linear_weight[idx[0]][idx[1]],
                                                                final_model.linear.weight[idx[0], idx[1]]))
                #print("When Printing: ", final_model.linear.weight)
                for i in range(len(class_name)):
                    print("Accuracy For Class [{0}]: {1:.3f}".format(class_name[i], acc[i]))

        '''
        acc = []
        final_model.linear.weight[indeces[0][0], indeces[0][1]] = 1000.0 #sample
        final_model.linear.weight[indeces[1][0], indeces[1][1]] = 1000.0
        final_model.linear.weight[indeces[2][0], indeces[2][1]] = 1000.0
        final_model.linear.weight[indeces[3][0], indeces[3][1]] = 1000.0
        final_model.linear.weight[indeces[4][0], indeces[4][1]] = 1000.0
        final_model.linear.weight[indeces[5][0], indeces[5][1]] = 1000.0
        final_model.linear.weight[indeces[6][0], indeces[6][1]] = 1000.0
        final_model.linear.weight[indeces[7][0], indeces[7][1]] = 1000.0
        final_model.linear.weight[indeces[8][0], indeces[8][1]] = 1000.0
        final_model.linear.weight[indeces[9][0], indeces[9][1]] = 1000.0
        print("After Changing: ", final_model.linear.weight)
        for item in test_loaders:
            correct = 0
            for data, target in tqdm(item):
                data = data.to(device)
                target = target.to(device)

                target_pred = final_model(data)
                target_prediction = target_pred.argmax(dim=1, keepdim=True)
                correct += target_prediction.eq(target.view_as(target_prediction)).sum().item()

            acc.append(100. * correct / len(item.sampler))

        #print("Result For Linear Weight Index [{0}, {1}] with the "
        #      "Original Weight value = {2:.5f} "
        #      "and New Weight Value = {3:.5f}\n".format(idx[0], idx[1], linear_weight[idx[0]][idx[1]],
        #                                                final_model.linear.weight[idx[0], idx[1]]))
        #print("When Printing: ", final_model.linear.weight)
        for i in range(len(class_name)):
            print("Accuracy For Class [{0}]: {1:.3f}".format(class_name[i], acc[i]))
        '''
'''
def show_plot(train_loss, test_loss, train_acc, test_acc, save_path):
    plt.figure(figsize=(10, 6), dpi=100)
    plt.title('Resnet20 Train and Test Loss', color='darkblue')
    plt.plot(train_loss, color='blue', label='Train Loss')
    plt.plot(test_loss, color='orange', label='Test Loss')
    plt.legend()
    plt.savefig(save_path + '/Loss.jpg', dpi=200)
    # plt.show()

    plt.figure(figsize=(10, 6), dpi=100)
    plt.title('Resnet20 Train and Test Accuracy', color='darkblue')
    plt.plot(train_acc, color='blue', label='Train Acc')
    plt.plot(test_acc, color='orange', label='Test Acc')
    plt.legend()
    plt.savefig(save_path + '/Acc.jpg', dpi=200)
    # plt.show()

    return
'''

def find_max_weight_and_location(model):
    max_weight = -10000
    max_weight_location = None

    # Iterate through all model parameters
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            current_max_weight, current_max_location = module.weight.data.max(), module.weight.data.argmax()

            if current_max_weight > max_weight:
                max_weight = current_max_weight.item()
                max_weight_location = (name, current_max_location)

    return max_weight, max_weight_location


# Function to find the top 100 max weights, layer names, and indices in convolutional layers
def find_top_conv_weights_and_layers(layer_gradients, num_top_weights=100):
    all_weights = []
    all_layers = []
    data_tuples = []
    for name, grads_matrix in layer_gradients.items():
        current_weights = grads_matrix.flatten()
        current_layer_name = name
        data_tuples = data_tuples + list(zip(current_weights.tolist(), [current_layer_name] * len(current_weights)))

    data_tuples.sort(key=lambda x: x[0], reverse=True)
    top_weights = [item[0] for item in data_tuples[:num_top_weights]]
    top_layers = [item[1] for item in data_tuples[:num_top_weights]]
    all_weights.extend(top_weights)
    all_layers.extend(top_layers)

    return all_weights, all_layers


def find_weight_indeces(model, layer_gradients, max_weight, layer_names):
    result = []
    # Print the results
    for weight, layer in zip(max_weight, layer_names):
        temp = []
        gradient_matrix = layer_gradients[layer]
        location = torch.nonzero(gradient_matrix == weight, as_tuple=False)
        model_layer = get_attribute(model, layer)
        temp.append(layer) #0
        temp.append(model_layer) #1
        temp.append(weight) #2
        temp.append(location) #3
        result.append(temp) #4

    return result


def get_attribute(model, attr_path):
    """
    Recursively get an attribute from a model.

    :param model: The PyTorch model or any Python object.
    :param attr_path: Dot-separated string to the desired attribute.
    :return: The attribute at the specified path.
    """
    attrs = attr_path.split('.')
    for attr in attrs:
        model = getattr(model, attr, None)
        if model is None:
            raise AttributeError(f"Attribute {'.'.join(attrs)} not found")
    return model


def layer_finder(model, layer_name):
    index = int(layer_name[-1])
    if layer_name == 'conv1':
        layer = getattr(model, 'conv1', None)
        return layer
    else:
        layer = getattr(model, layer_name[:-2], None)
        return layer[index]


def model_with_new_weight(model, layer_name, weight_index, random_value):
    if layer_name == 'conv1':
        model.conv1.weight[weight_index[0]][weight_index[1]][weight_index[2]][weight_index[3]] = random_value
    elif layer_name == 'layer1.0.conv1':
        model.layer1[0].conv1.weight[weight_index[0]][weight_index[1]][weight_index[2]][weight_index[3]] = random_value
    elif layer_name == 'layer1.0.conv2':
        model.layer1[0].conv2.weight[weight_index[0]][weight_index[1]][weight_index[2]][weight_index[3]] = random_value
    elif layer_name == 'layer1.1.conv1':
        model.layer1[1].conv1.weight[weight_index[0]][weight_index[1]][weight_index[2]][weight_index[3]] = random_value
    elif layer_name == 'layer1.1.conv2':
        model.layer1[1].conv2.weight[weight_index[0]][weight_index[1]][weight_index[2]][weight_index[3]] = random_value
    elif layer_name == 'layer1.2.conv1':
        model.layer1[2].conv1.weight[weight_index[0]][weight_index[1]][weight_index[2]][weight_index[3]] = random_value
    elif layer_name == 'layer1.2.conv2':
        model.layer1[2].conv2.weight[weight_index[0]][weight_index[1]][weight_index[2]][weight_index[3]] = random_value
    elif layer_name == 'layer2.0.conv1':
        model.layer2[0].conv1.weight[weight_index[0]][weight_index[1]][weight_index[2]][weight_index[3]] = random_value
    elif layer_name == 'layer2.0.conv2':
        model.layer2[0].conv2.weight[weight_index[0]][weight_index[1]][weight_index[2]][weight_index[3]] = random_value
    # elif layer_name == 'layer2.0.shortcut.0':
    #    model.layer2[0].shortcut[0].weight[weight_index[0]][weight_index[1]][weight_index[2]][weight_index[3]] = random_value
    elif layer_name == 'layer2.1.conv1':
        model.layer2[1].conv1.weight[weight_index[0]][weight_index[1]][weight_index[2]][weight_index[3]] = random_value
    elif layer_name == 'layer2.1.conv2':
        model.layer2[1].conv2.weight[weight_index[0]][weight_index[1]][weight_index[2]][weight_index[3]] = random_value
    elif layer_name == 'layer2.2.conv1':
        model.layer2[2].conv1.weight[weight_index[0]][weight_index[1]][weight_index[2]][weight_index[3]] = random_value
    elif layer_name == 'layer2.2.conv2':
        model.layer2[2].conv2.weight[weight_index[0]][weight_index[1]][weight_index[2]][weight_index[3]] = random_value
    elif layer_name == 'layer3.0.conv1':
        model.layer3[0].conv1.weight[weight_index[0]][weight_index[1]][weight_index[2]][weight_index[3]] = random_value
    elif layer_name == 'layer3.0.conv2':
        model.layer3[0].conv2.weight[weight_index[0]][weight_index[1]][weight_index[2]][weight_index[3]] = random_value
    # elif layer_name == 'layer3.0.shortcut.0':
    #    model.layer3[0].shortcut[0].weight[weight_index[0]][weight_index[1]][weight_index[2]][weight_index[3]] = random_value
    elif layer_name == 'layer3.1.conv1':
        model.layer3[1].conv1.weight[weight_index[0]][weight_index[1]][weight_index[2]][weight_index[3]] = random_value
    elif layer_name == 'layer3.1.conv2':
        model.layer3[1].conv2.weight[weight_index[0]][weight_index[1]][weight_index[2]][weight_index[3]] = random_value
    elif layer_name == 'layer3.2.conv1':
        model.layer3[2].conv1.weight[weight_index[0]][weight_index[1]][weight_index[2]][weight_index[3]] = random_value
    elif layer_name == 'layer3.2.conv2':
        model.layer3[2].conv2.weight[weight_index[0]][weight_index[1]][weight_index[2]][weight_index[3]] = random_value
    if layer_name == 'linear':
        model.linear.weight[weight_index[0]][weight_index[1]] = random_value

    return model


def evaluate_weights(model_path, num_of_class, device, data_loader, criterion, num_of_weights,
                     num_of_rand_num, name, title, mode=1):
    model = resnet20.ResNet20(num_classes=num_of_class).to(device)
    model.load_state_dict(torch.load(model_path + "Resnet20_FP_CIFAR10_BestModel.pt", map_location=torch.device('cpu')))
    conv_layers = []
    conv_layers.append(model.conv1)
    conv_layers.append(model.Layer[0])
    conv_layers.append(model.Layer[2])
    conv_layers.append(model.Block1_Layer1[0])
    conv_layers.append(model.Block1_Layer1[2])
    conv_layers.append(model.Block1_Layer2[0])
    conv_layers.append(model.Block1_Layer2[2])
    conv_layers.append(model.Block1_Layer3[0])
    conv_layers.append(model.Block1_Layer3[2])
    conv_layers.append(model.Block2_Layer1[0])
    conv_layers.append(model.Block2_Layer1[2])
    conv_layers.append(model.Block2_Layer2[0])
    conv_layers.append(model.Block2_Layer2[2])
    conv_layers.append(model.Block2_Layer3[0])
    conv_layers.append(model.Block2_Layer3[2])
    conv_layers.append(model.Block3_Layer1[0])
    conv_layers.append(model.Block3_Layer1[2])
    conv_layers.append(model.Block3_Layer2[0])
    conv_layers.append(model.Block3_Layer2[2])
    conv_layers.append(model.Block3_Layer3[0])
    conv_layers.append(model.Block3_Layer3[2])

    if mode == 1:
        target_layer = model.Block1_Layer1[0]
        for idx, (images, labels) in enumerate(data_loader):
            images.requires_grad = True
            target_layer.weight.requires_grad = True
            outputs = model(images)
            loss = -criterion(outputs, labels)
            model.zero_grad()
            loss.backward()
            gradients = torch.abs(target_layer.weight.grad)
            # gradients = target_layer.weight.grad
            break

        # Find Top 100 weights
        top_100_index_list = []
        gradients_copy = gradients.copy_(gradients).to(device)
        for i in range(num_of_weights):
            max_weight_index = (gradients_copy == torch.max(gradients_copy)).nonzero(as_tuple=False)[0]
            top_100_index_list.append(max_weight_index)
            gradients_copy[max_weight_index[0]][max_weight_index[1]][max_weight_index[2]][max_weight_index[3]] = -100

        # Generate 13 Random Number For each weight including inverse and big numbers (10 and -10)
        model = resnet20.ResNet20(num_classes=num_of_class).to(device)
        model.load_state_dict(torch.load(model_path + "Resnet20_FP_CIFAR10_BestModel.pt", map_location=torch.device('cpu')))
        random_weight = torch.zeros((num_of_weights, num_of_rand_num))
        for idx in range(num_of_weights):
            weight_index = top_100_index_list[idx]
            read_w = model.Block1_Layer1[0].weight[weight_index[0]][weight_index[1]][weight_index[2]][weight_index[3]]
            for i in range(num_of_rand_num - 3):
                random_weight[idx][i] = (read_w * torch.rand(1) + (read_w - read_w / 2.))
            random_weight[idx][num_of_rand_num - 5] = -1 * read_w
            random_weight[idx][num_of_rand_num - 4] = 10.0
            random_weight[idx][num_of_rand_num - 3] = 5.0
            random_weight[idx][num_of_rand_num - 2] = -5.0
            random_weight[idx][num_of_rand_num - 1] = -10.0

        # Evaluate The model
        acc = np.zeros((num_of_weights, num_of_rand_num))
        for idx, (data, target) in enumerate(data_loader):
            data = data.to(device)
            target = target.to(device)
            break

        with torch.no_grad():
            for i in range(num_of_weights):
                for j in range(num_of_rand_num):
                    model = resnet20.ResNet20(num_classes=num_of_class).to(device)
                    model.load_state_dict(
                        torch.load(model_path + "Resnet20_FP_CIFAR10_BestModel.pt", map_location=torch.device('cpu')))
                    model.eval()

                    index = top_100_index_list[i]
                    model.Block1_Layer1[0].weight[index[0]][index[1]][index[2]][index[3]] = random_weight[i][j]
                    correct = 0
                    target_pred = model(data)
                    target_prediction = target_pred.argmax(dim=1, keepdim=True)
                    if target_prediction == 1:
                        acc[i][j] = 100
                    else:
                        acc[i][j] = 0
                    print("The accuracy for weight number {0} with Random "
                          "Value = {1:.5f} is ---- {2:.3f}".format(i, random_weight[i][j], acc[i][j]))
                print("===========================================================================================")
        torch.save(top_100_index_list, "Top_Weight_Index_1Pic_1Layer.pt")
        torch.save(random_weight, "Random_Weight_1Pic_1Layer.pt")
        torch.save(acc, 'Acc_1Pic_1Layer.pt')
        print("===========================================================================================")
        print("===========================================================================================")
    elif mode == 2:
        gradients = []
        for idx, (images, labels) in enumerate(data_loader):
            outputs = model(images)
            loss = -criterion(outputs, labels)
            model.zero_grad()
            loss.backward()
            for layer in conv_layers:
                gradients.append(torch.abs(layer.weight.grad))
            break

        layer_list = []
        top_100_index_list = []
        gradients_copy = gradients.copy()
        for i in range(num_of_weights):
            max_weight_index = []
            max_value = []
            for j in range(len(gradients_copy)):
                max_weight_index.append((gradients_copy[j] == torch.max(gradients_copy[j])).nonzero(as_tuple=False)[0])
                max_value.append(
                    gradients_copy[j][max_weight_index[0][0]][max_weight_index[0][1]][max_weight_index[0][2]][
                        max_weight_index[0][3]])

            max_index = (torch.tensor(max_value) == torch.tensor(np.max(max_value))).nonzero(as_tuple=False)[0]
            layer_list.append(max_index)
            top_100_index_list.append(max_weight_index[max_index])
            gradients_copy[max_index][max_weight_index[max_index][0]][max_weight_index[max_index][1]][
                max_weight_index[max_index][2]][max_weight_index[max_index][3]] = -100

        # Generate 13 Random Number For each weight including inverse and big numbers (10 and -10)
        model = resnet20.ResNet20(num_classes=num_of_class).to(device)
        model.load_state_dict(torch.load(model_path + "Resnet20_FP_CIFAR10_BestModel.pt", map_location=torch.device('cpu')))
        random_weight = torch.zeros((num_of_weights, num_of_rand_num))
        for idx in range(num_of_weights):
            weight_index = top_100_index_list[idx]
            read_w = layer_finder(model, layer_list[idx], weight_index)
            for i in range(num_of_rand_num - 3):
                random_weight[idx][i] = (read_w * torch.rand(1) + (read_w - read_w / 2.))
            random_weight[idx][num_of_rand_num - 5] = -1 * read_w
            random_weight[idx][num_of_rand_num - 4] = 10.0
            random_weight[idx][num_of_rand_num - 3] = 5.0
            random_weight[idx][num_of_rand_num - 2] = -5.0
            random_weight[idx][num_of_rand_num - 1] = -10.0

        # Evaluate The model
        acc = np.zeros((num_of_weights, num_of_rand_num))
        for idx, (data, target) in enumerate(data_loader):
            data = data.to(device)
            target = target.to(device)
            break

        with torch.no_grad():
            for i in range(num_of_weights):
                for j in range(num_of_rand_num):
                    model = resnet20.ResNet20(num_classes=num_of_class).to(device)
                    model.load_state_dict(
                        torch.load(model_path + "Resnet20_FP_CIFAR10_BestModel.pt", map_location=torch.device('cpu')))
                    model.eval()

                    index = top_100_index_list[i]
                    model = model_with_new_weight(model, layer_list[i], index, random_weight[i][j])
                    correct = 0
                    target_pred = model(data)
                    target_prediction = target_pred.argmax(dim=1, keepdim=True)
                    if target_prediction == 1:
                        acc[i][j] = 100
                    else:
                        acc[i][j] = 0
                    print("The accuracy for weight number {0} with Random "
                          "Value = {1:.5f} is ---- {2:.3f}".format(i, random_weight[i][j], acc[i][j]))
                print("===========================================================================================")

        torch.save(layer_list, "Layer_List_1Pic_AllLayer.pt")
        torch.save(top_100_index_list, "Top_Weight_Index_1Pic_AllLayer.pt")
        torch.save(random_weight, "Random_Weight_1Pic_AllLayer.pt")
        torch.save(acc, 'Acc_1Pic_AllLayer.pt')
        print("===========================================================================================")
        print("===========================================================================================")
    elif mode == 3:
        target_layer = model.Block1_Layer1[0]
        top_weights = torch.zeros(target_layer.weight.shape).to(device)
        for images, labels in tqdm(data_loader):
            images.requires_grad = True
            target_layer.weight.requires_grad = True
            outputs = model(images)
            loss = -criterion(outputs, labels)
            model.zero_grad()
            loss.backward()
            gradients = torch.abs(target_layer.weight.grad)
            # gradients = target_layer.weight.grad
            # Find the max gradient
            max_grad_index = (gradients == torch.max(gradients)).nonzero(as_tuple=False)[0]
            top_weights[max_grad_index[0]][max_grad_index[1]][max_grad_index[2]][max_grad_index[3]] += 1

        torch.save(top_weights, 'top_weights.pt')
        #top_weights = torch.load("top_weights.pt", map_location=torch.device('cpu'))
        # Select Top 100 weights
        top_100_index_list = []
        top_weights_copy = top_weights.copy_(top_weights).to(device)
        for i in range(num_of_weights):
            max_weight_index = (top_weights_copy == torch.max(top_weights_copy)).nonzero(as_tuple=False)[0]
            top_100_index_list.append(max_weight_index)
            top_weights_copy[max_weight_index[0]][max_weight_index[1]][max_weight_index[2]][max_weight_index[3]] = 0

        # Generate 10 Random Number For each weight including inverse and big numbers (10 and -10)
        model = resnet20.ResNet20(num_classes=num_of_class).to(device)
        model.load_state_dict(torch.load(model_path + "Resnet20_FP_CIFAR10_BestModel.pt", map_location=torch.device('cpu')))
        random_weight = torch.zeros((num_of_weights, num_of_rand_num))
        for idx in range(num_of_weights):
            weight_index = top_100_index_list[idx]
            read_w = model.Block1_Layer1[0].weight[weight_index[0]][weight_index[1]][weight_index[2]][weight_index[3]]
            for i in range(num_of_rand_num - 3):
                random_weight[idx][i] = (read_w * torch.rand(1) + (read_w - read_w / 2.))
            random_weight[idx][num_of_rand_num - 5] = -1 * read_w
            random_weight[idx][num_of_rand_num - 4] = 10.0
            random_weight[idx][num_of_rand_num - 3] = 5.0
            random_weight[idx][num_of_rand_num - 2] = -5.0
            random_weight[idx][num_of_rand_num - 1] = -10.0

        # Check weights
        acc = np.zeros((num_of_weights, num_of_rand_num))
        with torch.no_grad():
            for i in range(num_of_weights):
                for j in range(num_of_rand_num):
                    model = resnet20.ResNet20(num_classes=num_of_class).to(device)
                    model.load_state_dict(
                        torch.load(model_path + "Resnet20_FP_CIFAR10_BestModel.pt", map_location=torch.device('cpu')))
                    model.eval()
                    index = top_100_index_list[i]
                    model.Block1_Layer1[0].weight[index[0]][index[1]][index[2]][index[3]] = random_weight[i][j]
                    correct = 0
                    for idx, (data, target) in enumerate(data_loader):
                        data = data.to(device)
                        target = target.to(device)
                        target_pred = model(data)
                        target_prediction = target_pred.argmax(dim=1, keepdim=True)
                        correct += target_prediction.eq(target.view_as(target_prediction)).sum().item()

                    acc[i][j] = (100. * correct / len(data_loader.sampler))
                    print("The accuracy for weight number {0} with Random "
                          "Value = {1:.5f} is ---- {2:.3f}".format(i, random_weight[i][j], acc[i][j]))
                print("===========================================================================================")

        torch.save(top_100_index_list, "Top_Weight_Index_BatchPic_1Layer.pt")
        torch.save(random_weight, "Random_Weight_BatchPic_1Layer.pt")
        torch.save(acc, 'Acc_BatchPic_1Layer.pt')
        print("===========================================================================================")
        print("===========================================================================================")
    else:
        print()


def top_k_max_with_indices(lst, k):
    # Create a min heap with the first k elements
    heap = [(value, index) for index, value in enumerate(lst[:k])]
    heapq.heapify(heap)

    # Iterate over the remaining elements
    for i, value in enumerate(lst[k:], start=k):
        # Push the current element into the heap
        heapq.heappushpop(heap, (value, i))

    # Extract the top k elements from the heap and reverse the order
    top_k = sorted(heap, reverse=True)

    return top_k
