# import new Network name here and add in model_class args
import time

import torch

from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import torchvision

def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager_Loss()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)

    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]

        # data_1, train_label_1 = fusion_aug_image(data, train_label, args.mix_times, args, session=0)
        # data_2, train_label_2 = fusion_aug_image(data, train_label, args.mix_times, args, session=0)
        # data_1, train_label_1 = cutmix_aug_image(data, train_label, args.mix_times, 0, args)
        # data_2, train_label_2 = cutmix_aug_image(data, train_label, args.mix_times, 0, args)
    
        logits_1 = model(data_1)
        logits_2 = model(data_2)
        loss_acc = F.cross_entropy(logits_1, train_label_1) + F.cross_entropy(logits_2, train_label_2)
        # loss_acc = F.cross_entropy(model(data), train_label)

        total_loss = loss_acc

        lrc = scheduler.get_last_lr()[0]
        tl.add(total_loss.item(), len(train_label))
        tqdm_gen.set_description('Session 0, epo {}, lrc={:.4f}, loss_acc={:.4f}'.format(epoch, lrc, loss_acc.item()))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    tl = tl.item()
    ta = ta.item()

    return tl, ta

def inc_train(session, model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager_Loss()
    ta = Averager()
    model = model.train()

    tqdm_gen = tqdm(trainloader)

    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]
        data, train_label = fusion_aug_image(data, train_label, args.mix_times, args, session=0)

        logits = model(data)

        loss = F.cross_entropy(logits, train_label) 

        total_loss = loss

        lrc = scheduler.get_last_lr()[0]
        tl.add(total_loss.item(), len(train_label))
        tqdm_gen.set_description('Session {}, epo {}, lrc={:.4f},total loss={:.4f}'.format(session, epoch, lrc, total_loss.item()))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    tl = tl.item()
    ta = ta.item()

    return tl, ta

def ft(session, model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager_Loss()
    ta = Averager()
    model = model.train()
    tqdm_gen = tqdm(trainloader)

    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]
        
        if session != 0:
            train_label = train_label - (args.base_class + (session-1) * args.way)  
            
        loss_acc = F.cross_entropy(model(data), train_label)
        
        total_loss = loss_acc

        lrc = scheduler.get_last_lr()[0]
        tl.add(total_loss.item(), len(train_label))
        tqdm_gen.set_description('Session {}, epo {}, lrc={:.4f}, loss_acc={:.4f}'.format(session, epoch, lrc, loss_acc.item()))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    tl = tl.item()
    ta = ta.item()

    return tl, ta

def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []

    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.fc.weight.data[:args.base_class] = proto_list

    return model

def test(model, testloader, epoch, args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager_Loss()
    va = Averager()

    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            logits = model(data)
            logits = logits[:, :test_class]
            acc = count_acc(logits, test_label)

            va.add(acc, len(test_label))

        # vl = vl.item()
        va = va.item()
        
    print('epo {}, test, acc={:.4f}'.format(epoch, va))

    logs = dict(num_session=session + 1, acc=va)

    return vl, va, logs

def test_inc(testloader, epoch, args, sessions):
    test_class = args.base_class + sessions * args.way
    models = []
    vl = Averager_Loss()
    va = Averager()
    
    for session in range(sessions + 1):
        model = MYNET(args, mode=args.base_mode).cuda()
        ckpt = torch.load(os.path.join('checkpoint', args.dataset, str(args.epochs_base), 'session' + str(session) + '_best.pth'))
        model.load_state_dict(ckpt['params'])
        models.append(model)

    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            logitss = []
            entropies = []
            for model in models:
                model.eval()
                data, test_label = [_.cuda() for _ in batch]
                logits = model(data)
                logits = logits[:, :test_class]
                entropies.append(get_entropy(logits))
                logitss.append(logits)
            selected_model = entropies.index(min(entropies))
            logits = logitss[selected_model] 
            acc = count_acc(logits, test_label)
            
            va.add(acc, len(test_label))

        va = va.item()

    print('epo {}, test, acc={:.4f}'.format(epoch, va))

    logs = dict(num_session=session + 1, acc=va)

    return vl, va, logs

def test_inc2(testloader, epoch, args, sessions):
    
    sub_results = int(args.base_class / args.way)
    
    def get_test_class(args, session):
        if session == -1:
            return 0
        return args.base_class + session * args.way

    models = []
    vl = Averager_Loss()
    va = Averager()
    
    for session in range(sessions + 1):
        model = MYNET(args, mode=args.base_mode).cuda()
        ckpt = torch.load(os.path.join('checkpoint', args.dataset, str(args.epochs_base), 'session' + str(session) + '_best.pth'))
        model.load_state_dict(ckpt['params'])
        if session == 0:
            model.mode = "output_entropy_base"
        else:
            model.mode = "output_entropy_inc"
        models.append(model)

    with torch.no_grad():
        for j, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]

            true_label = test_label
            logitss = []
            entropies = []
            for i, model in enumerate(models):
                model.eval()
                logits = model(data)
                if i == 0:
                    entropies = [get_entropy(logits[:, int(args.way)*j : int(args.way)*(j+1)]) for j in range(sub_results)]
                else:
                    entropies.append(get_entropy(logits))
                logitss.append(logits)

            selected_model = entropies.index(min(entropies))
            selected_model = 0 if selected_model < sub_results else selected_model - (sub_results - 1)
            logits = logitss[selected_model]
            # print(selected_model)
            
            test_label = test_label - get_test_class(args, selected_model-1)
            acc = count_acc(logits, test_label)
            va.add(acc, len(test_label))    
        va = va.item()

    logs = dict(num_session=session + 1, acc=va)

    return vl, va, logs

def get_entropy(logits):
    b = F.softmax(logits, dim=1)*F.log_softmax(logits, dim=1)
    b = -1.0*b.sum(dim=1) / logits.size(0)
    return b

def cutmix_aug_image(data, label, mix_times, session, args):
    
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    original_data = data.clone()
    batch_size = data.size()[0]
    mix_data = []
    mix_target = []
    
    for _ in range(mix_times):
        index = torch.randperm(batch_size).cuda()
        for i in range(batch_size):
            if label[i] != label[index][i]:
                new_label = fusion_aug_generate_label(label[i].item(), label[index][i].item(), args, session)
                lam = np.random.beta(args.beta, args.beta)
                bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
                original_data[i][:, bbx1:bbx2, bby1:bby2] = original_data[index, :][i][:, bbx1:bbx2, bby1:bby2]
                mix_data.append(original_data[i])
                mix_target.append(new_label)

    new_target = torch.Tensor(mix_target)
    label = torch.cat((label, new_target.cuda().long()), 0)
    for item in mix_data:
        data = torch.cat((data, item.unsqueeze(0)), 0)
    return data, label

def fusion_aug_image(x, y, mix_times, args, session=0, alpha=20.0):  # mixup based
    batch_size = x.size()[0]
    mix_data = []
    mix_target = []

    for _ in range(mix_times):
        index = torch.randperm(batch_size).cuda()
        for i in range(batch_size):
            if y[i] != y[index][i]:
                new_label = fusion_aug_generate_label(y[i].item(), y[index][i].item(), args, session)
                lam = np.random.beta(alpha, alpha)
                if lam < 0.4 or lam > 0.6:
                    lam = 0.5
                mix_data.append(lam * x[i] + (1 - lam) * x[index, :][i])
                mix_target.append(new_label)

    new_target = torch.Tensor(mix_target)
    y = torch.cat((y, new_target.cuda().long()), 0)
    for item in mix_data:
        x = torch.cat((x, item.unsqueeze(0)), 0)

    return x, y

def fusion_aug_generate_label(y_a, y_b, args, session=0):
    current_total_cls_num = args.base_class + session * args.way
    if session == 0:  # base session -> increasing: [(args.base_class) * (args.base_class - 1)]/2
        y_a, y_b = y_a, y_b
        assert y_a != y_b
        if y_a > y_b:  # make label y_a smaller than y_b
            tmp = y_a
            y_a = y_b
            y_b = tmp
        label_index = ((2 * current_total_cls_num - y_a - 1) * y_a) / 2 + (y_b - y_a) - 1
    else:  # incremental session -> increasing: [(args.way) * (args.way - 1)]/2
        y_a = y_a - (current_total_cls_num - args.way)
        y_b = y_b - (current_total_cls_num - args.way)
        assert y_a != y_b
        if y_a > y_b:  # make label y_a smaller than y_b
            tmp = y_a
            y_a = y_b
            y_b = tmp
        label_index = int(((2 * args.way - y_a - 1) * y_a) / 2 + (y_b - y_a) - 1)
    return label_index + current_total_cls_num




