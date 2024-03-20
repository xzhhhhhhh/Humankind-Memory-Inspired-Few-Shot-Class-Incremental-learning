import os.path

import torch
import itertools

from .base import Trainer
import pandas as pd
from .helper import *
from utils import *
from dataloader.data_utils import *
import torch.nn.functional as F
from models.base.loss import cosLoss

class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.set_log_path()
        self.model = MYNET(self.args, mode=self.args.base_mode).cuda()

        self.args = set_up_datasets(self.args)

        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']

        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

    def get_optimizer_base(self):

        optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)

        return optimizer, scheduler
    
    def get_optimizer_base_ft(self):

        optimizer = torch.optim.SGD(self.model.fc_base.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)

        return optimizer, scheduler
    
    def get_optimizer_inc(self):

        optimizer = torch.optim.SGD(itertools.chain(self.model.encoder.layer4.parameters(), self.model.fc.parameters()), self.args.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)

        return optimizer, scheduler
    
    def get_optimizer_inc_ft(self):

        optimizer = torch.optim.SGD(self.model.fc_inc.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)

        return optimizer, scheduler

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader

    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        columns = ['num_session', 'acc', 'base_acc', 'new_acc', 'base_acc_given_new', 'new_acc_given_base']
        acc_df = pd.DataFrame(columns=columns)
        testacc_inc = []

        for session in range(args.start_session, args.sessions):
            
            train_set, trainloader, testloader = self.get_dataloader(session)

            if session == 0:  # load base class train img label
                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()
                
                for epoch in range(args.epochs_base):
                
                    tl, ta = base_train(self.model, trainloader, optimizer, scheduler, epoch, args)
                    tsl, tsa, logs = test(self.model, testloader, epoch, args, session)
                    if tsa > self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = tsa
                        self.trlog['max_acc_epoch'] = epoch
                    print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'], self.trlog['max_acc'][session]))
                    scheduler.step()

                if not args.not_data_init:
                    self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                    best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc_replace_head.pth')
                    print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    self.model.mode = 'avg_cos'
                    tsl, tsa, logs = test(self.model, testloader, 0, args, session)
                    self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                    print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))

                self.save_model(session)
                
                # fine-tune
                ckpt = torch.load(os.path.join(self.args.save_path, str(self.args.epochs_base), 'session0_best.pth'))
                self.model.load_state_dict(ckpt['params'])
                self.model.mode = 'output_entropy_base'
                optimizer_ft, scheduler_ft = self.get_optimizer_base_ft()
                for epoch in range(args.epochs_base_ft):
                    tl, ta = ft(session, self.model, trainloader, optimizer, scheduler, epoch, args)
                    tsl, tsa, logs = test(self.model, testloader, epoch, args, session)
                    print('epoch {}, test acc = {:.3f}'.format(epoch, tsa))
                    if tsa > self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = tsa
                        self.trlog['max_acc_epoch'] = epoch
                    print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'], self.trlog['max_acc'][session]))
                    scheduler_ft.step()

            else:# incremental learning sessions
                ckpt = torch.load(os.path.join(self.args.save_path, str(self.args.epochs_base), 'session0_best.pth'))
                self.model.load_state_dict(ckpt['params'])
                print("training session: [%d]" % session)
                optimizer_inc, scheduler_inc = self.get_optimizer_inc()
                self.model.mode = self.args.new_mode
                trainloader.dataset.transform = testloader.dataset.transform
                for epoch in range(self.args.epochs_new):
                    tl, ta = inc_train(session, self.model, trainloader, optimizer_inc, scheduler_inc, epoch, self.args)
                    scheduler_inc.step()
                
                self.save_model(session)
                tsl, tsa, logs = test_inc2(testloader, epoch, args, session)
                print('epoch {}, test acc = {:.3f}'.format(epoch, tsa))
                
                # fine-tune
                ckpt = torch.load(os.path.join(self.args.save_path, str(self.args.epochs_base), 'session' + str(session) + "_best.pth"))
                self.model.load_state_dict(ckpt['params'])
                self.model.mode = 'output_entropy_inc'
                optimizer, scheduler = self.get_optimizer_inc_ft()
                for epoch in range(self.args.epochs_new_ft):
                    tl, ta = ft(session, self.model, trainloader, optimizer, scheduler, epoch, args)
                    scheduler.step()
                
                self.save_model(session)
                tsl, tsa, logs = test_inc2(testloader, epoch, args, session)
                print('epoch {}, test acc = {:.3f}'.format(epoch, tsa))
                    
                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                print('test acc={:.3f}'.format(self.trlog['max_acc'][session]))
                
            self.save_model(session)

        print(self.trlog['max_acc'])
        print(testacc_inc)
      
        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)

    def set_save_path(self):
        self.args.save_path = os.path.join('checkpoint', self.args.dataset)
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)

    def set_log_path(self):
        if self.args.model_dir is not None:
            self.args.save_log_path = '%s/' % self.args.project
            self.args.save_log_path = self.args.save_log_path + '%s' % self.args.dataset
            if 'avg' in self.args.new_mode:
                self.args.save_log_path = self.args.save_log_path + '_prototype_' + self.args.model_dir.split('/')[-2][:7] + '/'
            if 'ft' in self.args.new_mode:
                self.args.save_log_path = self.args.save_log_path + '_WaRP_' + 'lr_new_%.3f-epochs_new_%d-keep_frac_%.2f/' % (
                    self.args.lr_new, self.args.epochs_new, self.args.fraction_to_keep)
            self.args.save_log_path = os.path.join('acc_logs', self.args.save_log_path)
            ensure_path(self.args.save_log_path)
            self.args.save_log_path = self.args.save_log_path + self.args.model_dir.split('/')[-2] + '.csv'
            
    def save_model(self, session):
        
        if not os.path.exists(os.path.join(self.args.save_path, str(self.args.epochs_base))):
            os.makedirs(os.path.join(self.args.save_path, str(self.args.epochs_base)))

        save_model_dir = os.path.join(self.args.save_path, str(self.args.epochs_base), 'session' + str(session) + '_best.pth')
        
        torch.save(dict(params=self.model.state_dict()), save_model_dir)
        print('Saving model to :%s' % save_model_dir)
