from torch.multiprocessing import Process, Event, Queue
import numpy as np
import pandas as pd
from time import sleep
from client import Client
import torch
from torch.nn import BCELoss
from torch.optim import SGD
from collections import OrderedDict
import copy
import random
import os
from multiprocessing import Pool

def set_random_seed(seed:int=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ## NOTE: If you want every run to be exactly the same each time
    ##       uncomment the following lines
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Server():
    def __init__(self, n_clients:int, n_rounds:int, n_workers:int, Model, Criterion, Optimizer, dfs_train, df_test, continuous_columns, discrete_columns, target_column, n_no_aggregation_rounds, learning_rate, local_epochs, batch_size, verbose, eval_every, devices, seed, criterion_params=None, optimizer_params=None):
        self.n_workers = n_workers
        self.devices = devices
        self.seed = seed
        
        self.n_rounds = n_rounds
        self.n_clients = n_clients
        self.n_clients_per_worker = self.n_clients // self.n_workers
        self.eval_every = eval_every
        self.n_no_aggregation_rounds = n_no_aggregation_rounds
        
        self.dfs_train = dfs_train
        self.df_test = df_test
        self.continuous_columns = continuous_columns,
        self.discrete_columns = discrete_columns
        self.target_column = target_column
        
        self.Model = Model
        self.Criterion = Criterion
        self.criterion_params = criterion_params
        self.Optimizer = Optimizer
        self.optimizer_params = optimizer_params
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        model = self.Model()
        self.state_dict = model.state_dict()
        
        self.verbose = verbose

    def _averge_state_dicts(self, state_dicts:list, weights:list=None):
        if weights == None:
            weights = [1.0 for _ in range(0, len(state_dicts))]
        avg_state_dict = OrderedDict([(key, 0.0) for key in state_dicts[0].keys()])
        for key in state_dicts[0].keys():
            for i in range(0, len(state_dicts)):
                avg_state_dict[key] += torch.mul(weights[i], torch.div(state_dicts[i][key].cpu(), len(state_dicts)))
        return avg_state_dict

    def _train_clients(self, queue, main_event, worker_event, device):
        # init model
        model = self.Model()
        model.to(device)
        
        for round in range(self.n_rounds):        
            # train clients
            clients = queue.get()
            for client in clients:
                set_random_seed(seed=self.seed+round)
                if round == 0:
                    test_loss, test_accuracy = client.test(model=model, device=device)
                    if self.verbose:
                        print('%s: test_loss = %.4f, test_accuracy = %.4f' % (client.name, test_loss, test_accuracy))
                elif round % self.eval_every == 0:
                    train_loss, train_accuracy = client.train(model=model, device=device)
                    val_loss, val_accuracy = client.validate(model=model, device=device)
                    test_loss, test_accuracy = client.test(model=model, device=device)
                    if self.verbose:
                        print('%s: train_loss = %.4f, train_accuracy = %.4f, val_loss = %.4f, val_accuracy = %.4f, test_loss = %.4f, test_accuracy = %.4f' % (client.name, train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy))
                else:
                    train_loss, train_accuracy = client.train(model=model, device=device)
                    #print('%s: train_loss = %.4f, train_accuracy = %.4f' % (client.name, train_loss, train_accuracy))
            queue.put(clients)
            worker_event.set()
            
            # wait for server
            main_event.wait()
            main_event.clear()

    def run(self):
        main_event = Event()
        #worker_events = [Event() for _ in range(self.n_workers)]
        #queues = [Queue() for _ in range(self.n_workers)]
        #workers = [Process(target=self._train_clients, args=(queue, main_event, worker_event, device)) for worker_event, queue, device in zip(worker_events, queues)]
        worker_events = []
        queues = []
        workers = []
        for i in range(self.n_workers):
            worker_events.append(Event())
            queues.append(Queue())
            device = self.devices[i % len(self.devices)]
            workers.append(Process(target=self._train_clients, args=(queues[i], main_event, worker_events[i], device)))

        if self.verbose:
            print('round %i' % 0)
        else:
            #print(' round %i' % 0, end='\r')
            print(' progress: [%s%s] ' % ('#' * (0 * 100 // self.n_rounds), '.' * (100 - 0 * 100 // self.n_rounds - 1)), end='\r')

        
        # server working
        clients = [
            Client(
                name = 'client_%i' % i,
                state_dict = copy.deepcopy(self.state_dict), 
                df_train = df_train, 
                df_test = self.df_test,
                Criterion = self.Criterion,
                Optimizer = self.Optimizer, 
                learning_rate = self.learning_rate,
                continuous_columns = self.continuous_columns,
                discrete_columns = self.discrete_columns, 
                target_column = self.target_column,
                local_epochs = self.local_epochs,
                batch_size = self.batch_size,
                criterion_params = self.criterion_params,
                optimizer_params = self.optimizer_params,
                device=self.devices[i % len(self.devices)],
            ) 
            for i, df_train in enumerate(self.dfs_train)
        ]
        clientss = [clients[i:i+self.n_clients_per_worker] for i in range(0, self.n_clients, self.n_clients_per_worker)]
        for queue, clients in zip(queues, clientss):
            queue.put(clients)
        
        # Clients working
        for worker in workers:
            worker.start()
        for worker_event in worker_events:
            worker_event.wait()
            worker_event.clear()
        
        if self.verbose:
            print()

        for round in range(1, self.n_rounds):
            
            if self.verbose:
                if round % self.eval_every == 0:
                    print('round %i ' % round)
                else:
                    print(' round %i' % round, end='\r')
            else:
                #print(' round %i' % round, end='\r')
                print(' progress: [%s%s] ' % ('#' * (round * 100 // self.n_rounds), '.' * (100 - round * 100 // self.n_rounds - 1)), end='\r')
                
            # server working
            if round < self.n_no_aggregation_rounds:
                clients = [client for queue in queues for client in queue.get()]
                for client in clients:
                    client.state_dict = copy.deepcopy(client.state_dict)
                clientss = [clients[x:x+self.n_clients_per_worker] for x in range(0, self.n_clients, self.n_clients_per_worker)]
                for queue, clients in zip(queues, clientss):
                    queue.put(clients)
            else:
                clients = [client for queue in queues for client in queue.get()]
                state_dicts = [client.state_dict for client in clients]
                trainset_sizes = [client.trainset_size for client in clients]
                weights = [trainset_size / sum(trainset_sizes) for trainset_size in trainset_sizes]
                avg_state_dict = self._averge_state_dicts(state_dicts=state_dicts, weights=weights)
                for client in clients:
                    client.state_dict = copy.deepcopy(avg_state_dict)
                clientss = [clients[x:x+self.n_clients_per_worker] for x in range(0, self.n_clients, self.n_clients_per_worker)]
                for queue, clients in zip(queues, clientss):
                    queue.put(clients)
            main_event.set()

            # clients working
            for worker_event in worker_events:
                worker_event.wait()
                worker_event.clear()
            
            if self.verbose and round % self.eval_every == 0:
                print()
        clients = [client for queue in queues for client in queue.get()]
        #for queue in queues:
        #    queue.close()
        # main_event.set()
        
        for worker in workers:
            #worker.join()
            worker.terminate()
        
        return {client.name: client.log for client in clients}
    
    '''
    def _train_clients2(self, args):
        clients, model, device = args
        model.to(device)
        for client in clients:
            client.train(model, device)
        return clients
    
    def _val_clients2(self, args):
        clients, model, device = args
        model.to(device)
        for client in clients:
            client.validate(model, device)
        return clients
    
    def _test_clients2(self, args):
        clients, model, device = args
        model.to(device)
        for client in clients:
            client.test(model, device)
        return clients
    
    def run2(self):
        clients = [
            Client(
                name = 'client_%i' % i,
                state_dict = copy.deepcopy(self.state_dict), 
                df_train = df_train, 
                df_test = self.df_test,
                Criterion = self.Criterion,
                Optimizer = self.Optimizer, 
                learning_rate = self.learning_rate,
                continuous_columns = self.continuous_columns,
                discrete_columns = self.discrete_columns, 
                target_column = self.target_column,
                local_epochs = self.local_epochs,
                batch_size = self.batch_size,
                criterion_params = self.criterion_params,
                optimizer_params = self.optimizer_params,
                device=self.devices[i % len(self.devices)],
            ) 
            for i, df_train in enumerate(self.dfs_train)
        ]
        cs = np.array_split(clients, self.n_workers)
        ms = [self.Model() for i in range(self.n_workers)]
        ds = [self.devices[i % len(self.devices)] for i in range(self.n_workers)]
        iterable = [(cs[i], ms[i], ds[i]) for i in range(self.n_workers)]
        with Pool(self.n_workers) as p:
            for round in range(self.n_rounds):
                if round == 0:
                    p.map(self._test_clients2, iterable)
                elif round % self.eval_every == 0:
                    p.map(self._train_clients2, iterable)
                    p.map(self._val_clients2, iterable)
                    p.map(self._test_clients2, iterable)
                else:
                    p.map(self._train_clients2, iterable)
                
                if round > self.n_no_aggregation_rounds:
                    state_dicts = [client.state_dict for client in clients]
                    trainset_sizes = [client.trainset_size for client in clients]
                    weights = [trainset_size / sum(trainset_sizes) for trainset_size in trainset_sizes]
                    avg_state_dict = self._averge_state_dicts(state_dicts=state_dicts, weights=weights)
                    for client in clients:
                        client.state_dict = copy.deepcopy(avg_state_dict)
        
        return {client.name: client.log for client in clients}

    '''
