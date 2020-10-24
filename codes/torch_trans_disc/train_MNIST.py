import numpy as np
import pandas as pd
import argparse
import math
import os
# import matplotlib.pyplot as plt
# from transport import *
# from sklearn.datasets import make_classification, make_moons
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.tensorboard import SummaryWriter
import time 
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from model_MNIST_conv import *
from data_loaders import *
from regularized_ot import *
from torchvision import models as tv_models

device = "cuda:0"

EPOCH = 5
CLASSIFIER_EPOCHS = 25
SUBEPOCH = 10
BATCH_SIZE = 256
DISC_BATCH_SIZE=64
SHUFFLE_BUFFER_SIZE=4096
IS_WASSERSTEIN = True
NUM_TRAIN_DOMAIN = 6
BIN_WIDTH = 15

def train_transformer_batch(X,Y,transported_X,source_u,dest_u,transformer,discriminator,classifier,transformer_optimizer,is_wasserstein=False,encoder=None):
    # print(X.size())
    if encoder is not None:
        with torch.no_grad():
            X = encoder(X).view(-1,16,28,28)
            # X_now = encoder(X_now).view(-1,16,28,28)
    transformer_optimizer.zero_grad()
    # print(X.size())
    X_pred = transformer(X,torch.cat([source_u,dest_u],dim=1))
    # domain_info = X[:,-1].view(-1,1)
    # X_pred_domain_info = torch.cat([X_pred, domain_info], dim=1)

    is_real = discriminator(X_pred,dest_u)

    # X_pred_class = torch.cat([X_pred,domain_info],dim=1)
    pred_class = classifier(X_pred, dest_u)
    # print(torch.max(X_pred))
    # assert False
    trans_loss,ld,lr, lc = discounted_transformer_loss(transported_X, X_pred,is_real, pred_class,Y,is_wasserstein)

    # gradients_of_transformer = trans_tape.gradient(trans_loss, transformer.trainable_variables)
    trans_loss.backward()

    transformer_optimizer.step()

    return trans_loss, ld, lr, lc


def train_discriminator_batch_wasserstein(X_old,source_u,dest_u, X_now, transformer, discriminator, discriminator_optimizer,encoder=None):
    
    if encoder is not None:
        with torch.no_grad():
            X_old = encoder(X_old).view(-1,16,28,28)
            X_now = encoder(X_now).view(-1,16,28,28)
    # print(X_old.size(),X_now.size())
    discriminator_optimizer.zero_grad()
    X_pred = transformer(X_old,torch.cat([source_u.to(device),dest_u.to(device)],dim=1))
    
    # X_pred_old_domain_info = torch.cat([X_pred_old, domain_info], dim=1)

    is_real_old = discriminator(X_pred,dest_u)
    is_real_now = discriminator(X_now,source_u)
    
    disc_loss = discriminator_loss_wasserstein(is_real_now, is_real_old)

    disc_loss.backward()

    discriminator_optimizer.step()
    for p in discriminator.parameters():
        p.data.clamp_(-0.01, 0.01)
    return disc_loss

def train_discriminator_batch(X_old, X_now, transformer, discriminator, discriminator_optimizer,encoder=None):

    if encoder is not None:
        with torch.no_grad():
            X_old = encoder(X_old).view(-1,16,28,28)
            X_now = encoder(X_now).view(-1,16,28,28)
    discriminator_optimizer.zero_grad()
    X_pred_old = transformer(X_old)
    domain_info = X_old[:,-1].view(-1,1)
    X_pred_old_domain_info = torch.cat([X_pred_old, domain_info], dim=1)

    is_real_old = discriminator(X_pred_old_domain_info)
    is_real_now = discriminator(X_now[:,0:-1])
    
    disc_loss = discriminator_loss(is_real_now, is_real_old)

    disc_loss.backward()
    discriminator_optimizer.step()


    return disc_loss


def train_classifier(X, source_u,dest_u, Y, classifier, transformer, classifier_optimizer,encoder=None):

    classifier_optimizer.zero_grad()
    if encoder is not None:
        with torch.no_grad():
            X = encoder(X).view(-1,16,28,28)
    # print(X.size(),source_u.size(),dest_u.size())
    X_pred = transformer(X,torch.cat([source_u.to(device),dest_u.to(device)],dim=1))
    # X_pred = transformer(X,U)
    # domain_info = U[:,-2:].view(-1,2)
    # X_pred_domain_info = torch.cat([X_pred, domain_info], dim=1)
    Y_pred = classifier(X_pred, dest_u)
    
    pred_loss = classification_loss(Y_pred, Y).mean()

    pred_loss.backward()
    classifier_optimizer.step()
    

    return pred_loss


def train_classifier_d(X, U, Y, classifier, classifier_optimizer,verbose=False,encoder=None):

    classifier_optimizer.zero_grad()
    if encoder is not None:
        with torch.no_grad():
            X = encoder(X).view(-1,16,28,28)
    Y_pred = classifier(X, U)
    pred_loss = classification_loss(Y_pred, Y).mean()
    # pred_loss = pred_loss.sum()
    pred_loss.backward()
    
    if verbose:
        # print(torch.cat([Y_pred, Y, Y*torch.log(Y_pred),
        # (Y*torch.log(Y_pred)).sum().unsqueeze(0).unsqueeze(0).repeat(BATCH_SIZE,1) ],dim=1))
        # for p in classifier.parameters():
        #     print(p.data)
        #     print(p.grad.data)
        #     print("____")
        # print(X.max())
        print(torch.cat([Y_pred.argmax(dim=1).view(-1,1).float(),Y.view(-1,1).float()],dim=1))
    classifier_optimizer.step()

    return pred_loss

def show_images(images,fname,num=1):
    '''Plots images given in an array
    
    
    
    Arguments:
        images {list of numpy arrays/tensors} -- List of images to be plotted
        fname {str} -- File name to store in 
    
    Keyword Arguments:
        num {number} -- Number of images to be shown
    '''
    cols = len(images)
    image = np.zeros((num*28,cols*28))
    indices = np.arange(images[0].shape[0])
    np.random.shuffle(indices)
    # print(indices)
    for i in range(num):
        for col in range(cols):
            index = indices[i]
            try:
                real_img  = images[col][index].detach().cpu().numpy().reshape((28,28))
            except:
                real_img  = images[col][index].reshape((28,28))

            image[i*28:(i+1)*28,28*col:28*(col+1)] = real_img/real_img.max()
    Image.fromarray(image*255).convert("L").save(fname)

def train(num_indices, source_indices, target_indices,use_vgg=True):

    I_d = np.eye(num_indices)


    if use_vgg:
        # print(tv_models.vgg16(pretrained=True).features)
        encoder = tv_models.vgg16(pretrained=True).features[:16].to(device)
        encoder.eval()
    else:
        encoder = None
    transformer = Transformer(28**2 + 2*2, 256, use_vgg=use_vgg).to(device)
    discriminator = Discriminator(28**2 + 2, 256,IS_WASSERSTEIN, use_vgg=use_vgg).to(device)
    classifier = ClassifyNet(28**2 + 2,256,10, use_vgg=use_vgg).to(device)

    transformer_optimizer   = torch.optim.Adagrad(transformer.parameters(),5e-2)
    classifier_optimizer    = torch.optim.Adagrad(classifier.parameters(),5e-3)
    discriminator_optimizer = torch.optim.Adagrad(discriminator.parameters(),1e-3)

    U_source = np.array(source_indices)
    U_target = np.array(target_indices)
    writer = SummaryWriter(comment='{}'.format(time.time()))

    ot_maps = [[None for x in range(len(source_indices))] for y in range(len(source_indices))]
    mnist_ind = (np.arange(NUM_TRAIN_DOMAIN*10000))
    np.random.shuffle(mnist_ind)
    # mnist_ind = mnist_ind[:6000] #np.tile(mnist_ind[:1000],NUM_TRAIN_DOMAIN)   
    mnist_ind = np.tile(mnist_ind[:1000],NUM_TRAIN_DOMAIN)   
    ot_data = [torch.utils.data.DataLoader(RotMNIST(indices=mnist_ind[(x-1)*1000:x*1000],bin_width=BIN_WIDTH,bin_index=x-1,n_bins=NUM_TRAIN_DOMAIN,vgg=use_vgg),1000,False) for x in source_indices]#len(source_indices))
    mnist_data = RotMNIST(indices=mnist_ind,bin_width=BIN_WIDTH,bin_index=0,n_bins=6,vgg=use_vgg)
    for i in range(len(source_indices)):
        for j in range(i,len(source_indices)):
            if i!=j:
                ot_sinkhorn = RegularizedSinkhornTransportOTDA(reg_e=0.5, alpha=10, max_iter=50, norm="max", verbose=False)
                if use_vgg:
                    Xs = encoder(next(iter(ot_data[i]))[0]).view(1000,-1).detach().cpu().numpy()+1e-6
                    ys = next(iter(ot_data[i]))[2].view(1000,-1).detach().cpu().numpy()
                    Xt = encoder(next(iter(ot_data[j]))[0]).view(1000,-1).detach().cpu().numpy()+1e-6
                    yt = next(iter(ot_data[j]))[2].view(1000,-1).detach().cpu().numpy()
                    out_shape = (1000,16,28,28)
                else:
                    Xs = next(iter(ot_data[i]))[0].view(1000,-1).detach().cpu().numpy()+1e-6
                    ys = next(iter(ot_data[i]))[2].view(1000,-1).detach().cpu().numpy()
                    Xt = next(iter(ot_data[j]))[0].view(1000,-1).detach().cpu().numpy()+1e-6
                    yt = next(iter(ot_data[j]))[2].view(1000,-1).detach().cpu().numpy() 
                    out_shape = (1000,28,28)

                ot_sinkhorn.fit(Xs=Xs, ys=ys, Xt=Xt, yt=yt, iteration=0)
                ot_maps[i][j] = ot_sinkhorn.transform(Xs).reshape(out_shape)
              # show_images([next(iter(ot_data[i]))[0],next(iter(ot_data[j]))[0],ot_maps[i][j]],num=4,fname='{}-{}.png'.format(i,j))
            else:
                if use_vgg:
                    out_shape = (1000,16,28,28)
                    Xs = encoder(next(iter(ot_data[i]))[0])#.view(out_shape).detach().cpu().numpy()
                    # print(Xs.size())
                    # assert False
                else:
                    out_shape = (1000,28,28)
                    Xs = next(iter(ot_data[i]))[0].view(out_shape).detach().cpu().numpy()
                ot_maps[i][j] = Xs
    
    # print(ot_maps)
    print("-------------TRAINING CLASSIFIER----------")
    class_step = 0

    for epoch in range(CLASSIFIER_EPOCHS):
        past_dataset = torch.utils.data.DataLoader((mnist_data),BATCH_SIZE,True)
        class_loss = 0
        for batch_X, batch_U, batch_Y in tqdm(past_dataset):

            # batch_X = torch.cat([batch_X,batch_U.view(-1,2)],dim=1)

            l = train_classifier_d(batch_X,batch_U,batch_Y,classifier,classifier_optimizer,verbose=False,encoder=encoder)
            class_step += 1
            class_loss += l
        print("Epoch %d Loss %f"%(epoch,class_loss),flush=False)
    # classifier.load_state_dict(torch.load("classifier.pth"))
    all_steps_t = 0
    all_steps_d = 0
    for index in range(1, len(source_indices)):

        print('Domain %d' %index)
        print('----------------------------------------------------------------------------------------------')

        past_data = (RotMNIST(indices=mnist_ind[:source_indices[index-1]*(6000//NUM_TRAIN_DOMAIN)],bin_width=BIN_WIDTH,bin_index=source_indices[0]-1,n_bins=6,vgg=use_vgg)) #,BATCH_SIZE,True)
        # present_dataset = torch.utils.data.Dataloader(torch.utils.data.TensorDataset(X_source[index], U_source[index], 
        #                   Y_source[index]),BATCH_SIZE,True,repeat(
        #                   math.ceil(X_past.shape[0]/X_source[index].shape[0])))

        num_past_batches = len(past_data) // BATCH_SIZE
        # X_past = np.vstack([X_past, X_source[index]])
        past_data = torch.utils.data.DataLoader(past_data,BATCH_SIZE,True)
        # Y_past = np.vstack([Y_past, Y_source[index]])
        # U_past = np.hstack([U_past, U_source[index]])
        
        p = RotMNIST(indices=mnist_ind[:source_indices[(index-1)]*(6000//NUM_TRAIN_DOMAIN)],bin_width=BIN_WIDTH,bin_index=source_indices[0]-1,n_bins=6,transported_samples=ot_maps,target_bin=source_indices[index]-1,vgg=use_vgg,verbose=True)

        # print(len(p), p.vgg)  # TODO
        all_data = torch.utils.data.DataLoader(p,
                    BATCH_SIZE,True)            # for batch_X, batch_U, batch_Y, batch_transported in all_dataset:
        curr_data = torch.utils.data.DataLoader(RotMNIST(indices=mnist_ind[source_indices[index-1]*(6000//NUM_TRAIN_DOMAIN):source_indices[(index)]*(6000//NUM_TRAIN_DOMAIN)],bin_width=BIN_WIDTH,bin_index=source_indices[index-1]-1,n_bins=6,vgg=use_vgg),BATCH_SIZE,True,drop_last=True)
        num_all_batches  = len(p) // BATCH_SIZE
        step_c = 0

        for epoch in range(int(EPOCH*(1+index/5))):

            loss1, loss2 = 0,0
            step_t,step_d = 0,0

            all_dataset = iter(all_data)
            past_dataset = iter(past_data)
            curr_dataset = iter(curr_data)
            loop1 = True
            loop2 = False
            while (loop1 or loop2):
                if step_d < num_past_batches:
                  batch_X, batch_U, batch_Y = next(past_dataset)
                  batch_U = batch_U.view(-1,2)
                  this_U = np.array([U_source[index]*BIN_WIDTH]*batch_U.shape[0]).reshape((batch_U.shape[0],1)) +\
                           np.random.randint(BIN_WIDTH,size=(batch_U.shape[0],1))
                  this_U = np.hstack([np.array([U_source[index]]*batch_U.shape[0]).reshape((batch_U.shape[0],1)),
                                      this_U])
                  this_U = torch.tensor(this_U).float().view(-1,2).to(device)
                  # Do this in a better way


                  # Better to shift this to the dataloader
                  try:
                      real_X,real_U,_ = next(curr_dataset)
                  except:
                      curr_dataset = iter(curr_data)
                      real_X,real_U,_ = next(curr_dataset)
                  if IS_WASSERSTEIN:
                      loss_d = train_discriminator_batch_wasserstein(batch_X, batch_U,this_U,real_X, transformer, discriminator, discriminator_optimizer,encoder=encoder) #train_discriminator_batch(batch_X, real_X)
                  else:
                      loss_d = train_discriminator_batch(batch_X, batch_U,this_U,real_X, transformer, discriminator, discriminator_optimizer,encoder=encoder)
                  loss2 += loss_d
                  writer.add_scalar('Loss/disc',loss_d.detach().cpu().numpy(),step_d+all_steps_d)
                  step_d += 1
                  loop2 = True
                else:
                  loop2 = False
                if step_t < num_all_batches:
                    batch_X, batch_U, batch_Y, batch_transported = next(all_dataset)
                    batch_U = batch_U.view(-1,2)
                    this_U = np.array([U_source[index]*BIN_WIDTH]*batch_U.shape[0]).reshape((batch_U.shape[0],1)) +\
                             np.random.randint(0,5,size=(batch_U.shape[0],1))
                    this_U = np.hstack([np.array([U_source[index]]*batch_U.shape[0]).reshape((batch_U.shape[0],1)),
                                        this_U/(BIN_WIDTH * 6)])
                    this_U = torch.tensor(this_U).float().view(-1,2).to(device)
                    # print(batch_X.size(),batch_U.size(),this_U.size(),batch_transported.size())
                    loss_t,ltd,lr,lc = train_transformer_batch(batch_X,batch_Y,batch_transported,batch_U,this_U,
                                    transformer,discriminator,classifier,
                                    transformer_optimizer,is_wasserstein=IS_WASSERSTEIN,encoder=encoder) #train_transformer_batch(batch_X)
                    loss1 += loss_t
                    writer.add_scalar('Loss/transformer',loss_t.detach().cpu().numpy(),step_t+all_steps_t)
                    writer.add_scalar('Loss/transformer_disc',ltd.detach().cpu().numpy(),step_t+all_steps_t)
                    writer.add_scalar('Loss/transformer_rec',lr.detach().cpu().numpy(),step_t+all_steps_t)
                    writer.add_scalar('Loss/transformer_classifier',lc.detach().cpu().numpy(),step_t+all_steps_t)
                    step_t += 1
                    loop1 = True
                # for batch_X, batch_U, batch_Y in past_dataset:
                else:
                    loop1 = False

                print('Epoch %d - %f, %f' % (epoch, loss1.detach().cpu().numpy()/step_t, loss2),flush=True,end='\r')

            all_steps_d += step_d
            all_steps_t += step_t
            print(' ')
        # show_images([batch_transported,batch_X,transformer(batch_X,torch.cat([batch_U,this_U],dim=1))],'Domain-{}.png'.format(index),10)

    # torch.save(classifier.state_dict(),"classifier.pth")
    print("___________TESTING____________")
    # classifier = ClassifyNet(28**2 + 2,256,10).to(device)

    for i in range(len(target_indices)):
        print(U_target[i])

        source_dataset = torch.utils.data.DataLoader(RotMNIST(indices=mnist_ind[:source_indices[-1]*(6000//NUM_TRAIN_DOMAIN)],bin_width=BIN_WIDTH,bin_index=0,n_bins=6,vgg=use_vgg),BATCH_SIZE,True)



        step = 0
        for epoch in range(CLASSIFIER_EPOCHS//2):

          loss = 0
            
          for batch_X, batch_U, batch_Y in source_dataset:
              batch_U = batch_U.view(-1,2)
              this_U = np.array([U_target[i]*BIN_WIDTH]*batch_U.shape[0]).reshape((batch_U.shape[0],1)) +\
                       np.random.randint(0,5,size=(batch_U.shape[0],1))
              this_U = np.hstack([np.array([U_target[i]]*batch_U.shape[0]).reshape((batch_U.shape[0],1)),
                                  this_U/(BIN_WIDTH * 6)])
              this_U = torch.tensor(this_U).float().view(-1,2).to(device)
              # batch_X = torch.cat([batch_X, batch_U, this_U], dim=1)
              step += 1
              loss += train_classifier(batch_X,batch_U,this_U, batch_Y, classifier,transformer, classifier_optimizer,encoder=encoder)

          print('Epoch: %d - ClassificationLoss: %f' % (epoch, loss))

        
        #print(classifier.trainable_variables)
        index = target_indices[i]
        print(index)
        td = RotMNIST(indices=mnist_ind[(index-1)*(6000//NUM_TRAIN_DOMAIN):(index)*(6000//NUM_TRAIN_DOMAIN)],bin_width=BIN_WIDTH,bin_index=index-1,n_bins=6,vgg=use_vgg)
        print(len(mnist_ind[(index-1)*(6000//NUM_TRAIN_DOMAIN):(index)*(6000//NUM_TRAIN_DOMAIN)]))
        target_dataset = torch.utils.data.DataLoader(td,BATCH_SIZE,False,drop_last=True)
        Y_pred = []
        Y_label = []
        for batch_X, batch_U, batch_Y in tqdm(target_dataset):
            batch_U = batch_U.view(-1,2)
            if encoder is not None:
                batch_X = encoder(batch_X).view(-1,16,28,28)
            batch_Y_pred = classifier(batch_X, batch_U).detach().cpu().numpy()
            Y_pred = Y_pred + [np.argmax(batch_Y_pred,axis=1)]
            Y_label = Y_label + [batch_Y.detach().cpu().numpy()]
        print(len(Y_pred),len(Y_label))
        # print(Y_pred[0].shape,Y_label[0].shape)
        Y_pred = np.hstack(Y_pred)
        Y_label = np.hstack(Y_label)
        print('shape: ',Y_pred.shape)
        print(accuracy_score(Y_label, Y_pred))
        print(confusion_matrix(Y_label, Y_pred))
        print(classification_report(Y_label, Y_pred))    
    return transformer,discriminator,classifier

def train_cross_grad(src_indices,target_indices,num_bins=6,steps=75):
    model_enc = tv_models.vgg16(pretrained=True).features[:5]
    model_enc.eval()
    model_gn  = GradNet(12,6,use_vgg=True)
    model_enc.to(device)
    model_gn.to(device)
    optimizer_enc = torch.optim.Adagrad(model_enc.parameters(),5e-3)
    optimizer_gn = torch.optim.Adagrad(model_gn.parameters(),5e-2)
    mnist_ind = (np.arange(NUM_TRAIN_DOMAIN*10000))
    np.random.shuffle(mnist_ind)
    mnist_ind = mnist_ind[:6000] #np.tile(mnist_ind[:1000],NUM_TRAIN_DOMAIN)   
    for idx in range(1,len(src_indices)):
        source_indices = mnist_ind[:src_indices[idx-1]*(1000)]
        grad_target_indices =  mnist_ind[src_indices[idx-1]*(1000):src_indices[idx]*(1000)]
        ds = RotMNISTCGrad(source_indices,grad_target_indices,BIN_WIDTH,src_indices[0]-1,6,src_indices[idx]-1)
        for epoch in range(EPOCH):
            data = torch.utils.data.DataLoader(ds,BATCH_SIZE,True)
            for img_1,img_2,time_diff in data:
                # optimizer_enc.zero_grad()
                optimizer_gn.zero_grad()
                i1 = model_enc(img_1).view(-1,16,28,28)
                i2 = model_enc(img_2).view(-1,16,28,28)
                time_diff_pred = model_gn(i1,i2)
                # print(time_diff.size(),time_diff_pred.size())
                loss = ((time_diff.view(-1,1) - time_diff_pred.view(-1,1))**2).sum()
                loss.backward()
                # optimizer_enc.step()
                optimizer_gn.step()
                print('Epoch %d - %f' % (epoch, loss.detach().cpu().numpy()),flush=True,end='\r')
    print("")
    past_data = torch.utils.data.DataLoader(RotMNIST(indices=mnist_ind[:source_indices[idx]*(6000//NUM_TRAIN_DOMAIN)],bin_width=BIN_WIDTH,bin_index=src_indices[0]-1,n_bins=6,vgg=True),BATCH_SIZE,False,drop_last=True) 
    time = (60/75)
    new_images = []
    new_labels = []
    for img,u,label in past_data:
        # new_img = torch.zeros_like(img).normal_(0.,1.)
        new_img = img.clone().detach()
        new_img.requires_grad = True
        optim = torch.optim.SGD([new_img],lr=1e-3)
        i1 = model_enc(img).view(-1,16,28,28)
        i2 = model_enc(new_img).view(-1,16,28,28)
        for s in range(steps):
            # optim.zero_grad()
            # print(model(img,new_img).size(),(u[:,1]-time).size())
            loss = ((model_gn(i1,i2) - (u[:,1]-time).view(-1,1))**2).sum()
            # loss.backward()
            grad = torch.autograd.grad(loss,i2)
            print('Step %d - %f' % (s, loss.detach().cpu().numpy()),flush=True,end='\r')
            with torch.no_grad():
                # print(grad)/
                i2 = i2 - 1e-1*grad[0].data
                i2 = i2.detach().clone()
                i2.requires_grad = True
                # new_img.grad.zero_()
            # optim.step()
        new_images.append(new_img.detach().cpu().numpy())
        new_labels.append(label.view(-1,1).detach().cpu().numpy())
    # print([x.size() for x in new_labels])
    new_ds_x, new_ds_y = np.vstack(new_images), np.vstack(new_labels)
    new_ds_u = np.hstack([np.array([time]*len(new_ds_x)).reshape(-1,1),np.array([5/6]*len(new_ds_x)).reshape(-1,1)])
    print(new_ds_x.shape,new_ds_u.shape,new_ds_y.shape)


    # Using augmented data to classify. Can also use augmented data for pre-training only
    classifier = ClassifyNet(3,3,10,use_vgg=True).to(device)

    classifier_optimizer    = torch.optim.Adagrad(classifier.parameters(),5e-3)
    class_data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(new_ds_x).float().to(device),torch.tensor(new_ds_u).float().to(device),
     torch.tensor(new_ds_y).long().to(device)),BATCH_SIZE,False)
    classifier.train()
    mnist_data = RotMNIST(indices=mnist_ind,bin_width=BIN_WIDTH,bin_index=0,n_bins=6,vgg=True)

    for epoch in range(CLASSIFIER_EPOCHS):
        past_dataset = torch.utils.data.DataLoader((mnist_data),BATCH_SIZE,True)
        class_loss = 0
        for batch_X, batch_U, batch_Y in tqdm(past_dataset):

            # batch_X = torch.cat([batch_X,batch_U.view(-1,2)],dim=1)
            batch_X = model_enc(batch_X).view(-1,16,28,28)

            l = train_classifier_d(batch_X,batch_U,batch_Y,classifier,classifier_optimizer,verbose=False)
            # class_step += 1
            class_loss += l
        print("Epoch %d Loss %f"%(epoch,class_loss),flush=False)

    for epoch in range(CLASSIFIER_EPOCHS):
        for X,U,Y in class_data:
            X = model_enc(X).view(-1,16,28,28)
            l = train_classifier_d(X, U, Y, classifier, classifier_optimizer,verbose=False)
            print('Epoch %d - %f' % (epoch, l.detach().cpu().numpy()),flush=True)

    index = target_indices[0]
    td = RotMNIST(indices=mnist_ind[(index-1)*(6000//NUM_TRAIN_DOMAIN):(index)*(6000//NUM_TRAIN_DOMAIN)],bin_width=BIN_WIDTH,bin_index=index-1,n_bins=6,vgg=True)
    print(len(mnist_ind[(index-1)*(6000//NUM_TRAIN_DOMAIN):(index)*(6000//NUM_TRAIN_DOMAIN)]))
    target_dataset = torch.utils.data.DataLoader(td,BATCH_SIZE,False,drop_last=True)
    Y_pred = []
    Y_label = []
    for batch_X, batch_U, batch_Y in tqdm(target_dataset):
        batch_U = batch_U.view(-1,2)
        # if encoder is not None:
        batch_X = model_enc(batch_X).view(-1,16,28,28)
        batch_Y_pred = classifier(batch_X, batch_U).detach().cpu().numpy()
        Y_pred = Y_pred + [np.argmax(batch_Y_pred,axis=1)]
        Y_label = Y_label + [batch_Y.detach().cpu().numpy()]
    print(len(Y_pred),len(Y_label))
    # print(Y_pred[0].shape,Y_label[0].shape)
    Y_pred = np.hstack(Y_pred)
    Y_label = np.hstack(Y_label)
    print('shape: ',Y_pred.shape)
    print(accuracy_score(Y_label, Y_pred))
    print(confusion_matrix(Y_label, Y_pred))
    print(classification_report(Y_label, Y_pred))    


    return classifier
if __name__ == "__main__":
    # X_data, Y_data, U_data = load_moons(11)
    parser = argparse.ArgumentParser()
    """ Arguments: arg """
    parser.add_argument('--algo')
    
    args = parser.parse_args()
    
    
    # train_baselines(X_data, Y_data, U_data, 11, [7,8], [9,10])
    if args.algo == "cg":
        train_cross_grad([1,2,3,4],[5],6)
    else:
        train(6,[1,2,3,4],[5])
    # torch.save({"trans":t.state_dict(),"disc":d.state_dict(),"classifier":c.state_dict()},"./model.pth")