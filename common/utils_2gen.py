# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from common.utils import triplet_loss,adv_loss




def lossfun_one_batch(device, model, model_pos_gen, model_neg_gen, dis_model, opt,
                      fea_opt, opt_pos_gen, opt_neg_gen, opt_dis, params, batch, epoch):
    # the first half of a batch are the anchors and the latters
    # are the positive examples corresponding to each anchor
    lambda_ = 1
    lambda1 = 1.0
    lambda2 = 50.0

    model.train()
    model_pos_gen.train()
    model_neg_gen.train()
    dis_model.train()

    ancs, poss, negs = batch
    ancs = ancs.split(params.batch_size, dim=0)
    poss = poss.split(params.batch_size, dim=0)
    negs = negs.split(params.batch_size, dim=0)

    total_loss_pos_gen = 0
    total_loss_neg_gen = 0
    total_loss_m = 0

    for i in range(len(ancs)):
        anc = ancs[i].to(device)
        pos = poss[i].to(device)
        neg = negs[i].to(device)

        model.zero_grad()
        model_pos_gen.zero_grad()
        model_neg_gen.zero_grad()
        dis_model.zero_grad()

        anc_out = model(anc)  # (N, 512)
        pos_out = model(pos)  # (N, 512)
        neg_out = model(neg)  # (N, 512)

        if epoch < params.neg_gen_epoch:
            # t_loss = torch.mean(F.relu(torch.norm(anc_out - pos_out, dim=1) ** 2
            #                            - torch.norm(anc_out - neg_out) ** 2 + params.alpha))
            t_loss = F.triplet_margin_loss(anc_out, pos_out, neg_out, margin=params.alpha)
            t_loss.backward()
            fea_opt.step()
            continue

        # Train dis_model
        batch_concat = torch.cat((anc_out, pos_out, neg_out), dim=1)
        g_pos = model_pos_gen(batch_concat)
        #print(g_pos.shape) # (N, 512)
        ###
        g_pos = (pos_out + g_pos)/2
        ###
        batch_concat_g = torch.cat((anc_out, g_pos, neg_out), dim=1)
        fake = model_neg_gen(batch_concat_g)  # (N, 512)
        ###
        fake = (neg_out + fake)/2
        ###
        batch_fake = torch.cat((anc_out, pos_out, fake), dim=0)
        #pos_out or g_pos?
        embedding_fake = dis_model(batch_fake)  # (3 * N, 512)
        loss_m = lambda_ * triplet_loss(embedding_fake, margin=params.alpha)
        total_loss_m += loss_m.item() * anc.size(0)

        loss_m.backward()
        #opt.step()
        opt_dis.step()

        model.zero_grad()
        model_pos_gen.zero_grad()
        model_neg_gen.zero_grad()
        dis_model.zero_grad()

        # train gen_model
        anc_detach  = anc_out.detach()
        pos_detach  = pos_out.detach()
        neg_detach = neg_out.detach()

        batch_concat_g = torch.cat((anc_detach, g_pos.detach(), neg_detach), dim=1)
        fake = model_neg_gen(batch_concat_g)  # (N, 512)
        ##########
        #fake = (neg_detach + fake)/2
        ##########
        loss_hard = torch.mean(torch.norm(fake - anc_detach, dim=1))
        loss_reg = torch.mean(torch.norm(fake - neg_detach, dim=1))
        ###
        fake = (neg_detach + fake)/2
        ###
        batch_fake = torch.cat((anc_detach, pos_detach, fake), dim=0)
        #pos_out or g_pos?
        embedding_fake = dis_model(batch_fake)  # (3 * N, 512)


        loss_adv = adv_loss(embedding_fake, margin=params.alpha)
        loss_gen = loss_hard + lambda1 * loss_reg + lambda2 * loss_adv
        total_loss_neg_gen += loss_gen.item() * anc.size(0)

        loss_gen.backward()
        opt_neg_gen.step()
        model.zero_grad()
        model_pos_gen.zero_grad()
        model_neg_gen.zero_grad()
        dis_model.zero_grad()

        ##############################################
        #pos gen
        anc_detach = anc_out.detach()
        neg_detach = neg_out.detach()
        pos_detach = pos_out.detach()

        batch_concat = torch.cat((anc_detach, pos_detach, neg_detach), dim=1)
        g_pos = model_pos_gen(batch_concat)
        ##########
        #g_pos = (pos_detach + g_pos)/2
        ##########
        loss_hard_pos = torch.mean(torch.norm(g_pos - anc_detach, dim=1))
        loss_reg_pos = torch.mean(torch.norm(g_pos - pos_detach, dim=1))
        loss_gen_pos = loss_hard_pos + lambda1 * loss_reg_pos

        total_loss_pos_gen += loss_gen_pos.item() * anc.size(0)

        loss_gen_pos.backward()
        opt_pos_gen.step()

    return total_loss_pos_gen, total_loss_neg_gen, total_loss_m


def lossfun_one_batch_retain(device, model, model_pos_gen, model_neg_gen, dis_model, opt,
                      fea_opt, opt_pos_gen, opt_neg_gen, opt_dis, params, batch, epoch):
    #print('RETAIN!')
    # the first half of a batch are the anchors and the latters
    # are the positive examples corresponding to each anchor
    lambda_ = 1
    lambda1 = 1.0
    lambda2 = 50.0

    model.train()
    model_pos_gen.train()
    model_neg_gen.train()
    dis_model.train()

    ancs, poss, negs = batch
    ancs = ancs.split(params.batch_size, dim=0)
    poss = poss.split(params.batch_size, dim=0)
    negs = negs.split(params.batch_size, dim=0)

    total_loss_pos_gen = 0
    total_loss_neg_gen = 0
    total_loss_m = 0

    for i in range(len(ancs)):
        anc = ancs[i].to(device)
        pos = poss[i].to(device)
        neg = negs[i].to(device)

        model.zero_grad()
        model_pos_gen.zero_grad()
        model_neg_gen.zero_grad()
        dis_model.zero_grad()

        anc_out = model(anc)  # (N, 512)
        pos_out = model(pos)  # (N, 512)
        neg_out = model(neg)  # (N, 512)

        if epoch < params.neg_gen_epoch:
            # t_loss = torch.mean(F.relu(torch.norm(anc_out - pos_out, dim=1) ** 2
            #                            - torch.norm(anc_out - neg_out) ** 2 + params.alpha))
            t_loss = F.triplet_margin_loss(anc_out, pos_out, neg_out, margin=params.alpha)
            t_loss.backward()
            fea_opt.step()
            continue

        # Train dis_model
        batch_concat = torch.cat((anc_out, pos_out, neg_out), dim=1)
        g_pos = model_pos_gen(batch_concat)
        #print(g_pos.shape) # (N, 512)
        ###
        #g_pos = (pos_out + g_pos)/2
        ###
        batch_concat_g = torch.cat((anc_out, g_pos, neg_out), dim=1)
        fake = model_neg_gen(batch_concat_g)  # (N, 512)
        ###
        #fake = (neg_out + fake)/2
        ###
        batch_fake = torch.cat((anc_out, pos_out, fake), dim=0)
        #pos_out or g_pos?
        embedding_fake = dis_model(batch_fake)  # (3 * N, 512)
        loss_m = lambda_ * triplet_loss(embedding_fake, margin=params.alpha)
        total_loss_m += loss_m.item() * anc.size(0)

        loss_m.backward(retain_graph=True)
        #opt.step()
        opt_dis.step()

        loss_hard = torch.mean(torch.norm(fake - anc_out, dim=1))
        loss_reg = torch.mean(torch.norm(fake - neg_out, dim=1))
        ###
        #fake = (neg_detach + fake)/2
        ###
        #pos_out or g_pos?
        loss_adv = adv_loss(embedding_fake, margin=params.alpha)
        loss_gen = loss_hard + lambda1 * loss_reg + lambda2 * loss_adv
        total_loss_neg_gen += loss_gen.item() * anc.size(0)

        loss_gen.backward(retain_graph=True)
        opt_neg_gen.step()

        loss_hard_pos = torch.mean(torch.norm(g_pos - anc, dim=1))
        loss_reg_pos = torch.mean(torch.norm(g_pos - pos, dim=1))
        loss_gen_pos = loss_hard_pos + lambda1 * loss_reg_pos

        total_loss_pos_gen += loss_gen_pos.item() * anc.size(0)

        loss_gen_pos.backward()
        opt_pos_gen.step()

    return total_loss_pos_gen, total_loss_neg_gen, total_loss_m


def lossfun_one_batch_baseline(device, model, dis_model, opt,
                               opt_dis, params, batch):
    # the first half of a batch are the anchors and the latters
    # are the positive examples corresponding to each anchor

    model.train()
    dis_model.train()

    ancs, poss, negs = batch
    ancs = ancs.split(params.batch_size, dim=0)
    poss = poss.split(params.batch_size, dim=0)
    negs = negs.split(params.batch_size, dim=0)

    total_loss_m = 0

    for i in range(len(ancs)):
        anc = ancs[i].to(device)
        pos = poss[i].to(device)
        neg = negs[i].to(device)

        model.zero_grad()
        dis_model.zero_grad()

        anc_out = model(anc)  # (N, 512)
        pos_out = model(pos)  # (N, 512)
        neg_out = model(neg)  # (N, 512)

        # Train dis_model
        batch_concat = torch.cat((anc_out, pos_out, neg_out), dim=0)
        embeddings = dis_model(batch_concat)  # (3 * N, 512)
        loss_m = triplet_loss(embeddings, margin=params.alpha)
        total_loss_m += loss_m.item() * anc.size(0)

        loss_m.backward()
        opt.step()
        opt_dis.step()

    return total_loss_m



