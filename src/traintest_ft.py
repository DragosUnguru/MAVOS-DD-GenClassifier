import sys
import os
import datetime
import time
from utilities import *
import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


def forward_loss_mask(model, masking_output):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        ## Gaussian loss: converges the output towards 0s and 1s
        loss_gauss = model.lambda_gauss * (torch.exp(model.alpha*(masking_output - 0.5)**2) / model.beta).mean()
        ## KL Divergence loss ensure the fixed masking ratio
        loss_kl = model.lambda_kl * (model.kl_divergence(masking_output.mean(-1), 0.25) + model.kl_divergence((1-masking_output).mean(-1), 0.75))
        ## Diversity loss to prevent converging module to generate fixed set of mask for different samples
        loss_diversity = model.lambda_diversity * model.diversity_loss(masking_output)

        return loss_gauss, loss_kl, loss_diversity


def train_adversarial(model, train_loader, test_loader, args):
    """
    Single-phase adversarial training with two optimization steps per iteration:
    
    Step 1 (Discriminator update): 
        - Freeze masking_net, unfreeze gen_classifier
        - Train gen_classifier to correctly classify generative methods

    Step 2 (Generator/Main task update):
        - Freeze gen_classifier, unfreeze masking_net + backbone
        - Train masking_net to FOOL gen_classifier (adversarial)
        - Train backbone for main real/fake classification
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(True)
    
    batch_time, per_sample_time, data_time, per_sample_data_time, per_sample_dnn_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    loss_meter = AverageMeter()
    gen_loss_meter = AverageMeter()
    adv_loss_meter = AverageMeter()
    
    best_epoch, best_mAP, best_acc = 0, -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.save_dir
    
    if not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    
    model.to(device)
    
    print("="*60)
    print("Adversarial Training: Two-step optimization per iteration")
    print("="*60)
    print(f"  - Lambda adversarial: {args.lambda_adv}")
    print(f"  - Gen loss weight: {args.gen_loss_weight}")
    print(f"  - Mask ratio: {args.mask_ratio}")
    print("="*60)
    
    # Setup parameter groups for different components
    # 1. Gen classifier parameters
    gen_classifier_params = list(model.module.gen_classifier.parameters())
    
    # 2. Masking net parameters  
    masking_net_params = list(model.module.masking_net.parameters())
    
    # 3. Backbone + main classifier parameters (everything else)
    gen_classifier_names = set([f'gen_classifier.{n}' for n, _ in model.module.gen_classifier.named_parameters()])
    masking_net_names = set([f'masking_net.{n}' for n, _ in model.module.masking_net.named_parameters()])
    
    backbone_params = []
    mlp_params = []
    
    mlp_list = [
        'a2v.mlp.linear.weight', 'a2v.mlp.linear.bias',
        'v2a.mlp.linear.weight', 'v2a.mlp.linear.bias',
        'mlp_vision.weight', 'mlp_vision.bias',
        'mlp_audio.weight', 'mlp_audio.bias',
        'mlp_head.fc1.weight', 'mlp_head.fc1.bias',
        'mlp_head.fc2.weight', 'mlp_head.fc2.bias',
        'mlp_head.fc3.weight', 'mlp_head.fc3.bias',
    ]
    
    for name, param in model.module.named_parameters():
        if name in gen_classifier_names or name in masking_net_names:
            continue
        if name in mlp_list:
            mlp_params.append(param)
        else:
            backbone_params.append(param)
    
    # Optimizer for gen_classifier (discriminator)
    optimizer_D = torch.optim.Adam(
        gen_classifier_params,
        lr=args.lr * args.head_lr,
        weight_decay=5e-7,
        betas=(0.95, 0.999)
    )
    
    # Optimizer for masking_net + backbone + main classifier (generator + main task)
    optimizer_G = torch.optim.Adam([
        {'params': masking_net_params, 'lr': args.lr},
        {'params': backbone_params, 'lr': args.lr},
        {'params': mlp_params, 'lr': args.lr * args.head_lr}
    ], weight_decay=5e-7, betas=(0.95, 0.999))
    
    trainables = [p for p in model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    print('Gen classifier parameter number is : {:.3f} million'.format(sum(p.numel() for p in gen_classifier_params) / 1e6))
    print('Masking net parameter number is : {:.3f} million'.format(sum(p.numel() for p in masking_net_params) / 1e6))
    
    scheduler_D = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_D, 
        list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),
        gamma=args.lrscheduler_decay
    )
    scheduler_G = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_G, 
        list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),
        gamma=args.lrscheduler_decay
    )
    
    main_metrics = args.metrics
    
    # Loss functions
    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    args.loss_fn = loss_fn
    
    # Loss for generative method classification (multi-label BCE)
    gen_loss_fn = nn.BCEWithLogitsLoss()
    
    epoch += 1
    scaler = GradScaler()
    
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start adversarial training...")
    result = np.zeros([args.n_epochs, 6])  # acc, mAP, AUC, lr, gen_loss, adv_loss
    model.train()
    
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))

        for i, (a_input, v_input, labels, gen_labels, _) in enumerate(tqdm(train_loader)):
            assert a_input.shape[0] == v_input.shape[0]
            B = a_input.shape[0]
            a_input = a_input.to(device, non_blocking=True)
            v_input = v_input.to(device, non_blocking=True)
            labels = labels.to(device)
            gen_labels = gen_labels.to(device)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / B)
            dnn_start_time = time.time()

            # STEP 1: Train Discriminator (gen_classifier)
            # Goal: Correctly classify which generative method was used

            model.module.freeze_maskingnet()
            model.module.unfreeze_gen_classifier()
            
            optimizer_D.zero_grad()
            
            with autocast():
                # Forward pass -- we need features but don't want gradients through masking_net
                with torch.no_grad():
                    # Get masked features without gradient
                    output, gen_output_detached, video_mask, _ = model(
                        a_input, v_input, 
                        apply_mask=True, 
                        hard_mask=True, 
                        hard_mask_ratio=args.mask_ratio,
                        adversarial=False  # Don't use GRL here
                    )

                # Re-compute gen_output with gradients for gen_classifier only
                # We need to get the fused features and pass through gen_classifier
                # This requires a slight modification -- for now, do a second forward
                _, gen_output, _, _ = model(
                    a_input, v_input, 
                    apply_mask=True, 
                    hard_mask=True, 
                    hard_mask_ratio=args.mask_ratio,
                    adversarial=True  # Get gen_output
                )
                
                # Discriminator loss: classify generative methods correctly
                loss_D = gen_loss_fn(gen_output, gen_labels)
            
            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)
            scaler.update()
            
            gen_loss_meter.update(loss_D.item(), B)
            
            # STEP 2: Train Generator (masking_net) + Main Task (backbone)
            # Goals: 
            #   - Masking_net: Fool the gen_classifier (adversarial)
            #   - Backbone: Correctly classify real/fake
            
            model.module.unfreeze_maskingnet()
            model.module.freeze_gen_classifier()
            
            optimizer_G.zero_grad()
            
            with autocast():
                # Forward pass with gradient through masking_net
                output, gen_output, video_mask, _ = model(
                    a_input, v_input, 
                    apply_mask=True, 
                    hard_mask=True, 
                    hard_mask_ratio=args.mask_ratio,
                    adversarial=True
                )
                
                # Main classification loss (real/fake)
                main_loss = loss_fn(output, labels)
                
                # Adversarial loss: FOOL the gen_classifier
                # We want gen_classifier to be WRONG, so we maximize its loss
                adv_loss = -gen_loss_fn(gen_output, gen_labels)
                
                # Mask regularization loss (optional)
                mask_reg_loss = 0.0
                if video_mask is not None and hasattr(args, 'mask_loss_lambda') and args.mask_loss_lambda > 0:
                    mask_reg_loss = args.mask_loss_lambda * sum(forward_loss_mask(model.module, video_mask))
                
                # Total generator loss
                loss_G = main_loss + args.lambda_adv * adv_loss + mask_reg_loss

            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)
            scaler.update()
            
            loss_meter.update(main_loss.item(), B)
            adv_loss_meter.update(adv_loss.item(), B)
            
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/a_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/a_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Time {per_sample_time.avg:.5f}\t'
                  'Main Loss {loss_meter.val:.4f}\t'
                  'Gen Loss (D) {gen_loss_meter.val:.4f}\t'
                  'Adv Loss (G) {adv_loss_meter.val:.4f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time,
                   loss_meter=loss_meter, gen_loss_meter=gen_loss_meter, 
                   adv_loss_meter=adv_loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1
        
        print('start validation')

        # Unfreeze everything for validation
        model.module.unfreeze_maskingnet()
        model.module.unfreeze_gen_classifier()
        
        stats, valid_loss = validate(model, test_loader, args)

        mAP = stats['AP_macro']
        mAUC = stats['AUC_macro']
        acc = stats['accuracy']

        if main_metrics == 'mAP':
            print("mAP: {:.6f}".format(mAP))
        else:
            print("acc: {:.6f}".format(acc))
        print("AUC: {:.6f}".format(mAUC))
        print("d_prime: {:.6f}".format(d_prime(mAUC)))
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("gen_loss (D): {:.6f}".format(gen_loss_meter.avg))
        print("adv_loss (G): {:.6f}".format(adv_loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))

        result[epoch-1, :] = [acc, mAP, mAUC, optimizer_G.param_groups[0]['lr'], gen_loss_meter.avg, adv_loss_meter.avg]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        print('validation finished')
        
        if mAP > best_mAP:
            best_mAP = mAP
            if main_metrics == 'mAP':
                best_epoch = epoch

        if acc > best_acc:
            best_acc = acc
            if main_metrics == 'acc':
                best_epoch = epoch

        if best_epoch == epoch:
            torch.save(model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            torch.save(optimizer_G.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))
        if args.save_model == True:
            torch.save(model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
        
        scheduler_D.step()
        scheduler_G.step()
            
        print('Epoch-{0} lr_D: {1} lr_G: {2}'.format(epoch, optimizer_D.param_groups[0]['lr'], optimizer_G.param_groups[0]['lr']))
        
        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        per_sample_dnn_time.reset()
        loss_meter.reset()
        gen_loss_meter.reset()
        adv_loss_meter.reset()


def train(model, train_loader, test_loader, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(True)
    
    batch_time, per_sample_time, data_time, per_sample_data_time, per_sample_dnn_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    loss_meter = AverageMeter()
    
    best_epoch, best_mAP, best_acc = 0, -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.save_dir
    
    if not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    
    model.to(device)

    if args.train_mask:
        print("Training MaskingNet. I.e. freezing AVFF and using soft mask")

        model.module.freeze_backbone()
        hard_mask = False
    else:
        print("Training AVFF. I.e. freezing MaskingNet and using hard mask")

        model.module.freeze_maskingnet()
        hard_mask = True
    
    # possible mlp layer name list, mlp layers are newly initialized layers in the finetuning stage (i.e., not pretrained) and should use a larger lr during finetuning
    mlp_list = [
        'a2v.mlp.linear.weight',
        'a2v.mlp.linear.bias',
        'v2a.mlp.linear.weight',
        'v2a.mlp.linear.bias',
        'mlp_vision.weight',
        'mlp_vision.bias',
        'mlp_audio.weight',
        'mlp_audio.bias',
        'mlp_head.fc1.weight',
        'mlp_head.fc1.bias',
        'mlp_head.fc2.weight',
        'mlp_head.fc2.bias'
    ]
    mlp_params = list(filter(lambda kv: kv[0] in mlp_list, model.module.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in mlp_list, model.module.named_parameters()))
    mlp_params = [i[1] for i in mlp_params]
    base_params = [i[1] for i in base_params]
    
    trainables = [p for p in model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    optimizer = torch.optim.Adam([{'params': base_params, 'lr': args.lr}, {'params': mlp_params, 'lr': args.lr * args.head_lr}], weight_decay=5e-7, betas=(0.95, 0.999))
    base_lr = optimizer.param_groups[0]['lr']
    mlp_lr = optimizer.param_groups[1]['lr']
    lr_list = [args.lr, mlp_lr]
    print('base lr, mlp lr : ', base_lr, mlp_lr)
    
    print('Total newly initialized MLP parameter number is : {:.3f} million'.format(sum(p.numel() for p in mlp_params) / 1e6))
    print('Total pretrained backbone parameter number is : {:.3f} million'.format(sum(p.numel() for p in base_params) / 1e6))
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)
    main_metrics = args.metrics
    
    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    args.loss_fn = loss_fn
    
    epoch += 1
    scaler = GradScaler()
    
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 4])  # for each epoch, 10 metrics to record
    model.train()
    
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))

        start_time = time.time()
        for i, (a_input, v_input, labels, _) in enumerate(tqdm(train_loader)):
            # print(f"step 1: ", time.time() - start_time)
            # start_time = time.time()
            assert a_input.shape[0] == v_input.shape[0]
            B = a_input.shape[0]
            a_input = a_input.to(device, non_blocking=True)
            v_input = v_input.to(device, non_blocking=True)
            labels = labels.to(device)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / B)
            dnn_start_time = time.time()
            
            # print(f"step 2: ", time.time() - start_time)
            # start_time = time.time()
            with autocast():
                output, _, video_mask, _ = model(a_input, v_input, apply_mask=True, hard_mask=hard_mask, hard_mask_ratio=args.mask_ratio, adversarial=False)
                loss = loss_fn(output, labels) + args.mask_loss_lambda * sum(forward_loss_mask(model.module, video_mask))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # print(f"step 5: ", time.time() - start_time)
            # start_time = time.time()
            
            # loss_av is the main loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/a_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/a_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Loss {loss_meter.val:.4f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1
            
            # print(f"step 6: ", time.time() - start_time)
            # start_time = time.time()
        
        print('start validation')
        stats, valid_loss = validate(model, test_loader, args)

        # mAP = np.mean([stat['AP'] for stat in stats])
        # mAUC = np.mean([stat['auc'] for stat in stats])
        # acc = stats[0]['acc'] # this is just a trick, acc of each class entry is the same, which is the accuracy of all classes, not class-wise accuracy
        mAP = stats['AP_macro']
        mAUC = stats['AUC_macro']
        acc = stats['accuracy']

        if main_metrics == 'mAP':
            print("mAP: {:.6f}".format(mAP))
        else:
            print("acc: {:.6f}".format(acc))
        print("AUC: {:.6f}".format(mAUC))
        print("d_prime: {:.6f}".format(d_prime(mAUC)))
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))

        result[epoch-1, :] = [acc, mAP, mAUC, optimizer.param_groups[0]['lr']]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        print('validation finished')
        
        if mAP > best_mAP:
            best_mAP = mAP
            if main_metrics == 'mAP':
                best_epoch = epoch

        if acc > best_acc:
            best_acc = acc
            if main_metrics == 'acc':
                best_epoch = epoch

        if best_epoch == epoch:
            torch.save(model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))
        if args.save_model == True:
            torch.save(model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if main_metrics == 'mAP':
                scheduler.step(mAP)
            elif main_metrics == 'acc':
                scheduler.step(acc)
        else:
            scheduler.step()
            
        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        
        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        per_sample_dnn_time.reset()
        loss_meter.reset()
        
def validate(model, val_loader, args, output_pred=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()

    end = time.time()
    A_predictions, A_targets, A_loss = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            # Handle both old format (4 values) and new format (5 values with gen_labels)
            if len(batch) == 5:
                a_input, v_input, labels, gen_labels, _ = batch
            else:
                a_input, v_input, labels, _ = batch
            
            a_input = a_input.to(device)
            v_input = v_input.to(device)
            labels = labels.to(device)

            with autocast():
                audio_output, _, _, _ = model(a_input, v_input, apply_mask=False, adversarial=False)

            predictions = audio_output.to('cpu').detach()

            A_predictions.append(predictions)
            A_targets.append(labels)

            labels = labels.to(device)
            loss = args.loss_fn(audio_output, labels)
            A_loss.append(loss.to('cpu').detach())

            batch_time.update(time.time() - end)
            end = time.time()

        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)

        stats = calculate_stats(audio_output.cpu(), target.cpu())

    if output_pred == False:
        return stats, loss
    else:
        # used for multi-frame evaluation (i.e., ensemble over frames), so return prediction and target
        return stats, audio_output, target