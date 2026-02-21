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
                # Single forward pass with detached features for gen_classifier
                # This means gradients only flow through gen_classifier, not masking_net/backbone
                _, gen_output, video_mask, _ = model(
                    a_input, v_input, 
                    apply_mask=True, 
                    hard_mask=True, 
                    hard_mask_ratio=args.mask_ratio,
                    adversarial=True,
                    detach_features_for_gen=True  # Detach features
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
                # Don't detach features - we want gradients to flow back
                output, gen_output, video_mask, _ = model(
                    a_input, v_input, 
                    apply_mask=True, 
                    hard_mask=False, 
                    hard_mask_ratio=args.mask_ratio,
                    adversarial=True,
                    detach_features_for_gen=False  # Gradients flow through
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


def validate_contrastive(model, val_loader, args, output_pred=False):
    """Validation function for contrastive learning model."""
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
            # Handle both formats
            if len(batch) == 5:
                a_input, v_input, labels, gen_labels, _ = batch
            else:
                a_input, v_input, labels, _ = batch
            
            a_input = a_input.to(device)
            v_input = v_input.to(device)
            labels = labels.to(device)

            with autocast():
                # Contrastive model returns (output, video_mask, projections)
                output, _, _ = model(a_input, v_input, apply_mask=False, return_projections=False)

            predictions = output.to('cpu').detach()

            A_predictions.append(predictions)
            A_targets.append(labels)

            loss = args.loss_fn(output, labels)
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
        return stats, audio_output, target


def train_contrastive(model, train_loader, test_loader, args):
    """
    Adversarial Contrastive Learning training loop with multi-label method support.
    
    Two-pass per iteration:
    
    Step 1 (Discriminator - Method Projector):
        - Freeze masking_net, unfreeze gen_method_projector
        - Train gen_method_projector to cluster samples by generative method
        - Uses multi-label supervised contrastive loss (samples can have 2 methods)
    
    Step 2 (Generator - Masking Net + Main Task):
        - Freeze gen_method_projector, unfreeze masking_net + backbone
        - Train masking_net to FOOL the method projector (adversarial)
        - Train backbone for real/fake detection using:
          * Classification loss (BCE/CE)
          * Supervised contrastive loss (real vs fake clustering)
    
    The goal: masking_net learns to mask method-specific artifacts, forcing the
    backbone to learn GENERAL deepfake features that generalize to unseen methods.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(True)
    
    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    
    # Main task losses
    cls_loss_meter = AverageMeter()
    supcon_loss_meter = AverageMeter()
    total_loss_meter = AverageMeter()
    
    # Adversarial losses
    method_loss_meter = AverageMeter()  # Discriminator loss (method clustering)
    method_acc_meter = AverageMeter()   # Method clustering accuracy
    adv_loss_meter = AverageMeter()     # Adversarial loss (fool method projector)
    
    best_epoch, best_mAP, best_acc = 0, -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.save_dir
    
    if not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    
    model.to(device)
    
    # Get config from args with defaults
    apply_mask = getattr(args, 'apply_mask', True)
    mask_ratio = getattr(args, 'mask_ratio', 0.4)
    lambda_adv = getattr(args, 'lambda_adv', 0.1)
    lambda_supcon = getattr(args, 'lambda_supcon', 1.0)
    
    print("="*60)
    print("Adversarial Contrastive Learning Training (Multi-Label)")
    print("="*60)
    print(f"  - Classification weight: {args.cls_weight}")
    print(f"  - SupCon weight: {lambda_supcon}")
    print(f"  - Adversarial weight: {lambda_adv}")
    print(f"  - Apply masking: {apply_mask}")
    print(f"  - Mask ratio: {mask_ratio}")
    print("="*60)
    
    # Setup parameter groups for different components
    mlp_list = [
        'a2v.mlp.linear.weight', 'a2v.mlp.linear.bias',
        'v2a.mlp.linear.weight', 'v2a.mlp.linear.bias',
        'mlp_vision.weight', 'mlp_vision.bias',
        'mlp_audio.weight', 'mlp_audio.bias',
        'mlp_head.fc1.weight', 'mlp_head.fc1.bias',
        'mlp_head.fc2.weight', 'mlp_head.fc2.bias',
        'mlp_head.fc3.weight', 'mlp_head.fc3.bias',
    ]
    
    # 1. Gen method projector parameters (discriminator)
    gen_method_projector_params = list(model.module.gen_method_projector.parameters())
    
    # 2. Masking net parameters
    masking_net_params = list(model.module.masking_net.parameters())
    
    # 3. Everything else (backbone + fusion_projector + classifier)
    gen_method_projector_names = set([f'gen_method_projector.{n}' for n, _ in model.module.gen_method_projector.named_parameters()])
    masking_net_names = set([f'masking_net.{n}' for n, _ in model.module.masking_net.named_parameters()])
    
    mlp_params = []
    base_params = []
    
    for name, param in model.module.named_parameters():
        if not param.requires_grad:
            continue
        if name in gen_method_projector_names or name in masking_net_names:
            continue
        if name in mlp_list or 'projector' in name:
            mlp_params.append(param)
        else:
            base_params.append(param)
    
    # Optimizer for discriminator (gen_method_projector)
    optimizer_D = torch.optim.Adam(
        gen_method_projector_params,
        lr=args.lr * args.head_lr,
        weight_decay=5e-7,
        betas=(0.95, 0.999)
    )
    
    # Optimizer for generator (masking_net + backbone + fusion_projector + classifier)
    optimizer_G = torch.optim.Adam([
        {'params': masking_net_params, 'lr': args.lr},
        {'params': base_params, 'lr': args.lr},
        {'params': mlp_params, 'lr': args.lr * args.head_lr},
    ], weight_decay=5e-7, betas=(0.95, 0.999))
    
    trainables = [p for p in model.parameters() if p.requires_grad]
    print('Total parameter number: {:.3f} M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print('Total trainable parameter number: {:.3f} M'.format(sum(p.numel() for p in trainables) / 1e6))
    print('Discriminator (gen_method_projector): {:.3f} M'.format(sum(p.numel() for p in gen_method_projector_params) / 1e6))
    print('Masking net: {:.3f} M'.format(sum(p.numel() for p in masking_net_params) / 1e6))
    
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
    
    # Classification loss
    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    args.loss_fn = loss_fn
    
    epoch += 1
    scaler = GradScaler()
    
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("Starting adversarial contrastive training...")
    
    # Results: acc, mAP, AUC, lr, supcon_loss, method_loss, adv_loss
    result = np.zeros([args.n_epochs, 7])  # acc, mAP, mAUC, lr, supcon, method, adv
    model.train()
    
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        model.train()
        
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))

        for i, batch in enumerate(tqdm(train_loader)):
            # Must have gen_labels for adversarial training
            if len(batch) == 5:
                a_input, v_input, labels, gen_labels, _ = batch
            else:
                raise ValueError("Adversarial contrastive training requires gen_labels (5-element batch)")
            
            B = a_input.shape[0]
            a_input = a_input.to(device, non_blocking=True)
            v_input = v_input.to(device, non_blocking=True)
            labels = labels.to(device)
            gen_labels = gen_labels.to(device)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / B)
            dnn_start_time = time.time()

            # ================================================================
            # STEP 1: Train Discriminator (gen_method_projector)
            # Goal: Cluster samples by their generative method (multi-label)
            # ================================================================
            model.module.freeze_maskingnet()
            model.module.unfreeze_gen_method_projector()
            
            optimizer_D.zero_grad()
            
            with autocast():
                # Forward pass - get projections with DETACHED features for gen_method_projector
                # This ensures gradients only flow through gen_method_projector
                output, video_mask, projections = model(
                    a_input, v_input,
                    apply_mask=apply_mask,
                    hard_mask=True,  # Use hard mask for discriminator
                    hard_mask_ratio=mask_ratio,
                    return_projections=True
                )
                
                # Discriminator loss: cluster by generative method (multi-label aware)
                # Use detached features so gradients only flow through gen_method_projector
                detached_projections = {
                    'fused_features': projections['fused_features'].detach()
                }
                loss_D, method_acc = model.module.compute_method_discrimination_loss(
                    detached_projections, gen_labels
                )
            
            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)
            scaler.update()
            
            method_loss_meter.update(loss_D.item(), B)
            method_acc_meter.update(method_acc.item(), B)
            
            # ================================================================
            # STEP 2: Train Generator (masking_net) + Main Task (backbone)
            # Goals:
            #   - Masking_net: Fool the gen_method_projector (adversarial)
            #   - Backbone: Real/fake classification + contrastive clustering
            # ================================================================
            model.module.unfreeze_maskingnet()
            model.module.freeze_gen_method_projector()
            
            optimizer_G.zero_grad()
            
            with autocast():
                # Forward pass - gradients flow through masking_net
                output, video_mask, projections = model(
                    a_input, v_input,
                    apply_mask=apply_mask,
                    hard_mask=False,  # Use soft mask for generator (better gradients)
                    hard_mask_ratio=mask_ratio,
                    return_projections=True
                )
                
                # === Main Task Losses ===
                
                # 1. Classification loss (real/fake)
                cls_loss = loss_fn(output, labels)
                
                # 2. Supervised contrastive loss (real vs fake clustering)
                supcon_loss = model.module.compute_contrastive_loss(projections, labels)
                
                # === Adversarial Loss ===
                
                # Adversarial loss: FOOL the method projector (multi-label aware)
                adv_loss, _ = model.module.compute_adversarial_method_loss(
                    projections, gen_labels
                )
                
                # Total generator loss
                total_loss = args.cls_weight * cls_loss + lambda_supcon * supcon_loss + lambda_adv * adv_loss
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer_G)
            scaler.update()
            
            # Update meters
            cls_loss_meter.update(cls_loss.item(), B)
            supcon_loss_meter.update(supcon_loss.item(), B)
            adv_loss_meter.update(adv_loss.item(), B)
            total_loss_meter.update(total_loss.item(), B)
            
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time) / B)
            per_sample_dnn_time.update((time.time() - dnn_start_time) / B)

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {per_sample_time.avg:.5f}\t'
                      'Cls {cls_loss.val:.4f}\t'
                      'SupCon {supcon_loss.val:.4f}\t'
                      'Method(D) {method_loss.val:.4f}\t'
                      'MethodAcc {method_acc.val:.3f}\t'
                      'Adv(G) {adv_loss.val:.4f}'.format(
                       epoch, i, len(train_loader),
                       per_sample_time=per_sample_time,
                       cls_loss=cls_loss_meter,
                       supcon_loss=supcon_loss_meter,
                       method_loss=method_loss_meter,
                       method_acc=method_acc_meter,
                       adv_loss=adv_loss_meter), flush=True)
                
                if np.isnan(total_loss_meter.avg):
                    print("Training diverged...")
                    return

            end_time = time.time()
            global_step += 1
        
        print('Starting validation...')
        stats, valid_loss = validate_contrastive(model, test_loader, args)

        mAP = stats['AP_macro']
        mAUC = stats['AUC_macro']
        acc = stats['accuracy']

        if main_metrics == 'mAP':
            print("mAP: {:.6f}".format(mAP))
        else:
            print("acc: {:.6f}".format(acc))
        print("AUC: {:.6f}".format(mAUC))
        print("d_prime: {:.6f}".format(d_prime(mAUC)))
        print("train_cls_loss: {:.6f}".format(cls_loss_meter.avg))
        print("train_supcon_loss: {:.6f}".format(supcon_loss_meter.avg))
        print("train_method_loss (D): {:.6f}".format(method_loss_meter.avg))
        print("train_method_acc: {:.6f}".format(method_acc_meter.avg))
        print("train_adv_loss (G): {:.6f}".format(adv_loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))

        result[epoch-1, :] = [acc, mAP, mAUC, optimizer_G.param_groups[0]['lr'],
                              supcon_loss_meter.avg, method_loss_meter.avg, adv_loss_meter.avg]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        print('Validation finished')
        
        if mAP > best_mAP:
            best_mAP = mAP
            if main_metrics == 'mAP':
                best_epoch = epoch

        if acc > best_acc:
            best_acc = acc
            if main_metrics == 'acc':
                best_epoch = epoch

        if best_epoch == epoch:
            torch.save(model.state_dict(), "%s/models/best_model.pth" % (exp_dir))
            torch.save(optimizer_G.state_dict(), "%s/models/best_optim_G_state.pth" % (exp_dir))
            torch.save(optimizer_D.state_dict(), "%s/models/best_optim_D_state.pth" % (exp_dir))
        if args.save_model:
            torch.save(model.state_dict(), "%s/models/model.%d.pth" % (exp_dir, epoch))
        
        scheduler_G.step()
        scheduler_D.step()
            
        print('Epoch-{0} lr_G: {1} lr_D: {2}'.format(
            epoch, optimizer_G.param_groups[0]['lr'], optimizer_D.param_groups[0]['lr']))
        
        finish_time = time.time()
        print('Epoch {:d} training time: {:.3f}'.format(epoch, finish_time - begin_time))

        epoch += 1

        # Reset meters
        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        per_sample_dnn_time.reset()
        cls_loss_meter.reset()
        supcon_loss_meter.reset()
        method_loss_meter.reset()
        method_acc_meter.reset()
        adv_loss_meter.reset()
        total_loss_meter.reset()


def train_contrastive_random_mask(model, train_loader, test_loader, args):
    """
    Contrastive Learning training loop with RANDOM masking (no learned MaskingNet).
    
    Random masking (like dropout over visual tokens)
    
    Losses:
    1. Classification loss (BCE/CE) for real/fake detection
    2. Supervised contrastive loss for clustering in embedding space
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(True)
    
    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    
    cls_loss_meter = AverageMeter()
    supcon_loss_meter = AverageMeter()
    total_loss_meter = AverageMeter()
    
    best_epoch, best_mAP, best_acc = 0, -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.save_dir
    
    if not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    
    model.to(device)
    
    apply_mask = getattr(args, 'apply_mask', True)
    mask_ratio = getattr(args, 'mask_ratio', 0.4)
    lambda_supcon = getattr(args, 'supcon_weight', 1.0)
    
    print("="*60)
    print("Contrastive Learning with Random Masking (Single-Step)")
    print("="*60)
    print(f"  - Classification weight: {args.cls_weight}")
    print(f"  - SupCon weight: {lambda_supcon}")
    print(f"  - Apply masking: {apply_mask}")
    print(f"  - Mask ratio: {mask_ratio}")
    print(f"  - No MaskingNet — random masking (dropout-style)")
    print(f"  - No adversarial game — single optimizer")
    print("="*60)
    
    # Single set of parameter groups — no adversarial split needed
    mlp_list = [
        'a2v.mlp.linear.weight', 'a2v.mlp.linear.bias',
        'v2a.mlp.linear.weight', 'v2a.mlp.linear.bias',
        'mlp_vision.weight', 'mlp_vision.bias',
        'mlp_audio.weight', 'mlp_audio.bias',
        'mlp_head.fc1.weight', 'mlp_head.fc1.bias',
        'mlp_head.fc2.weight', 'mlp_head.fc2.bias',
        'mlp_head.fc3.weight', 'mlp_head.fc3.bias',
    ]
    
    mlp_params = []
    base_params = []
    
    for name, param in model.module.named_parameters():
        if not param.requires_grad:
            continue
        if name in mlp_list or 'projector' in name:
            mlp_params.append(param)
        else:
            base_params.append(param)
    
    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': args.lr},
        {'params': mlp_params, 'lr': args.lr * args.head_lr},
    ], weight_decay=5e-7, betas=(0.95, 0.999))
    
    trainables = [p for p in model.parameters() if p.requires_grad]
    print('Total parameter number: {:.3f} M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print('Total trainable parameter number: {:.3f} M'.format(sum(p.numel() for p in trainables) / 1e6))
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),
        gamma=args.lrscheduler_decay
    )
    
    main_metrics = args.metrics
    
    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    args.loss_fn = loss_fn
    
    epoch += 1
    scaler = GradScaler()
    
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("Starting contrastive training with random masking...")
    
    result = np.zeros([args.n_epochs, 5])  # acc, mAP, mAUC, lr, supcon_loss
    model.train()
    
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        model.train()
        
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))

        for i, batch in enumerate(tqdm(train_loader)):
            # Support both 4-element and 5-element batches
            if len(batch) == 5:
                a_input, v_input, labels, gen_labels, _ = batch
            else:
                a_input, v_input, labels, _ = batch
            
            B = a_input.shape[0]
            a_input = a_input.to(device, non_blocking=True)
            v_input = v_input.to(device, non_blocking=True)
            labels = labels.to(device)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / B)
            dnn_start_time = time.time()

            optimizer.zero_grad()
            
            with autocast():
                output, video_mask, projections = model(
                    a_input, v_input,
                    apply_mask=apply_mask,
                    mask_ratio=mask_ratio,
                    return_projections=True
                )
                
                # 1. Classification loss (real/fake)
                cls_loss = loss_fn(output, labels)
                
                # 2. Supervised contrastive loss (real vs fake clustering)
                supcon_loss = model.module.compute_contrastive_loss(projections, labels)
                
                # Total loss — no adversarial term
                total_loss = args.cls_weight * cls_loss + lambda_supcon * supcon_loss
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update meters
            cls_loss_meter.update(cls_loss.item(), B)
            supcon_loss_meter.update(supcon_loss.item(), B)
            total_loss_meter.update(total_loss.item(), B)
            
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time) / B)
            per_sample_dnn_time.update((time.time() - dnn_start_time) / B)

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {per_sample_time.avg:.5f}\t'
                      'Cls {cls_loss.val:.4f}\t'
                      'SupCon {supcon_loss.val:.4f}\t'
                      'Total {total_loss.val:.4f}'.format(
                       epoch, i, len(train_loader),
                       per_sample_time=per_sample_time,
                       cls_loss=cls_loss_meter,
                       supcon_loss=supcon_loss_meter,
                       total_loss=total_loss_meter), flush=True)
                
                if np.isnan(total_loss_meter.avg):
                    print("Training diverged...")
                    return

            end_time = time.time()
            global_step += 1
        
        print('Starting validation...')
        stats, valid_loss = validate_contrastive(model, test_loader, args)

        mAP = stats['AP_macro']
        mAUC = stats['AUC_macro']
        acc = stats['accuracy']

        if main_metrics == 'mAP':
            print("mAP: {:.6f}".format(mAP))
        else:
            print("acc: {:.6f}".format(acc))
        print("AUC: {:.6f}".format(mAUC))
        print("d_prime: {:.6f}".format(d_prime(mAUC)))
        print("train_cls_loss: {:.6f}".format(cls_loss_meter.avg))
        print("train_supcon_loss: {:.6f}".format(supcon_loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))

        result[epoch-1, :] = [acc, mAP, mAUC, optimizer.param_groups[0]['lr'],
                              supcon_loss_meter.avg]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        print('Validation finished')
        
        if mAP > best_mAP:
            best_mAP = mAP
            if main_metrics == 'mAP':
                best_epoch = epoch

        if acc > best_acc:
            best_acc = acc
            if main_metrics == 'acc':
                best_epoch = epoch

        if best_epoch == epoch:
            torch.save(model.state_dict(), "%s/models/best_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))
        if args.save_model:
            torch.save(model.state_dict(), "%s/models/model.%d.pth" % (exp_dir, epoch))
        
        scheduler.step()
            
        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        
        finish_time = time.time()
        print('Epoch {:d} training time: {:.3f}'.format(epoch, finish_time - begin_time))

        epoch += 1

        # Reset meters
        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        per_sample_dnn_time.reset()
        cls_loss_meter.reset()
        supcon_loss_meter.reset()
        total_loss_meter.reset()