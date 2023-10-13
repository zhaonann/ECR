import time
from models.BDs_MLP import Diag_MLP
import utils
from data import ExCustomDataset_BDs_MLP, data_split_diag, train_val_test_split, train_val_test_split_stratify, train_test_split_stratify
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import glob
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import numpy as np
import os
import sys
import glob
import platform
import logging
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, precision_recall_fscore_support
from torchmetrics import Specificity
from torchsampler import ImbalancedDatasetSampler
from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler

def main():
    
    # initialization
    args, parser = utils.init()

    pathpar = '/public/bme/home/meilang/codes/ECR/BRAIN_AGE_ESTIMATION_CONFERENCE' # path to BRAIN_AGE_ESTIMATION
    if args.diag_mode == "Five":
        data_csv = "/public_bme/share/sMRI/csv/ECR/ADNI_Five_Class.csv" # path to data
    elif args.diag_mode == "CN_MCI_AD":
        data_csv = "/public_bme/share/sMRI/csv/ECR/ADNI_EXPAND_ALL.csv" # path to data
    elif args.diag_mode == "CN_MCI_AD_EXP":
        data_csv = "/public_bme/share/sMRI/csv/ECR/HUASHAN_ADNI_OASIS_EXPAND_ALL.csv" # path to data
    elif args.diag_mode == "CN_LMCI_AD":
        data_csv = "/public_bme/share/sMRI/csv/ECR/ADNI_CN_LMCI_AD.csv" # path to data
    elif args.diag_mode == "CN_EMCI":
        data_csv = "/public_bme/share/sMRI/csv/ECR/ADNI_CN_EMCI.csv"
    elif args.diag_mode == "CN_AD":
        data_csv = "/public_bme/share/sMRI/csv/ECR/ADNI_NC_AD.csv" # path to data
    elif args.diag_mode == "CN_LMCI":
        data_csv = "/public_bme/share/sMRI/csv/ECR/ADNI_CN_LMCI.csv" # path to data
    elif args.diag_mode == "LMCI_AD":
        data_csv = "/public_bme/share/sMRI/csv/ECR/ADNI_LMCI_AD.csv" # path to data
    elif args.diag_mode == "CN_MCI":
        data_csv = "/public_bme/share/sMRI/csv/ECR/ADNI_CN_MCI_GRAINED.csv" # path to data
    elif args.diag_mode == "MCI_AD":
        data_csv = "/public_bme/share/sMRI/csv/ECR/ADNI_MCI_AD.csv" # path to data

    outpath =os.path.join(pathpar, 'OUTPUT')
    if not os.path.exists(outpath):
        os.mkdir(outpath)   

    args.note = ""
    args.parsave = "{}-{}".format('BDs_threedim_3view_GAF', time.strftime("%Y%m%d-%H%M%S")) # spectral graph conv
    args.parsave = os.path.join(outpath, args.parsave)
    if not os.path.exists(args.parsave):
        os.makedirs(args.parsave)

    utils.print_options(args.parsave, args, parser, "train")
    # cuda
    assert args.GPU_num <= torch.cuda.device_count(), 'GPU exceed the maximum num'
    if torch.cuda.is_available():
        if args.GPU_no:
            device = torch.device("cuda:"+args.GPU_no[0])
        else:
            device = torch.device("cuda:0")
    else:
        logging.info("no gpu device available")
        device = torch.device('cpu')

    data_ACC = np.zeros((3, 2))

    utils.seed_torch(seed=3407) 

    for exp in range(0, args.n_exps):

        print("******** Training on exp %d ********" % (exp+1)) 
        
        args.save = os.path.join(args.parsave, "exp_"+str(exp))
        if not os.path.exists(args.save):
            os.mkdir(args.save)

        pathpar1 = os.path.abspath(os.getcwd())
        utils.create_exp_dir(args.save, scripts_to_save=glob.glob(os.path.join(pathpar1, "models", "BDs_MLP.py")))
        utils.create_exp_dir(args.save, scripts_to_save=glob.glob(os.path.join(pathpar1, "train_BDs_redist.py")))
        utils.create_exp_dir(args.save, scripts_to_save=glob.glob(os.path.join(pathpar1, "utils.py")))
        utils.create_exp_dir(args.save, scripts_to_save=glob.glob(os.path.join(pathpar1, "data.py")))

        path_logging = os.path.join(args.save, 'logging')
        if not os.path.exists(path_logging):
            os.makedirs(path_logging)

        log_format = "%(asctime)s %(message)s"
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.INFO,
            format=log_format,
            datefmt="%m/%d %I:%M:%S %p",
        )
        fh = logging.FileHandler(os.path.join(path_logging, "log.txt"))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)

        dash_writer = SummaryWriter(os.path.join(args.save, "TENSORBOARD"))

        Layout = {}
        Layout["RESULT" ] = {
            "Train_Loss": ["Multiline", ["sMRI"]],
            "Val_Loss": ["Multiline", ["sMRI"]],
            "Train_ACC": ["Multiline", ["sMRI"]],
            "Val_ACC": ["Multiline", ["sMRI"]],
        }
        dash_writer.add_custom_scalars(Layout)

        path_weights = os.path.join(args.save, "WEIGHTS")
        if not os.path.exists(path_weights):
            os.makedirs(path_weights)


        path_figs = os.path.join(args.save, "FIGURES_CSV")
        if not os.path.exists(path_figs):
            os.makedirs(path_figs)

        num_workers = 0 if platform.system() == "Windows" else 8
        
        # df = pd.read_csv(data_csv)
        # dataset = ExCustomDataset_BDs_MLP(df=df, transforms=False)

        # train_sampler, val_sampler, in_test_sampler, _, n_trains, n_vals, n_in_tests, _ = data_split_diag(dataset)

        # train_queue = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler,
        #                         shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)

        # dataset = ExCustomDataset_BDs_MLP(df=df, transforms=False)

        # val_queue = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler,
        #                         shuffle=False, num_workers=num_workers, pin_memory=True)
        # in_test_queue = DataLoader(dataset, batch_size=args.batch_size, sampler=in_test_sampler,
        #                         shuffle=False, num_workers=num_workers, pin_memory=True)

        # loss func
        if args.diag_mode == "Five":
            target_name = ["CN", "EMCI", 'MCI', "LMCI", "AD"]
            n_HC = 56 + 82
            n_EMCI = 344
            n_MCI = 100
            n_LMCI = 207
            n_AD = 132
            n_total = n_HC + n_EMCI + n_MCI + n_LMCI + n_AD
            w1 = 0.25*( n_EMCI + n_MCI + n_LMCI + n_AD)/n_total
            w2 = 0.25*(n_HC  + n_MCI + n_LMCI + n_AD)/n_total
            w3 = 0.25*(n_HC + n_EMCI  + n_LMCI + n_AD)/n_total
            w4 = 0.25*(n_HC + n_EMCI + n_MCI +  n_AD)/n_total
            w5 = 0.25*(n_HC + n_EMCI + n_MCI + n_LMCI )/n_total
            sample_weight = torch.tensor([w1, w2, w3, w4, w5]).to(device)
            class_weights = [w1, w2, w3]
        elif args.diag_mode == "CN_MCI_AD":
            target_name = ["CN", 'MCI', "AD"]
            n_HC = 56 + 82
            n_MCI = 651
            n_AD = 132
            n_total = n_HC + n_MCI + n_AD
            w1 = 0.5*(n_MCI + n_AD)/n_total
            w2 = 0.5*(n_HC + n_AD)/n_total
            w3 = 0.5*(n_HC + n_MCI)/n_total
            sample_weight = torch.tensor([w1, w2, w3]).to(device)
            class_weights = [w1, w2, w3]
        elif args.diag_mode == "CN_MCI_AD_EXP":
            target_name = ["CN_Ex", 'MCI_Ex', "AD_Ex"]
            n_HC = 56 + 82
            n_MCI = 651 + 100
            n_AD = 132 + 43
            n_total = n_HC + n_MCI + n_AD
            w1 = 0.5*(n_MCI + n_AD)/n_total
            w2 = 0.5*(n_HC + n_AD)/n_total
            w3 = 0.5*(n_HC + n_MCI)/n_total
            sample_weight = torch.tensor([w1, w2, w3]).to(device)
            class_weights = [w1, w2, w3]
        elif args.diag_mode == "CN_LMCI_AD":
            target_name = ["CN", 'LMCI', "AD"]
            n_HC = 56 + 82
            n_MCI = 207
            n_AD = 132
            n_total = n_HC + n_MCI + n_AD
            w1 = 0.5*(n_MCI + n_AD)/n_total
            w2 = 0.5*(n_HC + n_AD)/n_total
            w3 = 0.5*(n_HC + n_MCI)/n_total
            sample_weight = torch.tensor([w1, w2, w3]).to(device)
            class_weights = [w1, w2, w3]
        elif args.diag_mode == "CN_EMCI":
            target_name = ["CN", "EMCI"]
            # sample_weight = None
            n_HC = 56 + 82
            n_MCI = 344
            n_total = n_HC + n_MCI
            w1 = (n_MCI)/n_total
            w2 = (n_HC)/n_total
            sample_weight = torch.tensor([w1, w2]).to(device)    
            class_weights = [w1, w2]
        elif args.diag_mode == "CN_AD":
            target_name = ["CN", "AD"]
            n_HC = 56 + 82
            n_AD = 132
            n_total = n_HC + n_AD
            w1 = (n_AD)/n_total
            w2 = (n_HC)/n_total
            sample_weight = torch.tensor([w1, w2]).to(device)   
            sample_weight = None       
        elif args.diag_mode == "CN_MCI":
            target_name = ["CN", "MCI"]
            n_HC = 56 + 82
            n_MCI = 100
            n_total = n_HC + n_MCI
            w1 = (n_MCI)/n_total
            w2 = (n_HC)/n_total
            sample_weight = torch.tensor([w1, w2]).to(device)       
        elif args.diag_mode == "CN_LMCI":
            target_name = ["CN", "LMCI"]
            n_HC = 56 + 82
            n_MCI = 207 # LMCI
            n_total = n_HC + n_MCI
            w1 = (n_MCI)/n_total
            w2 = (n_HC)/n_total
            sample_weight = torch.tensor([w1, w2]).to(device)      
        elif args.diag_mode == "LMCI_AD":
            target_name = ["LMCI", "AD"]
            n_HC = 207 # LMCI
            n_MCI = 132 # AD
            n_total = n_HC + n_MCI
            w1 = (n_MCI)/n_total
            w2 = (n_HC)/n_total
            sample_weight = torch.tensor([w1, w2]).to(device)     
        elif args.diag_mode == "MCI_AD":
            target_name = ["CN", "MCI"]
            n_MCI = 651
            n_AD = 132
            n_total = n_MCI + n_AD
            w1 = (n_AD)/n_total
            w2 = (n_MCI)/n_total
            sample_weight = torch.tensor([w1, w2]).to(device) 

        df_train, df_val, df_in_test, _, n_trains, n_vals, n_in_tests, _ = train_test_split_stratify(data_csv)

        dataset_train = ExCustomDataset_BDs_MLP(df=df_train, transforms=True)

        # train_sampler, val_sampler, in_test_sampler, _, n_trains, n_vals, n_in_tests, _ = data_split_diag(dataset)
        # labels = dataset_train.get_labels().tolist()
        # weights = [1.0 / class_weights[labels[i]] for i in range(n_trains)]

        # train_sampler_weight = WeightedRandomSampler(weights = weights, num_samples = len(weights), replacement = True)
        train_sampler = RandomSampler(dataset_train)

        train_queue = DataLoader(dataset_train, batch_size=args.batch_size, sampler=train_sampler,
                                shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)

        dataset_val = ExCustomDataset_BDs_MLP(df=df_val, transforms=False)

        val_queue = DataLoader(dataset_val, batch_size=args.batch_size, sampler=RandomSampler(dataset_val),
                                shuffle=False, num_workers=num_workers, pin_memory=True)
        
        # dataset_in_test = ExCustomDataset_BDs_MLP(df=df_in_test, transforms=False)
        # in_test_queue = DataLoader(dataset_in_test, batch_size=args.batch_size, sampler=RandomSampler(dataset_in_test),
        #                         shuffle=False, num_workers=num_workers, pin_memory=True)

        if args.diag_mode == "Five":
            n_classes = 5
        elif args.diag_mode.startswith("CN_MCI_AD"):
            n_classes = 3 # CN, MCI, AD
        elif args.diag_mode.startswith("CN_LMCI_AD"):
            n_classes = 3 # CN, LMCI, AD
        else:
            n_classes = 2
            
        model_s = Diag_MLP(in_channels=1, n_classes=n_classes).to(device)

        logging.info("sMRI, param size = %.3f MB", utils.count_parameters_in_MB(model_s))
        utils.print_networks(model_s, "sMRI")

        if args.pretrain_age_model:
            # threedim_path = "/public/bme/home/zhaonan/brain_age_est/codes/OUTPUT/threedim-20230731-212100/exp_0/WEIGHTS/model_s_best_weight.pkl"
            pretrain = "/public/bme/home/meilang/codes/ECR/BRAIN_AGE_ESTIMATION_CONFERENCE/OUTPUT/threedim_3view_GAF-20230811-153250_p/exp_0/WEIGHTS/model_s_best_weight.pkl"
            utils.load_pretrain_age_model(model = model_s, model_path_pretrain = pretrain)
            print("Load pretrain age model!", flush=True)

        optimizer_s = optim.AdamW(model_s.parameters(), lr=args.lr_s, weight_decay=args.wd_s)
        scheduler_s = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s, float(args.n_epochs))

        loss_func = nn.CrossEntropyLoss(weight=sample_weight, reduction='mean') # weight: Tensor of size C

        losses = {"train_CE_sMRI": [], "val_CE_sMRI": []}
        accs = {"train_acc_sMRI": [], "val_acc_sMRI": []}
        val_opt = {"Epoch": 0, "Opt_val": 0.0, "Opt_train": 0.0}

        for epoch in range(1, args.n_epochs+1):

            epoch_start_time = time.time()
            loss_dict = {"sMRI": []}
            acc_dict = {"sMRI": 0.0}

            if epoch > 1:
                scheduler_s.step()
            
            acc = 0.0
            model_s.train()

            for data in train_queue:
                sMRI, axial, sagittal, coronal, GAF, age_gt, sex, diag_gt = data[:8]
                sMRI = sMRI.to(device)
                axial = axial.to(device)
                sagittal = sagittal.to(device)
                coronal = coronal.to(device)
                GAF = GAF.to(device)
                age_gt = age_gt.unsqueeze(1).to(device)
                sex = sex.unsqueeze(1).to(device)
                diag_gt = diag_gt.to(device)
                
                optimizer_s.zero_grad()

                diag_logits = model_s(sMRI, axial, sagittal, coronal, GAF, age_gt, sex)
                
                diag_prob = nn.Softmax(dim=1)(diag_logits)

                loss_s = loss_func(diag_prob, diag_gt)

                loss_dict['sMRI'].append(loss_s.cpu().item())
                loss_s.backward()

                optimizer_s.step()

                # mae_dict['sMRI'] += torch.sum(torch.abs(age_gt - age_pred)).item()

                diag_pred = diag_prob.argmax(dim=1).tolist()
                acc += accuracy_score(y_true=diag_gt.tolist(), y_pred=diag_pred, normalize=False)

            loss_dict['sMRI'] = np.average(loss_dict['sMRI'])
            
            # mae_dict['sMRI'] = (mae_dict['sMRI']/n_trains)
            acc_dict['sMRI'] = acc/n_trains

            losses['train_CE_sMRI'].append(loss_dict['sMRI'])

            accs['train_acc_sMRI'].append(acc_dict['sMRI'])

            dash_writer.add_scalars("RESULT/Train_Loss", loss_dict, epoch)  
            dash_writer.add_scalars("RESULT/Train_ACC", acc_dict, epoch)  

            train_loss = loss_dict['sMRI']
            train_acc = acc_dict['sMRI']

            loss_dict = {"sMRI": []}
            acc_dict = {"sMRI": 0.0}
           
            acc = 0.0
            model_s.eval()
            with torch.no_grad():   
                for data in val_queue:
                    sMRI, axial, sagittal, coronal, GAF, age_gt, sex, diag_gt = data[:8]
                    sMRI = sMRI.to(device)
                    axial = axial.to(device)
                    sagittal = sagittal.to(device)
                    coronal = coronal.to(device)
                    GAF = GAF.to(device)
                    age_gt = age_gt.unsqueeze(1).to(device)
                    sex = sex.unsqueeze(1).to(device)
                    diag_gt = diag_gt.to(device)

                    diag_logits = model_s(sMRI, axial, sagittal, coronal, GAF, age_gt, sex)

                    diag_prob = nn.Softmax(dim=1)(diag_logits)

                    loss_s = loss_func(diag_prob, diag_gt)
                    loss_dict['sMRI'].append(loss_s.cpu().item())

                    # mae_dict['sMRI'] += torch.sum(torch.abs(age_gt - age_pred)).item()
                    diag_pred = diag_prob.argmax(dim=1).tolist()
                    acc += accuracy_score(y_true=diag_gt.tolist(), y_pred=diag_pred, normalize=False)

            loss_dict['sMRI'] = np.average(loss_dict['sMRI'])

            # mae_dict['sMRI'] = (mae_dict['sMRI']/n_vals)
            acc_dict['sMRI'] = acc/n_vals

            if acc_dict['sMRI'] >= val_opt["Opt_val"]: # record the best
                val_opt["Opt_val"] = acc_dict['sMRI']
                val_opt["Opt_train"] = train_acc
                val_opt["Epoch"] = epoch
                model_path_s = os.path.join(path_weights, 'model_s_best_weight.pkl')
                torch.save(model_s.state_dict(), model_path_s)

            losses['val_CE_sMRI'].append(loss_dict['sMRI'])

            accs['val_acc_sMRI'].append(acc_dict['sMRI'])

            dash_writer.add_scalars("RESULT/Val_Loss", loss_dict, epoch)  
            dash_writer.add_scalars("RESULT/Val_MAE", acc_dict, epoch)  

            logging.info("******* Epoch %d, Train CE Loss %.2f, Test CE Loss %.2f, Train ACC %.2f, Test ACC %.2f *******", epoch, train_loss, loss_dict['sMRI'], train_acc, acc_dict['sMRI'])

            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, args.n_epochs, time.time() - epoch_start_time))

        # model save
        model_path_s = os.path.join(path_weights, 'model_s_epoch_%d.pkl'%args.n_epochs)
        torch.save(model_s.state_dict(), model_path_s)

        val_acc_s = preds_store(model_s, val_queue, n_vals, stage="last_in_test", device=device, path_figs=path_figs, exp=exp, target_name=target_name, n_classes=n_classes)
        # in_test_acc_s = preds_store(model_s, in_test_queue, n_in_tests, stage="last_in_test", device=device, path_figs=path_figs, exp=exp, target_name=target_name, n_classes=n_classes)

        logging.info("*******Again Last Epoch %d,  Val ACC %.2f*******",
                     args.n_epochs, val_acc_s)

        model_path_s = os.path.join(path_weights, 'model_s_best_weight.pkl')
        model_s.load_state_dict(torch.load(model_path_s))

        # in_test_acc_s = preds_store(model_s, in_test_queue, n_in_tests, stage="best_in_test", device=device, path_figs=path_figs, exp=exp, target_name=target_name, n_classes=n_classes)
        val_acc_s = preds_store(model_s, val_queue, n_vals, stage="best_in_test", device=device, path_figs=path_figs, exp=exp, target_name=target_name, n_classes=n_classes)

        logging.info("******* Best Epoch %d, Train ACC %.2f, Val MAE %.2f, In Test MAE %.2f *******", 
                     val_opt["Epoch"], val_opt["Opt_train"], val_opt["Opt_val"], val_acc_s)
      
        data_ACC[exp][:2] = val_opt["Opt_val"], val_acc_s

        print('**** Exp %d Finished Training! ****' % exp)

        logging.shutdown()

        plt.figure(figsize=(20, 10))
        plt.plot(losses['train_CE_sMRI'], 'r--', label='train/sMRI')
        plt.plot(losses['val_CE_sMRI'], 'g--', label='val/sMRI')
        plt.xlabel("Epoch")
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss of exp {}'.format(str(exp)))
        plt.savefig(os.path.join(path_figs, "exp_%d_loss_history"%exp + ".png"))
    
        plt.figure(figsize=(20, 10))
        plt.plot(accs['train_acc_sMRI'], 'r--', label='train/sMRI')
        plt.plot(accs['val_acc_sMRI'], 'g--', label='val/sMRI')
        plt.xlabel("Epoch")
        plt.ylabel('MAE')
        plt.legend()
        plt.title('MAE of exp {}'.format(str(exp)))
        plt.savefig(os.path.join(path_figs, "exp_%d_mae_history"%exp + ".png"))

    utils.calc_MAE(data_ACC, args.parsave, args.n_exps)


def preds_store(model_s, queue, n_datas, stage="val", device=None, path_figs=None, exp=0, target_name=['CN', 'MCI', 'AD'], n_classes=5):
    """
    Save the age prediction.
    """
    diag_preds = []
    diag_gts = []
    IDs = []
    filenames = []
    sites = []
    sexs = []
    ages = []
    acc = 0.0

    # loss_func = nn.CrossEntropyLoss(reduction='mean')

    model_s.eval()
    with torch.no_grad():
        for data in tqdm(queue):
            sMRI, axial, sagittal, coronal, GAF, age_gt, sex, diag_gt, ID, filename, site = data
            sMRI = sMRI.to(device)
            axial = axial.to(device)
            sagittal = sagittal.to(device)
            coronal = coronal.to(device)
            GAF = GAF.to(device)
            age_gt = age_gt.unsqueeze(1).to(device)
            sex = sex.unsqueeze(1).to(device)
            diag_gt = diag_gt.to(device)

            diag_logits = model_s(sMRI, axial, sagittal, coronal, GAF, age_gt, sex)

            # maes += torch.sum(torch.abs(age_gt - age_pred)).item()
            diag_prob = nn.Softmax(dim=1)(diag_logits)

            # loss_s = loss_func(diag_prob, diag_gt)

            # mae_dict['sMRI'] += torch.sum(torch.abs(age_gt - age_pred)).item()
            diag_pred = diag_prob.argmax(dim=1).tolist()
            acc += accuracy_score(y_true=diag_gt.tolist(), y_pred=diag_pred, normalize=False)

            diag_preds.extend(diag_pred)
            diag_gts.extend(diag_gt.tolist())

            IDs.extend(ID.tolist())
            filenames.extend(filename)
            sites.extend(site.tolist())
            sexs.extend(sex.squeeze(1).tolist())
            ages.extend(age_gt.squeeze(1).tolist())

    acc_norm = acc / n_datas
    ages = np.around(ages, decimals=3)
    df = pd.DataFrame({'No': IDs, 'filename': filenames, 'diag_pred': diag_preds, 'diag_gt': diag_gts,
                       'age': ages, 'site': sites, 'sex': sexs})  # 1->male, 0-> female
    save_path = os.path.join(
        path_figs, 'diag_prediction_{}_exp_{}.csv'.format(stage, str(exp)))
    df.to_csv(save_path, index=False)
    
    res = classification_report(diag_gts, diag_preds, target_names=target_name)

    logging.info("******* Classification Report {}:*******".format(stage))
    logging.info("{}".format(res))
    
    spe_func = Specificity(task="multiclass", average="weighted", num_classes=n_classes) # binary
    spe_res = spe_func(torch.tensor(diag_preds), torch.tensor(diag_gts))

    logging.info("******* Classification Spe {:.2f}:*******".format(spe_res))
    return acc_norm

if __name__ == '__main__':
    main()
