from __future__ import print_function
import sys
import glob
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
import gc

import dataset
from history import *
from utils import *
from image import correct_yolo_boxes
from cfg import parse_cfg
from darknet import Darknet
import argparse

FLAGS = None
unparsed = None
device = None

# global variables
# Training settings
# Train parameters
use_cuda      = None
eps           = 1e-5
keep_backup   = 10
save_interval = 1  # epoches
test_interval = 1  # epoches
dot_interval  = 70  # batches

# Test parameters
evaluate = False
conf_thresh   = 0.25
nms_thresh    = 0.4
iou_thresh    = 0.5

# no test evalulation
no_eval = False
init_eval = False


# Model Summury
def model_summary(model):
    print("model_summary")
    print()
    print("Layer_name"+"\t"*7+"Number of Parameters")
    print("="*100)
    model_parameters = [layer for layer in model.parameters() if layer.requires_grad]
    layer_name = [child for child in model.children()]
    j = 0
    total_params = 0
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\t"*10)
    for i in layer_name:
        print()
        param = 0
        try:
            bias = (i.bias is not None)
        except:
            bias = False  
        if not bias:
            param =model_parameters[j].numel()+model_parameters[j+1].numel()
            j = j+2
        else:
            param =model_parameters[j].numel()
            j = j+1
        print(str(i)+"\t"*3+str(param))
        total_params+=param
    print("="*100)
    print(f"Total Params:{total_params}")
    print(f'pytorch_total_params : {pytorch_total_params}')
    print('\n\n')

# Training settings
def load_testlist(testlist):
    init_width = model.width
    init_height = model.height

    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    #kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    loader = torch.utils.data.DataLoader(
        dataset.listDataset(testlist,
                            imgRoot=image_folder,
                            wdatalist=None,
                            odweight=None,
                            shape=(init_width, init_height),
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                ]),
                            train=False),
        batch_size=batch_size, shuffle=False, **kwargs)
    return loader

def main():
    datacfg    = "cfg/c.data"#FLAGS.data
    cfgfile    = "cfg/model_structure.cfg"#FLAGS.config
    loc_cfg    = "cfg/setting.config"
    main_cfg   = "../../Training.config"

    localmax = False  # FLAGS.localmax
    no_eval    = False#FLAGS.no_eval
    init_eval  = False#FLAGS.init_eval

    data_options  = read_data_cfg(datacfg)
    main_options   = read_data_cfg(main_cfg)
    loc_options = read_data_cfg( loc_cfg )
    net_options   = parse_cfg(cfgfile)[0]

    global use_cuda
    use_cuda = torch.cuda.is_available() and (True if use_cuda is None else use_cuda)
    repo=reportObj
    globals()["trainlist"]     = main_options['train']#data_options['train']
    globals()["testlist"]      = main_options['valid']#data_options['valid']
    globals()["names"]         = main_options['names']
    print(main_options['names'])
    modify_nn_file(main_options['names'])
    
    globals()["image_folder"]  = main_options['image_folder']
    globals()["log_folder"]    = repo["log_folder"] = main_options['log_folder']
    globals()["report_folder"] = repo["report_folder"] = main_options['report_folder']
    
    globals()["backupdir"]     = data_options['backup']
    globals()["gpus"]          = data_options['gpus']  # e.g. 0,1,2,3
    globals()["ngpus"]         = len(gpus.split(','))
    globals()["num_workers"]   = int(data_options['num_workers'])

    globals()["batch_size"]    = repo["batch_size"] = int(main_options['batch'])
    globals()["max_batches"]   = int(main_options['max_batches'])
    globals()["learning_rate"] = repo["learning_rate"] = float(main_options['learning_rate'])
    globals()["momentum"]      = float(net_options['momentum'])
    globals()["decay"]         = float(net_options['decay'])
    globals()["steps"]         = [float(step) for step in net_options['steps'].split(',')]
    globals()["scales"]        = [float(scale) for scale in net_options['scales'].split(',')]

    training_proc_reset = eval(loc_options['training_process_init'])
    setwdata = eval(loc_options['setwdata'])
    weightfiles=[int(os.path.split(f)[1].replace('.weights', '')) for f in glob.glob(os.path.join(loc_options['weightfolder'], '*.weights'))]
    #latest_weightfile
    weightfile = loc_options['weightfile'] if training_proc_reset else os.path.join(loc_options['weightfolder'],str(max(weightfiles)).rjust(6,'0')+".weights") # FLAGS.weights
    globals()["wdata"] = loc_options['wdata'] if setwdata else None
    globals()["odw"] = int(loc_options["origd_weighting"]) if setwdata else None
    repo["model_name"]="YOLOv3"
    repo["version"]="v4.1"
    print(trainlist)
    print(testlist)
    print(wdata)
    print(weightfile)
    #print(p)
    global max_epochs
    try:
        max_epochs = int(main_options['max_epochs'])
    except KeyError:
        nsamples = file_lines(trainlist)
        max_epochs = (max_batches*batch_size)//nsamples+1

    seed = int(time.time())
    torch.manual_seed(seed)
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)
    global device
    device = torch.device("cuda" if use_cuda else "cpu")

    global model
    model = Darknet(cfgfile, use_cuda=use_cuda)
    if weightfile is not None:
        model.load_weights(weightfile)
    else:
        model.load_weights('weights/pretrained_weight/yolov3.weights')

    #model.print_network()
    model_summary(model)
    #print(p)
    nsamples = repo["training_size"] =file_lines(trainlist)
    repo["validation_size"] =file_lines(testlist)
    #initialize the model
    if training_proc_reset:
        model.seen = 0
        init_epoch = 0
    else:
        init_epoch = model.seen//nsamples

    global loss_layers
    loss_layers = model.loss_layers
    for l in loss_layers:
        l.seen = model.seen

    globals()["test_loader"] = load_testlist(testlist)
    if use_cuda:
        if ngpus > 1:
            model = torch.nn.DataParallel(model).to(device)
        else:
            model = model.to(device)

    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if key.find('.bn') >= 0 or key.find('.bias') >= 0:
            params += [{'params': [value], 'weight_decay': 0.0}]
        else:
            params += [{'params': [value], 'weight_decay': decay*batch_size}]
    global optimizer
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate/batch_size, momentum=momentum,  dampening=0, weight_decay=decay*batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    

    if evaluate:
        logging('evaluating ...')
        test(0)
    else:
        try:
            print("Training for ({:d},{:d})".format(init_epoch+1, max_epochs))
            print("Making Starting Report...")            
            reporting(repo)
            
            fscore = 0
            correct = 0
            if init_eval and not no_eval and init_epoch > test_interval:
                print('>> initial evaluating ...')
                mcorrect,mfscore = test(init_epoch)
                print('>> done evaluation.')
            else:
                mfscore = 0.5
                mcorrect = 0
            t1=time.time()
            for cnt, epoch in enumerate(range(init_epoch+1, max_epochs+1)):
                print(f"epoch number: {epoch}")
                nsamples = train(epoch)
                if epoch % save_interval == 0:
                    savemodel(epoch, nsamples)
                if not no_eval and epoch >= test_interval and (epoch%test_interval) == 0:
                    print('>> interim evaluating ...')
                    correct, fscore = test(epoch)
                    print('>> done evaluation.')
                if localmax and correct > mcorrect:
                    mfscore = fscore
                    mcorrect = correct
                    savemodel(epoch, nsamples, True)
                t2=time.time()
                reportObj['avg_epoch_time']=str(round((t2-t1)/(cnt+1), 2)).ljust(9,' ')
                print('-'*90)
            
        except KeyboardInterrupt:
            print('='*80)
            print('Exiting from training by interrupt')
            

        finally:
            print("Making End Report...")
            reporting(repo, True)
                
def adjust_learning_rate(optimizer, batch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/batch_size
    return lr

def curmodel():
    if ngpus > 1:
        cur_model = model.module
    else:
        cur_model = model
    return cur_model

def train(epoch):
    global processed_batches
    t0 = time.time()
    cur_model = curmodel()
    init_width = cur_model.width
    init_height = cur_model.height
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    #kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
                dataset.listDataset(trainlist,
                                    imgRoot=image_folder,
                                    wdatalist=wdata,
                                    odweight=odw,
                                    shape=(init_width, init_height),
                                    shuffle=False,#True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        ]), 
                                    train=True,
                                    seen=cur_model.seen,
                                    batch_size=batch_size,
                                    num_workers=num_workers
                                    ),
                collate_fn=dataset.custom_collate, 
                batch_size=batch_size, shuffle=False, **kwargs)

    processed_batches = cur_model.seen//batch_size
    #lr = adjust_learning_rate(optimizer, processed_batches)
    for param_group in optimizer.param_groups:
        lr=param_group['lr']
    logging('[%03d] processed %d samples, lr %e' % (epoch, epoch * len(train_loader.dataset), lr))
    #logging('[%03d] processed %d samples' % (epoch, epoch * len(train_loader.dataset)))
    
    
    model.train()
    t1 = time.time()
    avg_time = torch.zeros(9)
    for batch_idx, (data, target) in enumerate(train_loader):
        t2 = time.time()
        #adjust_learning_rate(optimizer, processed_batches)
        processed_batches = processed_batches + 1
        #if (batch_idx+1) % dot_interval == 0:
        #    sys.stdout.write('.')

        t3 = time.time()
        data, target = data.to(device), target.to(device)

        t4 = time.time()
        optimizer.zero_grad()

        t5 = time.time()
        output = model(data)
        
        t6 = time.time()
        org_loss = []
        for i, l in enumerate(loss_layers):
            l.seen = l.seen + data.data.size(0)
            ol=l(output[i]['x'], target)
            org_loss.append(ol)
        
        t7 = time.time()

        #for i, l in enumerate(reversed(org_loss)):
        #    l.backward(retain_graph=True if i < len(org_loss)-1 else False)
        # org_loss.reverse()
        sum(org_loss).backward()

        nn.utils.clip_grad_norm_(model.parameters(), 10000)
        #for p in model.parameters():
        #    p.data.add_(-lr, p.grad.data)

        t8 = time.time()
        optimizer.step()
        
        t9 = time.time()
        if False and batch_idx > 1:
            avg_time[0] = avg_time[0] + (t2-t1)
            avg_time[1] = avg_time[1] + (t3-t2)
            avg_time[2] = avg_time[2] + (t4-t3)
            avg_time[3] = avg_time[3] + (t5-t4)
            avg_time[4] = avg_time[4] + (t6-t5)
            avg_time[5] = avg_time[5] + (t7-t6)
            avg_time[6] = avg_time[6] + (t8-t7)
            avg_time[7] = avg_time[7] + (t9-t8)
            avg_time[8] = avg_time[8] + (t9-t1)
            print('-------------------------------')
            print('       load data : %f' % (avg_time[0]/(batch_idx)))
            print('     cpu to cuda : %f' % (avg_time[1]/(batch_idx)))
            print('cuda to variable : %f' % (avg_time[2]/(batch_idx)))
            print('       zero_grad : %f' % (avg_time[3]/(batch_idx)))
            print(' forward feature : %f' % (avg_time[4]/(batch_idx)))
            print('    forward loss : %f' % (avg_time[5]/(batch_idx)))
            print('        backward : %f' % (avg_time[6]/(batch_idx)))
            print('            step : %f' % (avg_time[7]/(batch_idx)))
            print('           total : %f' % (avg_time[8]/(batch_idx)))
        t1 = time.time()
        del data, target
        org_loss.clear()
        gc.collect()

    print('')
    t1 = time.time()
    nsamples = len(train_loader.dataset)
    logging('[%03d] training with %f samples/s' % (epoch, nsamples/(t1-t0)))
    return nsamples
    
def savemodel(epoch, nsamples, curmax=False):
    cur_model = curmodel()
    if curmax:
        logging('save local maximum weights to %s/localmax.weights' % (backupdir))
    else:
        logging('save weights to %s/%06d.weights' % (backupdir, epoch))
    cur_model.seen = epoch * nsamples
    if curmax: 
        cur_model.save_weights('%s/localmax.weights' % (backupdir))
    else:
        cur_model.save_weights('%s/%06d.weights' % (backupdir, epoch))
        old_wgts = '%s/%06d.weights' % (backupdir, epoch-keep_backup*save_interval)
        try: #  it avoids the unnecessary call to os.path.exists()
            os.remove(old_wgts)
        except OSError:
            pass

def test(epoch):
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i
        return 50

    model.eval()
    cur_model = curmodel()
    num_classes = cur_model.num_classes
    total       = 0.0
    proposals   = 0.0
    correct     = 0.0

    if cur_model.net_name() == 'region': # region_layer
        shape=(0,0)
    else:
        shape=(cur_model.width, cur_model.height)
    with torch.no_grad():
        for data, target, org_w, org_h in test_loader:
            data = data.to(device)
            output = model(data)
            all_boxes = get_all_boxes(output, shape, conf_thresh, num_classes, use_cuda=use_cuda)

            for k in range(len(all_boxes)):
                boxes = all_boxes[k]
                correct_yolo_boxes(boxes, org_w[k], org_h[k], cur_model.width, cur_model.height)
                boxes = np.array(nms(boxes, nms_thresh))

                truths = target[k].view(-1, 5)
                num_gts = truths_length(truths)
                total = total + num_gts
                num_pred = len(boxes)
                if num_pred == 0:
                    continue

                proposals += int((boxes[:,4]>conf_thresh).sum())
                for i in range(num_gts):
                    gt_boxes = torch.FloatTensor([truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]])
                    gt_boxes = gt_boxes.repeat(num_pred,1).t()
                    pred_boxes = torch.FloatTensor(boxes).t()
                    best_iou, best_j = torch.max(multi_bbox_ious(gt_boxes, pred_boxes, x1y1x2y2=False),0)
                    # pred_boxes and gt_boxes are transposed for torch.max
                    #print(f"pred_boxes[6][best_j]: {pred_boxes[6][best_j]} ; gt_boxes[6][0]: {gt_boxes[6][0]}; pred_boxes[6][best_j] == gt_boxes[6][0]===>>>{pred_boxes[6][best_j] == gt_boxes[6][0]}")                       
                    if best_iou > iou_thresh and pred_boxes[6][best_j] == gt_boxes[6][0]:
                        correct += 1
                        
    precision = 1.0*correct/(proposals+eps)
    recall = 1.0*correct/(total+eps)
    fscore = 2.0*precision*recall/(precision+recall+eps)
    savelog(log_folder, "[%03d] correct: %d, proposals: %d, precision: %f, recall: %f, fscore: %f" % (epoch, correct, proposals, precision, recall, fscore))
    return correct,fscore

if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--data', '-d',
     #   type=str, default="cfg/custom_usba.data", help='data definition file')
    
    #parser.add_argument('--config', '-c',
     #   type=str, default="cfg/yolov3-custom_usba.cfg", help='network configuration file')
    
    #parser.add_argument('--weights', '-w',
     #   type=str, default="weights/000226.weights",help='initial weights file')
    
    #parser.add_argument('--initeval', '-i', dest='init_eval', action='store_true',
     #   help='performs inital evalulation')
    
    #parser.add_argument('--noeval', '-n', dest='no_eval', action='store_true',
     #   help='prohibit test evalulation')
    
    #parser.add_argument('--reset', '-r',
     #   action="store_true", default=False, help='initialize the epoch and model seen value')
    
    #parser.add_argument('--localmax', '-l',
     #   action="store_true", default=False, help='save net weights for local maximum fscore')

    #FLAGS, _ = parser.parse_known_args()
    main()

