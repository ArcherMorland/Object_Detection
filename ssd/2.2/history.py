import os, time, uuid
from datetime import datetime
import numpy as np


colname = dict(
    timestamp='Time Stamp',
    sysid='System ID',
    model_name='Model Name',
    version='Version',
    training_size='Training Size',
    validation_size='Validation Size',
    #'NN Structure File',
    batch_size='Batch Size',
    learning_rate='Learning Rate',
    input_size="Input Size",
    total_loss='Total Loss',
    mAP='mAP',
    reason_code='Reason Code',
    end_reason='End Reason',
    avg_epoch_time='Average Epoch Time',
    other_hyperparam='Other Hyperparametres',
    )
        
reportObj = {k : "?" for k in colname}
reportObj["sysid"]=np.base_repr(uuid.uuid3(uuid.NAMESPACE_DNS, uuid.uuid1().hex).int, base=35).rjust(25,'Z')
reportObj["timestamp"]=datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def reporting(repo_obj, over=False):
    
    savefolder=repo_obj['report_folder']
    os.makedirs(savefolder, exist_ok=True)
    status= "End  " if over else "Start"
    if over : reportObj["timestamp"]=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    stime=repo_obj['timestamp']
    sysid=reportObj['sysid']
    model_name=repo_obj['model_name']
    version=repo_obj['version']
    training_size=reportObj['training_size']
    validation_size=reportObj['validation_size']
    batch_size=reportObj['batch_size']
    learning_rate=reportObj['learning_rate']
    total_loss=0.0#reportObj['total_loss']
    avg_epoch_time=reportObj['avg_epoch_time']
    mAP=0.0#reportObj['mAP']
    

    if not over:
        message_entry=f"{stime} [{sysid}] [{model_name} ({version})] [Status: {status}] Training Size: {training_size}, Validation Size: {validation_size}, Batch Size: {batch_size}, Learning Rate: {learning_rate}"
    else:
        message_entry=f"{stime} [{sysid}] [{model_name} ({version})] [Status: {status}] Total Loss: {total_loss}, Average Epoch Time:{avg_epoch_time}, mAP:{mAP}"
        
    with open(os.path.join(savefolder,'training.repo'), 'a') as f:     
        print(message_entry, file=f)
    

    
