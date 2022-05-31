import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import matplotlib.pyplot as plt
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

# 实例化
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

# try:
#     for epoch in range(world.TRAIN_epochs):
#         start = time.time()
#         if epoch %10 == 0:
#             cprint("[TEST]")
#             #Procedure.Test_threhold(dataset, Recmodel, epoch, w, world.config['multicore'])
#             Procedure.Test_Pretrain(dataset, Recmodel, epoch, w, world.config['multicore'])
#         output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
#         print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
#         torch.save(Recmodel.state_dict(), weight_file)
# finally:
#     if world.tensorboard:
#         w.close()

try:
    epoches = []
    precisions = []
    recalls = []
    f1s = []
    ndcgs = []

    for i in range(74):
        epoch_num = i * 10
        epoches.append(epoch_num)
        cprint("[TEST]")
        results = Procedure.Test_Pretrain(dataset, Recmodel, epoch_num, w, world.config['multicore'])
        precisions.append(results['precision'][0])
        recalls.append(results['recall'][0])
        f1s.append(results['f1'][0])
        ndcgs.append(results['ndcg'][0])

    plt.plot(epoches, precisions, label="precision")
    plt.plot(epoches, recalls, label="recall")
    plt.plot(epoches, f1s, label="f1")
    plt.plot(epoches, ndcgs, label="ndcg")

    # naming the x axis
    plt.xlabel('epoches')
    # naming the y axis
    plt.ylabel('')
    # giving a title to my graph
    plt.title('topk=100 with exclude')

    # show a legend on the plot
    plt.legend()

    # function to show the plot
    plt.savefig('topk100+exclude')

    plt.show()
finally:
    if world.tensorboard:
        w.close()