import os
import sys
import torch
sys.path.append(str(os.getcwd()))

from parametricSN.utils.helpers import experiments_cli, experiments_mpCommands

mlflow_exp_name = os.path.basename(__file__)
PROCESS_BATCH_SIZE = 2


RUN_FILE = "parametricSN/main.py"
OPTIM = "sgd"
LR = 0.1
LRS = 0.1
LRO = 0.1
LRMAX = 0.2
DF = 25
LEARNABLE = 1
EPOCHS = 500
INIT = "Tight-Frame"
RUNS_PER_SEED = 10
SCHEDULER = "OneCycleLR"
TRAIN_SAMPLE_NUM = 1000
TRAIN_BATCH_SIZE = 1000
AUGMENT = "autoaugment"
SCATT_PARAM_DISTANCE = 1


if __name__ == '__main__':
    PYTHON, DATA_ARG = experiments_cli()

    commands = []

    for SEED in [207715039]:#491659600]:#,207715039]:#,491659600,493572006,737523103]:#,827192296,877498678,1103100946,1210393663,1277404878,1377264326]:

        for aa in [(1,"Tight-Frame"),(1,"Random")]:
            LEARNABLE, INIT = aa

            command = "{} {} run-train -oname {} -olr {} -gseed {} -sl {} -me {} -omaxlr {} -odivf {} -sip {} -dtsn {} -os {} -daug {} -en {} -dtbs {} -spd {} {}".format(
                PYTHON,RUN_FILE,OPTIM,LR,SEED,LEARNABLE,EPOCHS,LRMAX,DF,INIT,TRAIN_SAMPLE_NUM,SCHEDULER,AUGMENT,mlflow_exp_name,TRAIN_BATCH_SIZE,SCATT_PARAM_DISTANCE,DATA_ARG)

            commands.append(command)
    
    experiments_mpCommands(
        processBatchSize=PROCESS_BATCH_SIZE,
        commands=commands
    )


    import sys
    import mlflow
    sys.path.append(str(os.getcwd()))
    from parametricSN.models.models_utils import compareParams
    from parametricSN.utils.helpers import getSimplePlot

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paramsTF = torch.load(os.path.join('/tmp',"{}_{}.pt".format('Tight-Frame',mlflow_exp_name.strip("\""))))
    paramsRand = torch.load(os.path.join('/tmp',"{}_{}.pt".format('Random',mlflow_exp_name.strip("\""))))

    param_distance = []
    for i in range(len(paramsTF)):
        param_distance.append(
            compareParams(
            params1=paramsTF[i]['params'],
            angles1=paramsTF[i]['angle'], 
            params2=paramsRand[i]['params'],
            angles2=paramsRand[i]['angle'],
            device=device
            )
        )

    paramDistancePlot = getSimplePlot(xlab='Epochs', ylab='Distance',
        title='TF and Randomly intialized parameters distances from one another as they are optimized', label='Distance',
        xvalues=[x+1 for x in range(len(param_distance))], yvalues=param_distance)


    temp = str(os.path.join(os.getcwd(),'mlruns'))
    if not os.path.isdir(temp):
        os.mkdir(temp)

    mlflow.set_tracking_uri('sqlite:///' + os.path.join(temp,'store.db'))
    mlflow.set_experiment(mlflow_exp_name.strip("\""))


    with mlflow.start_run(run_name='filter comparison'):
        mlflow.log_figure(paramDistancePlot, 'learnable_parameters/param_distance.pdf')
        print(f"finished logging to {'sqlite:///' + os.path.join(temp,'store.db')}")

    os.system('rm {}'.format(os.path.join('/tmp',"{}_{}.pt".format('Tight-Frame',mlflow_exp_name))))
    os.system('rm {}'.format(os.path.join('/tmp',"{}_{}.pt".format('Random',mlflow_exp_name))))
    exit(0)











