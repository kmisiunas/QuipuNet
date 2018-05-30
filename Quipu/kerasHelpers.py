# Project Quipu - helper methods


from keras.callbacks import TensorBoard, History
import glob


# Tensorflow Board 

def resetHistory():
    "Prepares history and tensorboard"
    import os, subprocess
    global tensorboard, history, epoch
    os.system("fuser 6006/tcp -k")
    # log files
    logs = glob.glob("/home/kmisiunas/Logs/keras/*")
    for f in logs: 
        os.remove(f)
    subprocess.Popen(["tensorboard","--logdir=/home/kmisiunas/Logs/keras"])
    tensorboard = TensorBoard(log_dir='/home/kmisiunas/Logs/keras', write_graph=False, write_grads=False)
    history = History()
    epoch = 1
    return tensorboard, history
    
def nextEpochNo():
    try: 
        return history.epoch[-1]
    except:
        return 0
