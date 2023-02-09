train_gpu = [1,]
classes_path = 'model_data/cls_classes.txt'
input_shape = [512, 512]
backbone = "resnet50"
model_path = "logs/2st/ep025-loss0.742-val_loss5.356-acc0.670.h5"

Init_Epoch = 25
Freeze_Epoch = 1
Freeze_batch_size = 32
UnFreeze_Epoch = 100
Unfreeze_batch_size = 16
Freeze_Train = False

# Init_lr = 1e-3
Init_lr = 1e-2
Min_lr = Init_lr * 0.01

# optimizer_type = "adam"
optimizer_type = "sgd"
momentum = 0.9
lr_decay_type = 'cos'
save_period = 1
save_dir = 'logs/2st'
num_workers = 1

train_annotation_path = "cls_train.txt"
test_annotation_path = 'cls_test.txt'

Small_Train = False
val_split = 0.9

