### July 7th ###
# Train logistic-full
#python train.py --feature_names='mean_rgb,mean_audio' --feature_sizes='1024,128' --train_data_pattern=../data/video/full/train*.tfrecord --train_dir ../models/video/full/logistic_model --model=LogisticModel --start_new_model --num_epochs 50

# Train moe-full
#python train.py --feature_names='mean_rgb,mean_audio' --feature_sizes='1024,128' --train_data_pattern=../data/video/full/train*.tfrecord --train_dir ../models/video/full/moe_model --model=MoeModel --start_new_model --num_epochs 50

# Train framelevellogistic-full
python train.py --frame_features --model=FrameLevelLogisticModel --feature_names='rgb,audio' --feature_sizes='1024,128' --train_data_pattern=../data/frame/big/train*.tfrecord --train_dir ../models/frame/full/logistic_model --start_new_model --num_epochs 50

### July 6th ###
# Train moe-small
#python train.py --feature_names='mean_rgb,mean_audio' --feature_sizes='1024,128' --train_data_pattern=../data/video/small/train*.tfrecord --train_dir ../models/video/moe_model --model=MoeModel --start_new_model --num_epochs 50

# Train framelevellogistic-small
#python train.py --frame_features --model=FrameLevelLogisticModel --feature_names='rgb,audio' --feature_sizes='1024,128' --train_data_pattern=../data/frame/small/train*.tfrecord --train_dir ../models/frame/logistic_model --start_new_model --num_epochs 50

# Train lstm-small
#python train.py --frame_features --model=LstmModel --feature_names='rgb,audio' --feature_sizes='1024,128' --train_data_pattern=../data/frame/small/train*.tfrecord --train_dir ../models/frame/lstm_model --start_new_model --num_epochs 50

# Train dbof-small
#python train.py --frame_features --model=DbofModel --feature_names='rgb,audio' --feature_sizes='1024,128' --train_data_pattern=../data/frame/small/train*.tfrecord --train_dir ../models/frame/dbof_model --start_new_model --num_epochs 50

# Evaluate framelevellogistic-small
#python eval.py --eval_data_pattern=../data/frame/small/validate*.tfrecord --train_dir ../models/frame/logistic_model --run_once=True

#Evaluate dbof-small
#python eval.py --eval_data_pattern=../data/frame/small/validate*.tfrecord --train_dir ../models/frame/dbof_model --run_once=True
