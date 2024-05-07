list_dir=(15000 21000 30000 40000 50000 80000)
# list_dir=(80000)
replica_scenes=('office_0_with_mlp_3dcl_loss_rgb_grads' 'office_2_with_mlp_3dcl_loss_rgb_grads' 'office_3_with_mlp_3dcl_loss_rgb_grads' 'office_4_with_mlp_3dcl_loss_rgb_grads')
for j in "${replica_scenes[@]}"; do
    for i in "${list_dir[@]}"; do
        my_variable=$(echo "$j" | awk -F '_with' '{print $1}')
        # cd /data/jaswanth/OM-Gaussian-Splatting-org
        # python render.py -m "output/messy_room/$j" --max_objects 12 --dataset messy_room --iteration "$i" --skip_train
        cd /data/jaswanth/Contrastive-Lift
        python /data/jaswanth/Contrastive-Lift/inference/evaluate.py --root_path "/data/jaswanth/OM-Gaussian-Splatting-org/data/replica/$my_variable" --exp_path "/data/jaswanth/OM-Gaussian-Splatting-org/output/replica/$j/test/ours_$i" 
        cd /data/jaswanth/OM-Gaussian-Splatting-org
    done
done
replica_scenes=('room_1_with_mlp_3dcl_loss_rgb_grads' 'room_0_with_mlp_3dcl_loss_rgb_grads' 'room_2_with_mlp_3dcl_loss_rgb_grads')
for j in "${replica_scenes[@]}"; do
    for i in "${list_dir[@]}"; do
        my_variable=$(echo "$j" | awk -F '_with' '{print $1}')
        # cd /data/jaswanth/OM-Gaussian-Splatting-org
        # python render.py -m "output/messy_room/$j" --max_objects 12 --dataset messy_room --iteration "$i" --skip_train
        cd /data/jaswanth/Contrastive-Lift
        python /data/jaswanth/Contrastive-Lift/inference/evaluate.py --root_path "/data/jaswanth/OM-Gaussian-Splatting-org/data/replica/$my_variable" --exp_path "/data/jaswanth/OM-Gaussian-Splatting-org/output/replica/$j/test/ours_$i" 
        cd /data/jaswanth/OM-Gaussian-Splatting-org
    done
done