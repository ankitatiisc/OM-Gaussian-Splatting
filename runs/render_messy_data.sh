list_dir=(15000 21000 30000 40000 50000)
replica_scenes=('large_corridor_25_with_mlp_triplet_loss_3d_los_rgb_grads' 'large_corridor_50_with_mlp_triplet_loss_3d_los_rgb_grads' 'large_corridor_100_with_mlp_triplet_loss_3d_los_rgb_grads' 'large_corridor_500_with_mlp_triplet_loss_3d_los_rgb_grads')

for j in "${replica_scenes[@]}"; do
    for i in "${list_dir[@]}"; do
        my_variable=$(echo "$j" | awk -F '_with' '{print $1}')
        cd /data/jaswanth/OM-Gaussian-Splatting-org
        python render.py -m "output/messy_room/$j" --max_objects 12 --dataset messy_room --iteration "$i" --skip_train
        cd /data/jaswanth/Contrastive-Lift
        python /data/jaswanth/Contrastive-Lift/inference/evaluate.py --root_path "/data/jaswanth/OM-Gaussian-Splatting-org/data/messy_room/$my_variable" --exp_path "/data/jaswanth/OM-Gaussian-Splatting-org/output/messy_room/$j/test/ours_$i" --MOS
        cd /data/jaswanth/OM-Gaussian-Splatting-org
    done
done