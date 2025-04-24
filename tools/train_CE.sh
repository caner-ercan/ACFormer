
# python -m torch.distributed.launch --nproc_per_node=1 /rsrch5/home/trans_mol_path/cercan/data/acformer/checkpoints/lizard/ACFormer_Lizard_finetune_250424.py --launcher pytorch ${@:3} --no-validate

python /rsrch5/home/trans_mol_path/cercan/code/ACFormer/tools/train.py --config /rsrch5/home/trans_mol_path/cercan/data/acformer/checkpoints/lizard/ACFormer_Lizard_finetune_250424.py