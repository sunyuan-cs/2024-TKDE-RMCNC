# # 0-Scene15        
# data=0
# tau=0.5
# gpu=1
# q=0.5
# lr=0.001
# epoch=120

# 1-Caltech101      
# data=1
# tau=1
# gpu=1
# q=2
# lr=0.001
# epoch=120

# # 3-NoisyMNIST
# data=3
# tau=0.5
# gpu=0
# q=0.5    
# lr=0.001
# epoch=120


# 5-DeepAnimal
data=5
tau=0.5
gpu=1
q=2
lr=0.0004
epoch=120

# # 7-wiki_2_view       
# data=7
# tau=2
# gpu=1
# q=0.5
# lr=0.001
# epoch=120



# 9-nuswide_deep_2_view       
# data=9
# tau=1
# gpu=1
# q=0.5
# lr=0.001
# epoch=120

# 10-xmedia_deep_2_view       
# data=10
# tau=0.5
# gpu=0
# q=0.5
# lr=0.001
# epoch=120

# 11-xrmb_2_view       
# data=11
# tau=1
# gpu=0
# q=2
# lr=0.001
# epoch=120

for align_rate in "0.5"  "1" 
do 
    echo "===================================================================================="
    echo "================================alignment rate:" $align_rate "================================"
    echo "===================================================================================="
    for i in "1" "2" "3" "4" "5"
    do
        python run_RMCNC.py --data $data --tau $tau --gpu $gpu -ap $align_rate --method pa --e $epoch -li $epoch --q $q -lr $lr
    done
done



