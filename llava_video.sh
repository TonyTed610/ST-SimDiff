



export NUMEXPR_MAX_THREADS=$(cat /sys/fs/cgroup/cpu.max | awk '{print $1/$2}')
export OPENAI_API_URL="https://api.gpt.ge/v1/chat/completions"
export OPENAI_API_KEY="sk-ghf6c88k9U21Su7h331cCe4eCb8e428aB138Bc64FdAeA682"

# ours
for cost in 0.3 ; do
for event_upper_bound in 0.2; do
    for similarity_lower_bound in 0.8; do
        for TASK in videomme; do
            python -m accelerate.commands.launch \
                --num_processes=2 \
                -m lmms_eval \
                --model llava_video \
                --model_args pretrained=/root/autodl-tmp/model/llava-video,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=64,cost=$cost,similarity_lower_bound=$similarity_lower_bound,event_upper_bound=$event_upper_bound,merge_type=new_topk,right=True,bottom=True,spatial=True,temporal=True,strategy=3,mm_spatial_pool_mode=bilinear\
                --tasks $TASK \
                --batch_size 1 \
                --output_path ./logs/ \
                # --limit 600
        done 
    done
done
done 



# st_topk
# for cost in 0.3; do
    # for TASK in videomme longvideobench_val_v; do
    #     python -m accelerate.commands.launch \
    #         --num_processes=6 \
    #         -m lmms_eval \
    #         --model llava_video \
    #         --model_args pretrained=/root/autodl-tmp/model/llava-video,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=64,cost=$cost,similarity_lower_bound=0,merge_type=st_topk,right=True,bottom=True,spatial=True,temporal=True,event=True,diverse=True,strategy=3,mm_spatial_pool_mode=bilinear\
    #         --tasks $TASK \
    #         --batch_size 1 \
    #         --output_path ./logs/
    #         # --limit 60
    # done 
    # for TASK in videomme egoschema; do
    #     python -m accelerate.commands.launch \
    #         --num_processes=6 \
    #         -m lmms_eval \
    #         --model llava_video \
    #         --model_args pretrained=/root/autodl-tmp/model/llava-video,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=64,cost=$cost,similarity_lower_bound=0,merge_type=st_topk,right=True,bottom=True,spatial=True,temporal=True,event=True,diverse=False,strategy=3,mm_spatial_pool_mode=bilinear\
    #         --tasks $TASK \
    #         --batch_size 1 \
    #         --output_path ./logs/
    #         # --limit 60
    # done 
    # for TASK in egoschema; do
    #     python -m accelerate.commands.launch \
    #         --num_processes=6 \
    #         -m lmms_eval \
    #         --model llava_video \
    #         --model_args pretrained=/root/autodl-tmp/model/llava-video,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=64,cost=$cost,similarity_lower_bound=0,merge_type=st_topk,right=True,bottom=True,spatial=True,temporal=True,event=False,diverse=True,strategy=3,mm_spatial_pool_mode=bilinear\
    #         --tasks $TASK \
    #         --batch_size 1 \
    #         --output_path ./logs/
    #         # --limit 60
    # done 
# done 


# for cost in 0.3 0.5; do
#     for TASK in longvideobench_val_v videomme; do
#         python -m accelerate.commands.launch \
#             --num_processes=6 \
#             -m lmms_eval \
#             --model llava_video \
#             --model_args pretrained=/root/autodl-tmp/model/llava-video,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=64,cost=$cost,similarity_lower_bound=0.7,merge_type=new_topk,right=True,bottom=False,spatial=True,temporal=True,strategy=3,mm_spatial_pool_mode=bilinear\
#             --tasks $TASK \
#             --batch_size 1 \
#             --output_path ./logs/ 
#     done 
# done 

# for cost in 0.3 0.5; do
#     for TASK in videomme longvideobench_val_v; do
#         python -m accelerate.commands.launch \
#             --num_processes=6 \
#             -m lmms_eval \
#             --model llava_video \
#             --model_args pretrained=/root/autodl-tmp/model/llava-video,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=64,cost=0.3,similarity_lower_bound=0.6,merge_type=new_topk,right=True,bottom=True,spatial=False,temporal=True,strategy=3,mm_spatial_pool_mode=bilinear\
#             --tasks $TASK \
#             --batch_size 1 \
#             --log_samples \
#             --log_samples_suffix llava_video_$TASK \
#             --output_path ./logs/ 
#     done 
# done 
#           --model_args pretrained=../model/llava-video,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=64,cost=$cost,similarity_lower_bound=0.7,merge_type=new_topk,right=True,bottom=True,strategy=3,mm_spatial_pool_mode=bilinear\
#--model_args pretrained=../model/llava-video,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=64,cost=$cost,similarity_lower_bound=0.6,merge_type=org,mm_spatial_pool_mode=bilinear\

# python -m accelerate.commands.launch \
#     --num_processes=4 \
#     -m lmms_eval \
#     --model llava_video \
#     --model_args pretrained=../model/llava-video,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=64\
#     --prepare_config "num_frames=64, pool_func=L2NormAvgPool2d, kernel_size=3, stride=3, temp=1, p=2, grid_size_list=(4, 4, 4), grid_freq_list=(2, 4, 8)" \
#     --tasks $TASK \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_video_$TASK \
#     --output_path ./logs/ 

# python -m accelerate.commands.launch \
#     --num_processes=4 \
#     -m lmms_eval \
#     --model llava_video \
#     --model_args pretrained=../model/llava-video,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=50\
#     --prepare_config "num_frames=50, pool_func=AdaptiveAvgPool2d, output_size=8|num_frames=6, grid_size=6" \
#     --tasks $TASK \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_video_$TASK \
#     --output_path ./logs/
# python -m accelerate.commands.launch \
#     --num_processes=4 \
#     -m lmms_eval \
#     --model llava_video \
#     --model_args pretrained=../model/llava-video,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=6\
#     --prepare_config "num_frames=6, grid_size=6" \
#     --tasks $TASK \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_video_$TASK \
#     --output_path ./logs/
# python -m accelerate.commands.launch \
#     --num_processes=4 \
#     -m lmms_eval \
#     --model llava_video \
#     --model_args pretrained=../model/llava-video,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=6\
#     --prepare_config "num_frames=6, grid_size=6" \
#     --tasks $TASK \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_video_$TASK \
#     --output_path ./logs/

# python -m accelerate.commands.launch \
#     --num_processes=4 \
#     -m lmms_eval \
#     --model llava_video \
#     --model_args pretrained=../model/llava-video,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=64\
#     --prepare_config "num_frames=64, pool_func=L2NormAvgPool2d, kernel_size=3, stride=3, temp=1, p=2, grid_size_list=(4, 4, 4), grid_freq_list=(2, 4, 8)" \
#     --tasks $TASK \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_video_$TASK \
#     --output_path ./logs/
# python -m accelerate.commands.launch \
#     --num_processes=4 \
#     -m lmms_eval \
#     --model llava_video \
#     --model_args pretrained=../model/llava-video,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=132\
#     --prepare_config "num_frames=132, pool_func=L2NormAvgPool2d, kernel_size=3, stride=3, temp=1, p=2, grid_size_list=(4, 4, 4), grid_freq_list=(2, 4, 8)" \
#     --tasks $TASK \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_video_$TASK \
#     --output_path ./logs/
# python -m accelerate.commands.launch \
#     --num_processes=4 \
#     -m lmms_eval \
#     --model llava_video \
#     --model_args pretrained=../model/llava-video,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=16\
#     --prepare_config "num_frames=16, pool_func=L2NormAvgPool2d, kernel_size=4, stride=4, temp=1, p=2, grid_size_list=(4, 4, 4), grid_freq_list=(2, 4, 8)" \
#     --tasks $TASK \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_video_$TASK \
#     --output_path ./logs/