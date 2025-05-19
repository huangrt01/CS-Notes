
nsys profile --duration=30 --stats=true --force-overwrite=true --export=json --output=nsys_profile_results \
--trace=cuda,nvtx,osrt,cudnn,cublas ./output 