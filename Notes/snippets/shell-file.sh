*** 查找大文件

sudo find / \( -path /root/rec -prune \) -o \( -type f -size +100M -exec ls -lh {} \; \) | awk '{ print $9 ": " $5 }'


*** 查找并remove文件

find /root/rec -mindepth 1 -type d -mmin +120 -exec rm -rf {} +

# 注意加号，是为了一次rm命令rm多个文件，而不是循环


*** 多条件查找

# 在多个目录(/path1, /path2)中，查找满足条件A(-name "*A*")或条件B(-name "*B*")的文件，并忽略错误输出
find /path1 /path2 -name "*A*" -o -name "*B*" 2>/dev/null


*** 文件传输

xxd -p myfile > tmp
xxd -r -p tmp > myfile


alias pull='rsync -aPv -e "ssh -p {port}" root@{ip_addr}:{src_dir} {dst_dir} --exclude "ckpts" --exclude "hstu_model" --exclude "dataio" --exclude ".idea" --exclude ".git"'
alias push='rsync -aPv -e "ssh -p {port}" {src_dir} root@{ip_addr}:{dst_dir} --exclude "ckpts" --exclude "hstu_model" --exclude "dataio" --exclude ".idea" --exclude ".git"'