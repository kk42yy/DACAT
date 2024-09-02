newp=".../Cholec80/data/frames_1fps/"
videop=".../cholec80/videos/video"

for mod in $(seq 1 1 80)
do
    curvideop=$newp"$(printf "%02d" $mod)"
    mkdir $curvideop
    ffmpeg -hide_banner -i $videop"$(printf "%02d" $mod)".mp4 -r 1 -start_number 0 $curvideop/%08d.jpg
done