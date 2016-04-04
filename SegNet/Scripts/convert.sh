cd /home/ubuntu/data/SegNet/ISBI/test_ISBI/final_output/
for file in *.png; do
	convert -type Grayscale "$file" "$file.2.png"
done
