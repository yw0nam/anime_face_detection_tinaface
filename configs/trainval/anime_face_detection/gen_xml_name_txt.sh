
ls -l ./data/anime_face_detector/train/Annotations/ | grep "^-" | awk '{print $9}' | cut -d '.' -f 1 > data/anime_face_detector/train/train.txt
ls -l ./data/anime_face_detector/val/Annotations/ | grep "^-" | awk '{print $9}' | cut -d '.' -f 1 > data/anime_face_detector/val/val.txt
