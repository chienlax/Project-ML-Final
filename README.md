# Requirements
- python3 -m pipreqs.pipreqs . --force

# Install
- pyinstaller --onefile --hidden-import k_means_constrained --hidden-import ortools ./build/main.py
# or
- pyinstaller main.spec

# Params
- timeslot: Số ca thi (default: 12)
- num_avai_room: Số phòng thi (tối đa) khả dụng (default: 40)
- student_per_room: số sinh viên 1 phòng (default: 40)
- min_student_per_room: Số sv tối thiểu/phòng (default: 10)
- input_dir: path folder input
- output_dir: path folder output
- exam: Đợt thi (vd: 2024_1)
- location: Địa điểm (vd: HN, HCM, ...)
- n_jobs

